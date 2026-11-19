"""Generate a small, train-only GFN2-xTB descriptor feasibility pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qktox.qm_descriptors import (
    QMCalculationError,
    XTBRunSettings,
    get_or_calculate_xtb_descriptors,
    get_xtb_version,
    sha256_file,
)


DESCRIPTOR_COLUMNS = (
    "total_energy_hartree",
    "total_energy_hartree_per_heavy_atom",
    "homo_ev",
    "lumo_ev",
    "homo_lumo_gap_ev",
    "molecular_dipole_debye",
    "molecular_polarizability_au",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return config


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _read_training_smiles(
    dataset_path: Path,
    split_path: Path,
    split_name: str,
    sample_size: int,
    seed: int,
) -> tuple[list[str], list[int], list[str], list[str]]:
    """Sample row positions from the frozen train partition without reading labels."""
    frozen = json.loads(split_path.read_text(encoding="utf-8-sig"))
    if split_name not in frozen:
        raise ValueError(f"Frozen split does not contain {split_name}")
    train_positions = np.asarray(frozen[split_name]["train"], dtype=np.int64)
    if train_positions.ndim != 1 or len(train_positions) < sample_size:
        raise ValueError("Frozen training partition is smaller than the requested pilot")
    if len(np.unique(train_positions)) != len(train_positions) or (train_positions < 0).any():
        raise ValueError("Frozen training positions must be unique non-negative indices")

    selected = np.sort(
        np.random.default_rng(seed).choice(train_positions, size=sample_size, replace=False)
    )
    selected_set = set(int(index) for index in selected)
    frame = pd.read_csv(
        dataset_path,
        usecols=["smiles"],
        skiprows=lambda line_number: line_number != 0
        and line_number - 1 not in selected_set,
        dtype={"smiles": str},
    )
    if len(frame) != sample_size or frame["smiles"].isna().any():
        raise ValueError("Could not read all selected training SMILES")
    molecule_ids = [f"herg_karim_{int(index):06d}" for index in selected]
    smiles = frame["smiles"].astype(str).tolist()
    checksums = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in smiles]
    return molecule_ids, selected.tolist(), smiles, checksums


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(config_path: Path) -> Path:
    config = _read_yaml(config_path)
    if config.get("dataset_key") != "herg_karim":
        raise ValueError("The QM descriptor pilot only supports herg_karim")
    if config.get("split_name") != "scaffold_r0":
        raise ValueError("The QM descriptor pilot only supports frozen scaffold_r0")
    if int(config.get("sample_size", 0)) != 32:
        raise ValueError("This feasibility script is restricted to exactly 32 molecules")
    if config.get("partition") != "train":
        raise ValueError("QM descriptors may only be generated for the training partition here")

    output_dir = ROOT / config["output_dir"]
    cache_dir = ROOT / config["cache_dir"]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite descriptor pilot output: {output_dir}")
    executable = (ROOT / config["xtb_executable"]).resolve()
    parameter_file = executable.parent.parent / "share" / "xtb" / "param_gfn2-xtb.txt"
    xtb_version = get_xtb_version(executable)
    settings = XTBRunSettings(
        executable=executable,
        seed=int(config["seed"]),
        optimization_level=str(config["optimization_level"]),
        timeout_seconds=int(config["timeout_seconds"]),
        xtb_version=xtb_version,
        executable_sha256=_sha256(executable),
        gfn2_parameters_sha256=_sha256(parameter_file),
    )
    dataset_path = ROOT / config["processed_dataset"]
    split_path = ROOT / config["frozen_split_file"]
    molecule_ids, positions, smiles, structure_checksums = _read_training_smiles(
        dataset_path,
        split_path,
        str(config["split_name"]),
        int(config["sample_size"]),
        int(config["seed"]),
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    geometry_dir = output_dir / "optimized_geometries"
    log_dir = output_dir / "runtime_logs"
    geometry_dir.mkdir()
    log_dir.mkdir()
    start = time.perf_counter()
    rows = []
    failures = []
    molecules = []

    for molecule_id, position, molecule_smiles, structure_sha256 in zip(
        molecule_ids, positions, smiles, structure_checksums
    ):
        molecule_start = time.perf_counter()
        try:
            result, cache_hit = get_or_calculate_xtb_descriptors(
                molecule_id,
                molecule_smiles,
                settings=settings,
                cache_dir=cache_dir,
            )
        except (QMCalculationError, ValueError, OSError) as exc:
            elapsed = time.perf_counter() - molecule_start
            failure = {
                "mol_id": molecule_id,
                "row_index": position,
                "smiles_sha256": structure_sha256,
                "elapsed_seconds": elapsed,
                "error": str(exc),
            }
            failures.append(failure)
            (log_dir / f"{molecule_id}.error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            continue

        geometry_path = geometry_dir / f"{molecule_id}.xyz"
        with geometry_path.open("w", encoding="utf-8", newline="") as geometry_file:
            geometry_file.write(result["optimized_geometry_xyz"])
        (log_dir / f"{molecule_id}.out").write_text(
            result["xtb_stdout"], encoding="utf-8"
        )
        if result["xtb_stderr"]:
            (log_dir / f"{molecule_id}.err").write_text(
                result["xtb_stderr"], encoding="utf-8"
            )
        descriptor_values = result["descriptors"]
        rows.append(
            {
                "mol_id": molecule_id,
                "row_index": position,
                "smiles_sha256": structure_sha256,
                "status": "cached" if cache_hit else "calculated",
                "cache_hit": cache_hit,
                "calculation_seconds": result["calculation_seconds"],
                "current_run_seconds": time.perf_counter() - molecule_start,
                "formal_charge": result["formal_charge"],
                "heavy_atom_count": result["heavy_atom_count"],
                "atom_count_with_hydrogens": result["atom_count_with_hydrogens"],
                **descriptor_values,
            }
        )
        molecules.append(
            {
                "mol_id": molecule_id,
                "row_index": position,
                "smiles_sha256": structure_sha256,
                "cache_hit": cache_hit,
                "geometry_seed": result["geometry_seed"],
                "formal_charge": result["formal_charge"],
                "heavy_atom_count": result["heavy_atom_count"],
                "atom_count_with_hydrogens": result["atom_count_with_hydrogens"],
                "optimized_geometry_sha256": result["optimized_geometry_sha256"],
            }
        )

    descriptor_columns = [
        "mol_id",
        "row_index",
        "smiles_sha256",
        "status",
        "cache_hit",
        "calculation_seconds",
        "current_run_seconds",
        "formal_charge",
        "heavy_atom_count",
        "atom_count_with_hydrogens",
        *DESCRIPTOR_COLUMNS,
    ]
    pd.DataFrame(rows, columns=descriptor_columns).to_csv(
        output_dir / "qm_descriptors.csv", index=False
    )
    pd.DataFrame(
        failures,
        columns=[
            "mol_id",
            "row_index",
            "smiles_sha256",
            "elapsed_seconds",
            "error",
        ],
    ).to_csv(output_dir / "failure_log.csv", index=False)
    elapsed_seconds = time.perf_counter() - start
    manifest = {
        "status": "completed" if not failures else "completed_with_failures",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "32-molecule train-only GFN2-xTB descriptor feasibility pilot",
        "dataset": config["dataset_key"],
        "split": config["split_name"],
        "partition": "train",
        "sample_seed": int(config["seed"]),
        "sample_size": int(config["sample_size"]),
        "row_indices": positions,
        "molecule_ids": molecule_ids,
        "structure_checksums": dict(zip(molecule_ids, structure_checksums)),
        "labels_read": False,
        "validation_or_test_data_read": False,
        "method": {
            "input_structure": "frozen curated isomeric SMILES; no additional protonation or tautomerization",
            "formal_charge": "preserved from curated SMILES and passed to xTB --chrg",
            "hydrogens": "RDKit Chem.AddHs for conformer generation and xTB input",
            "conformer_generation": "one deterministic RDKit ETKDGv3 conformer per molecule",
            "geometry_optimization": f"GFN2-xTB {settings.xtb_version}; --opt {settings.optimization_level}",
            "threads": 1,
            "descriptor_energy_normalization": "total energy retained in Hartree and normalized by heavy-atom count",
            "descriptors": list(DESCRIPTOR_COLUMNS),
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xtb": settings.xtb_version,
            "xtb_executable_sha256": settings.executable_sha256,
            "gfn2_parameter_file_sha256": settings.gfn2_parameters_sha256,
        },
        "inputs": {
            "processed_dataset": str(config["processed_dataset"]),
            "frozen_split_file": str(config["frozen_split_file"]),
            "frozen_split_file_sha256": _sha256(split_path),
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256(config_path),
        },
        "counts": {
            "requested": len(molecule_ids),
            "calculated_or_cached": len(rows),
            "failures": len(failures),
            "cache_hits": sum(bool(row["cache_hit"]) for row in rows),
        },
        "runtime_seconds": elapsed_seconds,
        "xtb_calculation_seconds": sum(
            float(row["calculation_seconds"]) for row in rows
        ),
        "molecules": molecules,
        "cache_dir": str(cache_dir.relative_to(ROOT)),
    }
    manifest["artifact_checksums"] = {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "experiment_manifest.json"
    }
    _write_json(output_dir / "experiment_manifest.json", manifest)
    print(
        f"QM descriptor pilot: {len(rows)}/{len(molecule_ids)} successful, "
        f"{len(failures)} failed; {elapsed_seconds:.1f}s; output={output_dir}"
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} xTB calculations failed; see {output_dir / 'failure_log.csv'}"
        )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "qm_descriptors_pilot.yaml",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    run(config_path)


if __name__ == "__main__":
    main()
