"""Generate train-only-sampled xTB coverage plus frozen validation/test features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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


PARTITIONS = ("train", "val", "test")
TRAIN_SIZES = (64, 128, 256, 512, 1024)
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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_coverage_plan(
    split_document: dict[str, Any],
    split_names: list[str],
    training_sizes: list[int],
    seed: int,
) -> dict[str, Any]:
    """Make deterministic nested training samples and full validation/test coverage."""
    expected_splits = {f"scaffold_r{index}" for index in range(5)}
    if set(split_names) != expected_splits or len(split_names) != len(expected_splits):
        raise ValueError("coverage requires each frozen scaffold_r0 through scaffold_r4 split")
    if not training_sizes or tuple(sorted(set(training_sizes))) != tuple(training_sizes):
        raise ValueError("training sizes must be non-empty, unique, and ascending")
    if any(size <= 0 for size in training_sizes) or seed < 0:
        raise ValueError("training sizes must be positive and seed non-negative")

    split_plans: dict[str, Any] = {}
    memberships: dict[int, set[str]] = {}
    unique_indices: set[int] = set()
    for split_offset, split_name in enumerate(split_names):
        if split_name not in split_document:
            raise ValueError(f"frozen split file does not contain {split_name}")
        source = split_document[split_name]
        partition_indices: dict[str, np.ndarray] = {}
        for partition in PARTITIONS:
            raw_values = source[partition]
            if not isinstance(raw_values, list) or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in raw_values
            ):
                raise ValueError(f"{split_name}/{partition}: row indices must be integers")
            values = np.asarray(raw_values, dtype=np.int64)
            if values.ndim != 1 or len(np.unique(values)) != len(values) or (values < 0).any():
                raise ValueError(f"{split_name}/{partition}: expected unique non-negative row indices")
            partition_indices[partition] = values
        if any(
            set(partition_indices[left].tolist()) & set(partition_indices[right].tolist())
            for index, left in enumerate(PARTITIONS)
            for right in PARTITIONS[index + 1 :]
        ):
            raise ValueError(f"{split_name}: frozen partitions overlap")
        if len(partition_indices["train"]) < max(training_sizes):
            raise ValueError(f"{split_name}: training partition is smaller than requested sample")

        rng = np.random.default_rng(seed + split_offset)
        permutation = rng.permutation(np.sort(partition_indices["train"]))
        train_samples = {
            str(size): sorted(int(value) for value in permutation[:size])
            for size in training_sizes
        }
        selected_train = set(int(value) for value in permutation[: max(training_sizes)])
        validation = sorted(int(value) for value in partition_indices["val"])
        test = sorted(int(value) for value in partition_indices["test"])
        selected_by_partition = {
            "train": sorted(selected_train),
            "val": validation,
            "test": test,
        }
        for partition, values in selected_by_partition.items():
            label = f"{split_name}:{partition}"
            for row_index in values:
                memberships.setdefault(row_index, set()).add(label)
                unique_indices.add(row_index)
        split_plans[split_name] = {
            "train_samples": train_samples,
            "val": [f"herg_karim_{value:06d}" for value in validation],
            "test": [f"herg_karim_{value:06d}" for value in test],
        }

    return {
        "dataset_key": "herg_karim",
        "sample_seed": seed,
        "training_sizes": training_sizes,
        "nested_training_samples": True,
        "split_seeds": {
            split_name: seed + offset
            for offset, split_name in enumerate(split_names)
        },
        "splits": split_plans,
        "unique_row_indices": sorted(unique_indices),
        "unique_molecule_ids": [
            f"herg_karim_{value:06d}" for value in sorted(unique_indices)
        ],
        "partition_memberships": {
            f"herg_karim_{row_index:06d}": sorted(values)
            for row_index, values in sorted(memberships.items())
        },
    }


def _read_selected_smiles(
    dataset_path: Path,
    row_indices: list[int],
) -> list[str]:
    """Read only selected curated structures; labels are never requested."""
    if not row_indices or row_indices != sorted(set(row_indices)):
        raise ValueError("row indices must be non-empty, unique, and sorted")
    selected = set(row_indices)
    frame = pd.read_csv(
        dataset_path,
        usecols=["smiles"],
        skiprows=lambda line_number: line_number != 0
        and line_number - 1 not in selected,
        dtype={"smiles": str},
    )
    if len(frame) != len(row_indices) or frame["smiles"].isna().any():
        raise ValueError("Could not read all requested curated SMILES")
    return frame["smiles"].astype(str).tolist()


def run(config_path: Path, *, resume: bool = False) -> Path:
    config_path = config_path.resolve()
    config = _read_yaml(config_path)
    if config.get("dataset_key") != "herg_karim":
        raise ValueError("QM descriptor coverage only supports herg_karim")
    if tuple(config.get("training_sizes", ())) != TRAIN_SIZES:
        raise ValueError(f"training_sizes must be exactly {list(TRAIN_SIZES)}")

    split_path = ROOT / config["frozen_split_file"]
    split_document = json.loads(split_path.read_text(encoding="utf-8-sig"))
    split_names = [str(value) for value in config["split_names"]]
    plan = build_coverage_plan(
        split_document,
        split_names,
        list(TRAIN_SIZES),
        int(config["sample_seed"]),
    )
    executable = (ROOT / config["xtb_executable"]).resolve()
    parameter_file = executable.parent.parent / "share" / "xtb" / "param_gfn2-xtb.txt"
    settings = XTBRunSettings(
        executable=executable,
        seed=int(config["geometry_seed"]),
        optimization_level=str(config["optimization_level"]),
        timeout_seconds=int(config["timeout_seconds"]),
        xtb_version=get_xtb_version(executable),
        executable_sha256=sha256_file(executable),
        gfn2_parameters_sha256=sha256_file(parameter_file),
    )
    row_indices = plan["unique_row_indices"]
    smiles_values = _read_selected_smiles(ROOT / config["processed_dataset"], row_indices)
    molecule_ids = [f"herg_karim_{index:06d}" for index in row_indices]
    checksums = [
        hashlib.sha256(smiles.encode("utf-8")).hexdigest() for smiles in smiles_values
    ]
    run_state = {
        "coverage_plan": plan,
        "configuration_sha256": sha256_file(config_path),
        "frozen_split_sha256": sha256_file(split_path),
        "structure_checksums": dict(zip(molecule_ids, checksums)),
        "method": {
            "geometry_seed": settings.seed,
            "optimization_level": settings.optimization_level,
            "xtb_version": settings.xtb_version,
            "xtb_executable_sha256": settings.executable_sha256,
            "gfn2_parameters_sha256": settings.gfn2_parameters_sha256,
        },
    }
    output_dir = ROOT / config["output_dir"]
    plan_path = output_dir / "coverage_plan.json"
    manifest_path = output_dir / "experiment_manifest.json"
    if resume and not output_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing descriptor output: {output_dir}")
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite descriptor output: {output_dir}")
        if manifest_path.exists():
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous_manifest.get("status") == "complete":
                raise FileExistsError(
                    f"Completed descriptor output already exists: {output_dir}"
                )
        if not plan_path.is_file() or json.loads(
            plan_path.read_text(encoding="utf-8")
        ) != run_state:
            raise ValueError("Cannot resume: saved run identity differs from this run")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(plan_path, run_state)
    (output_dir / "optimized_geometries").mkdir(exist_ok=True)
    (output_dir / "runtime_logs").mkdir(exist_ok=True)
    memberships = plan["partition_memberships"]

    started = time.perf_counter()
    rows = []
    molecules = []
    failures = []
    cache_hits = 0

    def write_progress(completed: int) -> None:
        if completed % 100 != 0 and completed != len(molecule_ids):
            return
        _write_json(
            output_dir / "progress.json",
            {
                "completed": completed,
                "requested": len(molecule_ids),
                "successful_or_cached": len(rows),
                "failures": len(failures),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        print(
            f"QM coverage: {completed}/{len(molecule_ids)} processed; "
            f"{len(failures)} failures",
            flush=True,
        )

    for offset, (molecule_id, row_index, smiles, checksum) in enumerate(
        zip(molecule_ids, row_indices, smiles_values, checksums), start=1
    ):
        molecule_started = time.perf_counter()
        try:
            result, cache_hit = get_or_calculate_xtb_descriptors(
                molecule_id,
                smiles,
                settings=settings,
                cache_dir=ROOT / config["cache_dir"],
            )
        except (QMCalculationError, ValueError, OSError) as exc:
            elapsed = time.perf_counter() - molecule_started
            failures.append(
                {
                    "mol_id": molecule_id,
                    "row_index": row_index,
                    "smiles_sha256": checksum,
                    "elapsed_seconds": elapsed,
                    "error": str(exc),
                }
            )
            (output_dir / "runtime_logs" / f"{molecule_id}.error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            write_progress(offset)
            continue

        error_path = output_dir / "runtime_logs" / f"{molecule_id}.error.txt"
        if error_path.exists():
            error_path.unlink()
        geometry_path = output_dir / "optimized_geometries" / f"{molecule_id}.xyz"
        with geometry_path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(result["optimized_geometry_xyz"])
        (output_dir / "runtime_logs" / f"{molecule_id}.out").write_text(
            result["xtb_stdout"], encoding="utf-8"
        )
        if result["xtb_stderr"]:
            (output_dir / "runtime_logs" / f"{molecule_id}.err").write_text(
                result["xtb_stderr"], encoding="utf-8"
            )
        cache_hits += int(cache_hit)
        rows.append(
            {
                "mol_id": molecule_id,
                "row_index": row_index,
                "smiles_sha256": checksum,
                "partition_memberships": json.dumps(memberships[molecule_id]),
                "status": "cached" if cache_hit else "calculated",
                "cache_hit": cache_hit,
                "calculation_seconds": result["calculation_seconds"],
                "formal_charge": result["formal_charge"],
                "heavy_atom_count": result["heavy_atom_count"],
                "atom_count_with_hydrogens": result["atom_count_with_hydrogens"],
                **result["descriptors"],
            }
        )
        molecules.append(
            {
                "mol_id": molecule_id,
                "row_index": row_index,
                "smiles_sha256": checksum,
                "optimized_geometry_sha256": result["optimized_geometry_sha256"],
                "cache_hit": cache_hit,
            }
        )
        write_progress(offset)

    elapsed_seconds = time.perf_counter() - started
    descriptor_frame = pd.DataFrame(
        rows,
        columns=[
            "mol_id",
            "row_index",
            "smiles_sha256",
            "partition_memberships",
            "status",
            "cache_hit",
            "calculation_seconds",
            "formal_charge",
            "heavy_atom_count",
            "atom_count_with_hydrogens",
            *DESCRIPTOR_COLUMNS,
        ],
    )
    descriptor_frame.to_csv(output_dir / "qm_descriptors.csv", index=False)
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
    runtime_packages = ("numpy", "pandas", "PyYAML", "rdkit")
    manifest = {
        "experiment": "nested low-data QM descriptor coverage, features only",
        "dataset": "herg_karim",
        "status": "failed" if failures else "complete",
        "labels_loaded": False,
        "training_sizes": list(TRAIN_SIZES),
        "requested_molecules": len(molecule_ids),
        "successful_or_cached": len(rows),
        "failures": len(failures),
        "cache_hits": cache_hits,
        "runtime_seconds": elapsed_seconds,
        "sample_seed": int(config["sample_seed"]),
        "geometry_seed": settings.seed,
        "method": {
            "software": "GFN2-xTB",
            "version": settings.xtb_version,
            "optimization_level": settings.optimization_level,
            "single_threaded": True,
            "geometry_generation": "RDKit ETKDGv3",
            "labels_or_test_features_used_for_selection": False,
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "rdkit": rdBase.rdkitVersion,
            **{
                package: importlib.metadata.version(package)
                for package in runtime_packages
            },
        },
        "input_checksums": {
            "configuration": sha256_file(config_path),
            "frozen_splits": sha256_file(split_path),
            "xtb_executable": settings.executable_sha256,
            "gfn2_parameters": settings.gfn2_parameters_sha256,
        },
        "coverage_plan": plan,
        "run_identity": run_state,
        "molecules": molecules,
        "cache_dir": str(config["cache_dir"]),
    }
    manifest["artifact_checksums"] = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "experiment_manifest.json"
    }
    _write_json(manifest_path, manifest)
    print(
        f"QM coverage: {len(rows)}/{len(molecule_ids)} successful, "
        f"{len(failures)} failed; {elapsed_seconds / 3600:.2f}h; output={output_dir}",
        flush=True,
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
        default=ROOT / "configs" / "qm_descriptor_coverage_1024.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    run(config_path, resume=args.resume)


if __name__ == "__main__":
    main()
