"""Reproducible GFN2-xTB molecular descriptors for curated structures."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_ENERGY_RE = re.compile(rf"\|\s*TOTAL ENERGY\s+({_FLOAT})\s*Eh", re.IGNORECASE)
_GAP_RE = re.compile(
    rf"\|\s*HOMO-LUMO GAP\s+({_FLOAT})\s*eV", re.IGNORECASE
)
_DIPOLE_RE = re.compile(
    rf"molecular dipole:.*?full:\s*({_FLOAT})\s+({_FLOAT})\s+"
    rf"({_FLOAT})\s+({_FLOAT})",
    re.IGNORECASE | re.DOTALL,
)
_POLARIZABILITY_RE = re.compile(
    rf"Mol\.\s*(?:α|alpha)\(0\)\s*/au\s*:\s*({_FLOAT})",
    re.IGNORECASE,
)
_MOLECULE_ID_RE = re.compile(r"^herg_karim_\d{6}$")
_OPTIMIZATION_LEVELS = {
    "crude",
    "sloppy",
    "loose",
    "normal",
    "tight",
    "verytight",
    "extreme",
}


class QMCalculationError(RuntimeError):
    """A molecule-specific conformer or xTB calculation failed."""


@dataclass(frozen=True)
class XTBRunSettings:
    """Inputs that determine the reproducible xTB calculation and cache key."""

    executable: Path
    seed: int
    optimization_level: str
    timeout_seconds: int
    xtb_version: str
    executable_sha256: str
    gfn2_parameters_sha256: str

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.optimization_level not in _OPTIMIZATION_LEVELS:
            raise ValueError("unsupported xTB geometry optimization level")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not self.xtb_version or not self.executable_sha256:
            raise ValueError("xTB version and executable checksum are required")
        if not self.gfn2_parameters_sha256:
            raise ValueError("GFN2-xTB parameter checksum is required")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_xtb_version(executable: Path) -> str:
    """Run the selected xTB executable and return its reported version line."""
    exe = executable.resolve()
    if not exe.is_file():
        raise FileNotFoundError(f"xTB executable does not exist: {exe}")
    environment = _xtb_environment(exe)
    try:
        result = subprocess.run(
            [str(exe), "--version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise QMCalculationError("xTB version probe timed out") from exc
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        raise QMCalculationError(
            f"xTB version probe exited {result.returncode}: {output[-2000:]}"
        )
    match = re.search(r"\bxtb version\s+([^\s(]+)(?:\s+\(([^)]+)\))?", output, re.I)
    if match is None:
        raise QMCalculationError("xTB did not report a recognizable version")
    commit = f" ({match.group(2)})" if match.group(2) else ""
    return f"{match.group(1)}{commit}"


def parse_xtb_output(output: str, heavy_atom_count: int) -> dict[str, float]:
    """Parse final GFN2-xTB energies and molecular properties from stdout."""
    if heavy_atom_count <= 0:
        raise ValueError("heavy_atom_count must be positive")

    energies = _ENERGY_RE.findall(output)
    gaps = _GAP_RE.findall(output)
    dipoles = list(_DIPOLE_RE.finditer(output))
    polarizabilities = _POLARIZABILITY_RE.findall(output)
    orbitals: dict[str, list[float]] = {"HOMO": [], "LUMO": []}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[-1].upper() not in {"(HOMO)", "(LUMO)"}:
            continue
        numeric_fields = [
            field
            for field in fields[:-1]
            if re.fullmatch(_FLOAT, field) is not None
        ]
        if numeric_fields:
            marker = fields[-1][1:-1].upper()
            orbitals[marker].append(_to_float(numeric_fields[-1]))

    if not energies or not gaps or not dipoles or not polarizabilities:
        raise QMCalculationError(
            "xTB output is missing a required energy, gap, dipole, or polarizability"
        )
    if not orbitals["HOMO"] or not orbitals["LUMO"]:
        raise QMCalculationError("xTB output is missing a HOMO or LUMO orbital energy")

    total_energy = _to_float(energies[-1])
    gap = _to_float(gaps[-1])
    homo = orbitals["HOMO"][-1]
    lumo = orbitals["LUMO"][-1]
    dipole = _to_float(dipoles[-1].group(4))
    polarizability = _to_float(polarizabilities[-1])
    if abs((lumo - homo) - gap) > 0.02:
        raise QMCalculationError(
            "reported HOMO, LUMO, and HOMO-LUMO gap are inconsistent"
        )

    values = {
        "total_energy_hartree": total_energy,
        "total_energy_hartree_per_heavy_atom": total_energy / heavy_atom_count,
        "homo_ev": homo,
        "lumo_ev": lumo,
        "homo_lumo_gap_ev": gap,
        "molecular_dipole_debye": dipole,
        "molecular_polarizability_au": polarizability,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise QMCalculationError("xTB descriptors contain non-finite values")
    return values


def build_initial_geometry(
    molecule_id: str,
    smiles: str,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    """Generate one deterministic ETKDGv3 conformer without changing chemistry."""
    if not _MOLECULE_ID_RE.fullmatch(molecule_id):
        raise ValueError(f"unexpected molecule ID: {molecule_id}")
    if not smiles or not smiles.strip():
        raise ValueError(f"{molecule_id}: SMILES must be non-empty")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise QMCalculationError(f"{molecule_id}: RDKit could not parse curated SMILES")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise QMCalculationError(f"{molecule_id}: curated structure has multiple fragments")
    input_canonical_smiles = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    formal_charge = int(Chem.GetFormalCharge(molecule))
    heavy_atom_count = int(molecule.GetNumHeavyAtoms())
    geometry_seed = _geometry_seed(seed, molecule_id, smiles)

    hydrogenated = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = geometry_seed
    parameters.numThreads = 1
    conformer_id = AllChem.EmbedMolecule(hydrogenated, parameters)
    if conformer_id < 0:
        raise QMCalculationError(f"{molecule_id}: ETKDGv3 conformer generation failed")
    xyz = Chem.MolToXYZBlock(hydrogenated, confId=conformer_id)
    metadata = {
        "input_canonical_isomeric_smiles": input_canonical_smiles,
        "formal_charge": formal_charge,
        "heavy_atom_count": heavy_atom_count,
        "atom_count_with_hydrogens": int(hydrogenated.GetNumAtoms()),
        "geometry_method": "RDKit ETKDGv3",
        "geometry_seed": geometry_seed,
        "conformer_count": 1,
        "additional_protonation_or_tautomerization": False,
    }
    return xyz, metadata


def calculate_xtb_descriptors(
    molecule_id: str,
    smiles: str,
    settings: XTBRunSettings,
) -> dict[str, Any]:
    """Calculate descriptors after one conformer and a GFN2-xTB optimization."""
    executable = settings.executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"xTB executable does not exist: {executable}")
    parameter_file = executable.parent.parent / "share" / "xtb" / "param_gfn2-xtb.txt"
    if not parameter_file.is_file():
        raise FileNotFoundError(f"GFN2-xTB parameter file does not exist: {parameter_file}")

    xyz, metadata = build_initial_geometry(molecule_id, smiles, settings.seed)
    environment = _xtb_environment(executable)
    command = [
        str(executable),
        "input.xyz",
        "--gfn",
        "2",
        "--opt",
        settings.optimization_level,
        "--chrg",
        str(metadata["formal_charge"]),
    ]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"qm_qrc_{molecule_id}_") as temporary:
        working_directory = Path(temporary)
        (working_directory / "input.xyz").write_text(xyz, encoding="utf-8")
        try:
            completed = subprocess.run(
                command,
                cwd=working_directory,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=settings.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise QMCalculationError(
                f"{molecule_id}: xTB timed out after {settings.timeout_seconds} seconds"
            ) from exc
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout)[-2000:]
            raise QMCalculationError(
                f"{molecule_id}: xTB exited {completed.returncode}: {details}"
            )

        optimized_xyz_path = working_directory / "xtbopt.xyz"
        if not optimized_xyz_path.is_file():
            raise QMCalculationError(f"{molecule_id}: xTB did not write xtbopt.xyz")
        optimized_xyz = optimized_xyz_path.read_text(encoding="utf-8")
        optimized_atom_count = _xyz_atom_count(optimized_xyz, molecule_id)
        if optimized_atom_count != metadata["atom_count_with_hydrogens"]:
            raise QMCalculationError(
                f"{molecule_id}: optimized geometry atom count changed "
                f"({optimized_atom_count} != {metadata['atom_count_with_hydrogens']})"
            )

        descriptors = parse_xtb_output(
            completed.stdout, int(metadata["heavy_atom_count"])
        )
        runtime_seconds = time.perf_counter() - started

    return {
        **metadata,
        "descriptors": descriptors,
        "optimized_geometry_xyz": optimized_xyz,
        "optimized_geometry_sha256": hashlib.sha256(
            optimized_xyz.encode("utf-8")
        ).hexdigest(),
        "calculation_seconds": runtime_seconds,
        "xtb_stdout": completed.stdout,
        "xtb_stderr": completed.stderr,
    }


def get_or_calculate_xtb_descriptors(
    molecule_id: str,
    smiles: str,
    *,
    settings: XTBRunSettings,
    cache_dir: Path,
) -> tuple[dict[str, Any], bool]:
    """Reuse only an exact ID, structure, software, and method cache match."""
    structure_sha256 = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
    cache_identity = {
        "descriptor_schema_version": "gfn2-xtb-descriptors-v1",
        "molecule_id": molecule_id,
        "smiles_sha256": structure_sha256,
        "rdkit_version": rdBase.rdkitVersion,
        "xtb_version": settings.xtb_version,
        "xtb_executable_sha256": settings.executable_sha256,
        "gfn2_parameters_sha256": settings.gfn2_parameters_sha256,
        "method": "GFN2-xTB",
        "geometry_method": "ETKDGv3",
        "seed": settings.seed,
        "optimization_level": settings.optimization_level,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{molecule_id}_{cache_key}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("cache_identity") != cache_identity:
            raise ValueError(f"QM cache identity mismatch: {cache_path}")
        result = payload.get("result")
        _validate_cached_result(result, molecule_id)
        return result, True

    result = calculate_xtb_descriptors(molecule_id, smiles, settings)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_identity": cache_identity,
        "result": result,
    }
    temporary_path = cache_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(cache_path)
    return result, False


def _validate_cached_result(result: Any, molecule_id: str) -> None:
    if not isinstance(result, dict):
        raise ValueError(f"QM cache result is invalid for {molecule_id}")
    descriptors = result.get("descriptors")
    if not isinstance(descriptors, dict) or not descriptors:
        raise ValueError(f"QM cache descriptors are missing for {molecule_id}")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in descriptors.values()
    ):
        raise ValueError(f"QM cache contains non-finite descriptors for {molecule_id}")
    geometry = result.get("optimized_geometry_xyz")
    if not isinstance(geometry, str) or not geometry:
        raise ValueError(f"QM cache geometry is missing for {molecule_id}")
    if hashlib.sha256(geometry.encode("utf-8")).hexdigest() != result.get(
        "optimized_geometry_sha256"
    ):
        raise ValueError(f"QM cache geometry checksum mismatch for {molecule_id}")


def _geometry_seed(seed: int, molecule_id: str, smiles: str) -> int:
    digest = hashlib.sha256(f"{seed}:{molecule_id}:{smiles}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big") & 0x7FFFFFFF or 1


def _xyz_atom_count(xyz: str, molecule_id: str) -> int:
    first_line = xyz.splitlines()[0] if xyz.splitlines() else ""
    try:
        count = int(first_line.strip())
    except ValueError as exc:
        raise QMCalculationError(f"{molecule_id}: invalid optimized XYZ atom count") from exc
    if count <= 0:
        raise QMCalculationError(f"{molecule_id}: optimized XYZ atom count must be positive")
    return count


def _to_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _xtb_environment(executable: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["XTBPATH"] = str(executable.parent.parent / "share" / "xtb")
    environment["OMP_NUM_THREADS"] = "1"
    environment["OMP_MAX_ACTIVE_LEVELS"] = "1"
    environment["OMP_STACKSIZE"] = "2G"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    return environment
