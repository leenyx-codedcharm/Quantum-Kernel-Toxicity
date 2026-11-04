"""Reproducible molecular features for the frozen hERG_Karim dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, Lipinski, inchi, rdFingerprintGenerator
from rdkit.Chem.rdchem import Mol

from .curate import standardise

LOGGER = logging.getLogger(__name__)
DATASET_KEY = "herg_karim"
SPLIT_NAMES = tuple(f"scaffold_r{repeat}" for repeat in range(5))
RAW_COLUMNS = {"Drug_ID", "Drug", "Y"}
PROCESSED_COLUMNS = {"orig_id", "smiles", "inchikey", "parent_key", "label"}
DESCRIPTOR_FUNCTIONS = (
    ("MolWt", Descriptors.MolWt),
    ("ExactMolWt", Descriptors.ExactMolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Lipinski.NumHDonors),
    ("NumHAcceptors", Lipinski.NumHAcceptors),
    ("NumRotatableBonds", Lipinski.NumRotatableBonds),
    ("HeavyAtomCount", lambda molecule: molecule.GetNumHeavyAtoms()),
    ("RingCount", Descriptors.RingCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("FormalCharge", Chem.GetFormalCharge),
)


@dataclass(frozen=True)
class StandardizedMolecule:
    original_smiles: str
    input_canonical_smiles: str
    input_inchikey: str
    canonical_smiles: str
    inchikey: str
    parent_key: str
    label: int


def fingerprint_matrix(
    molecules: Sequence[Mol],
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> np.ndarray:
    """Create a uint8 ECFP matrix using RDKit's Morgan generator API."""
    if radius < 0 or n_bits <= 0:
        raise ValueError("Fingerprint radius must be non-negative and n_bits must be positive")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=use_chirality,
    )
    matrix = np.zeros((len(molecules), n_bits), dtype=np.uint8)
    for row_index, molecule in enumerate(molecules):
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), matrix[row_index])
    return matrix


def descriptor_frame(molecules: Sequence[Mol], mol_ids: Sequence[str]) -> pd.DataFrame:
    """Calculate the configured RDKit descriptors without fitting transforms."""
    if len(molecules) != len(mol_ids):
        raise ValueError("Molecule and molecule-ID counts differ")
    values = [
        [function(molecule) for _, function in DESCRIPTOR_FUNCTIONS]
        for molecule in molecules
    ]
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size and not np.isfinite(matrix).all():
        bad_rows = np.flatnonzero(~np.isfinite(matrix).all(axis=1)).tolist()
        bad_ids = [mol_ids[index] for index in bad_rows]
        raise ValueError(f"Non-finite RDKit descriptor values for molecule IDs: {bad_ids}")
    return pd.DataFrame(
        matrix,
        columns=[name for name, _ in DESCRIPTOR_FUNCTIONS],
    ).assign(mol_id=list(mol_ids)).loc[:, ["mol_id", *(name for name, _ in DESCRIPTOR_FUNCTIONS)]]


def _read_config(root: Path, config_path: Path | None) -> dict[str, Any]:
    path = config_path or root / "configs" / "features.yaml"
    if not path.is_absolute():
        path = root / path
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: configuration must be a YAML mapping")
    if config.get("dataset_key") != DATASET_KEY:
        raise ValueError(f"{path}: this pipeline only supports {DATASET_KEY}")
    fingerprint = config.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError(f"{path}: fingerprint configuration must be a mapping")
    radius = fingerprint.get("radius")
    n_bits = fingerprint.get("n_bits")
    use_chirality = fingerprint.get("use_chirality")
    if type(radius) is not int or radius < 0:
        raise ValueError(f"{path}: fingerprint.radius must be a non-negative integer")
    if type(n_bits) is not int or n_bits <= 0:
        raise ValueError(f"{path}: fingerprint.n_bits must be a positive integer")
    if type(use_chirality) is not bool:
        raise ValueError(f"{path}: fingerprint.use_chirality must be a boolean")
    for key in ("source_file", "processed_file", "curation_log", "split_file", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{path}: {key} must be a non-empty path")
    config["_config_path"] = path
    return config


def _required_file(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required feature-generation input is missing: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_dataset(path: Path, required_columns: set[str], dtype: dict[str, type]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=dtype)
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path}: missing required columns {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError(f"{path}: dataset is empty")
    return frame


def _prepare_raw_records(
    raw: pd.DataFrame,
) -> tuple[
    dict[str, StandardizedMolecule],
    dict[str, str],
    list[int],
    dict[str, dict[str, list[str]]],
]:
    if raw["Drug_ID"].isna().any() or raw["Drug"].isna().any() or raw["Y"].isna().any():
        raise ValueError("Raw hERG_Karim Drug_ID, Drug, and Y columns must not contain missing values")
    if raw["Drug_ID"].duplicated().any():
        raise ValueError("Raw hERG_Karim contains duplicate Drug_ID values")

    labels = pd.to_numeric(raw["Y"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Raw hERG_Karim labels must be binary 0/1 values")

    records: dict[str, StandardizedMolecule] = {}
    invalid: dict[str, str] = {}
    canonical_groups: dict[str, list[str]] = {}
    input_canonical_groups: dict[str, list[str]] = {}
    parent_groups: dict[str, list[str]] = {}
    for row, label in zip(raw.itertuples(index=False), labels):
        drug_id = str(row.Drug_ID)
        original_smiles = str(row.Drug)
        input_molecule = Chem.MolFromSmiles(original_smiles)
        if input_molecule is None:
            invalid[drug_id] = "parse_fail"
            continue
        input_canonical_smiles = Chem.MolToSmiles(
            input_molecule,
            canonical=True,
            isomericSmiles=True,
        )
        input_inchikey = inchi.MolToInchiKey(input_molecule)
        canonical_smiles, inchikey, _ = standardise(original_smiles)
        if canonical_smiles is None:
            invalid[drug_id] = "standardise_fail"
            continue
        if not isinstance(inchikey, str) or not inchikey:
            invalid[drug_id] = "inchikey_unavailable"
            continue
        record = StandardizedMolecule(
            original_smiles=original_smiles,
            input_canonical_smiles=input_canonical_smiles,
            input_inchikey=input_inchikey,
            canonical_smiles=canonical_smiles,
            inchikey=inchikey,
            parent_key=inchikey.split("-")[0],
            label=int(label),
        )
        records[drug_id] = record
        canonical_groups.setdefault(canonical_smiles, []).append(drug_id)
        input_canonical_groups.setdefault(input_canonical_smiles, []).append(drug_id)
        parent_groups.setdefault(record.parent_key, []).append(drug_id)
    return records, invalid, list(labels.astype(int)), {
        "canonical": canonical_groups,
        "input_canonical": input_canonical_groups,
        "parent": parent_groups,
    }


def _validate_curation(
    raw: pd.DataFrame,
    processed: pd.DataFrame,
    log: pd.DataFrame,
    raw_records: dict[str, StandardizedMolecule],
    invalid: dict[str, str],
    labels: list[int],
    groups: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    if processed["orig_id"].isna().any() or processed["smiles"].isna().any():
        raise ValueError("Curated hERG_Karim orig_id and smiles columns must not contain missing values")
    processed_ids = processed["orig_id"].astype(str).tolist()
    if len(processed_ids) != len(set(processed_ids)):
        raise ValueError("Curated hERG_Karim contains duplicate orig_id values")
    if processed["label"].isna().any():
        raise ValueError("Curated hERG_Karim labels must not contain missing values")
    curated_labels = pd.to_numeric(processed["label"], errors="coerce")
    if curated_labels.isna().any() or not set(curated_labels.unique()).issubset({0, 1}):
        raise ValueError("Curated hERG_Karim labels must be binary 0/1 values")
    if processed["parent_key"].duplicated().any():
        raise ValueError("Curated hERG_Karim has duplicate parent_key values")

    raw_by_id: dict[str, tuple[StandardizedMolecule | None, int]] = {
        str(drug_id): (raw_records.get(str(drug_id)), label)
        for drug_id, label in zip(raw["Drug_ID"], labels)
    }
    if any(drug_id not in raw_by_id for drug_id in processed_ids):
        raise ValueError("Curated orig_id values do not map to raw Drug_ID values")
    expected_representatives = {
        ids[0]
        for ids in groups["parent"].values()
        if len({raw_by_id[drug_id][1] for drug_id in ids}) == 1
    }
    if set(processed_ids) != expected_representatives:
        raise ValueError("Curated rows do not match the existing standardization and duplicate-resolution rules")

    identity_discrepancies: list[dict[str, Any]] = []
    curated_structures: dict[str, tuple[str, str, Mol]] = {}
    for row, drug_id, curated_label in zip(processed.itertuples(index=False), processed_ids, curated_labels):
        record = raw_records.get(drug_id)
        if record is None:
            raise ValueError(f"Curated Drug_ID {drug_id} is not a valid raw molecule")
        raw_record, raw_label = raw_by_id[drug_id]
        if raw_record is None or raw_record is not record:
            raise ValueError(f"Could not map curated Drug_ID {drug_id} back to its raw record")
        if int(curated_label) != raw_label or int(curated_label) != record.label:
            raise ValueError(f"Label mismatch between raw and curated data for Drug_ID {drug_id}")
        processed_canonical, processed_inchikey, _ = standardise(str(row.smiles))
        if processed_canonical is None or not isinstance(processed_inchikey, str):
            raise ValueError(f"Curated structure is invalid for Drug_ID {drug_id}")
        processed_parent_key = processed_inchikey.split("-")[0]
        if processed_parent_key != record.parent_key:
            raise ValueError(f"Curated and raw structures have different parent identities for Drug_ID {drug_id}")
        feature_molecule = Chem.MolFromSmiles(processed_canonical)
        if feature_molecule is None:
            raise ValueError(f"Could not parse curated canonical SMILES for Drug_ID {drug_id}")
        curated_structures[drug_id] = (processed_canonical, processed_inchikey, feature_molecule)
        discrepancies: list[str] = []
        if record.canonical_smiles != processed_canonical:
            discrepancies.append("raw_vs_curated_canonical_smiles")
        if str(row.inchikey) != processed_inchikey:
            discrepancies.append("stored_vs_recomputed_inchikey")
        if str(row.parent_key) != processed_parent_key:
            discrepancies.append("stored_vs_recomputed_parent_key")
        if discrepancies:
            identity_discrepancies.append({"Drug_ID": drug_id, "differences": discrepancies})

    parent_groups = groups["parent"]
    duplicate_groups = [
        ids for ids in parent_groups.values()
        if len(ids) > 1 and len({raw_by_id[drug_id][1] for drug_id in ids}) == 1
    ]
    conflict_groups = [
        ids for ids in parent_groups.values()
        if len({raw_by_id[drug_id][1] for drug_id in ids}) > 1
    ]
    merge_log = log[log["reason"] == "duplicate_parent"]
    conflict_log = log[log["reason"] == "label_conflict"]
    invalid_log = log[log["reason"].isin(["parse_fail", "standardise_fail"])]
    if not set(log["reason"]).issubset(
        {"duplicate_parent", "label_conflict", "parse_fail", "standardise_fail"}
    ):
        raise ValueError("Curation log contains unrecognized reasons")
    if len(merge_log) != len(duplicate_groups):
        raise ValueError("Curation log duplicate-parent count does not match the raw data")
    if len(conflict_log) != len(conflict_groups):
        raise ValueError("Curation log label-conflict count does not match the raw data")
    if len(invalid_log) != sum(reason in {"parse_fail", "standardise_fail"} for reason in invalid.values()):
        raise ValueError("Curation log invalid-structure count does not match the raw data")
    if not (merge_log["action"] == "merge").all() or not (conflict_log["action"] == "drop").all():
        raise ValueError("Curation log actions do not match duplicate and conflict outcomes")
    if not (invalid_log["action"] == "drop").all():
        raise ValueError("Curation log must mark invalid structures as dropped")
    logged_merge_sizes = sorted(pd.to_numeric(merge_log["n"], errors="raise").astype(int).tolist())
    observed_merge_sizes = sorted(len(ids) for ids in duplicate_groups)
    logged_conflict_sizes = sorted(pd.to_numeric(conflict_log["n"], errors="raise").astype(int).tolist())
    observed_conflict_sizes = sorted(len(ids) for ids in conflict_groups)
    if logged_merge_sizes != observed_merge_sizes or logged_conflict_sizes != observed_conflict_sizes:
        raise ValueError("Curation log group sizes do not match the raw molecular groups")

    merged_rows = sum(len(ids) - 1 for ids in duplicate_groups)
    conflict_rows = sum(len(ids) for ids in conflict_groups)
    invalid_rows = len(invalid)
    if len(raw) - len(processed) != merged_rows + conflict_rows + invalid_rows:
        raise ValueError("Raw-to-curated row change is not accounted for by merges, conflicts, and invalid structures")
    return {
        "duplicate_parent_groups": duplicate_groups,
        "conflicting_parent_groups": conflict_groups,
        "merged_rows": merged_rows,
        "conflict_rows": conflict_rows,
        "invalid_rows": invalid_rows,
        "curated_structures": curated_structures,
        "identity_discrepancies": identity_discrepancies,
    }


def _make_metadata(
    processed: pd.DataFrame,
    raw_records: dict[str, StandardizedMolecule],
    curated_structures: dict[str, tuple[str, str, Mol]],
) -> tuple[pd.DataFrame, list[Mol]]:
    metadata_rows: list[dict[str, Any]] = []
    molecules: list[Mol] = []
    for row_index, row in enumerate(processed.itertuples(index=False)):
        drug_id = str(row.orig_id)
        record = raw_records.get(drug_id)
        if record is None:
            raise ValueError(f"Curated molecule {drug_id} has no valid raw structure")
        canonical_smiles, inchikey, molecule = curated_structures[drug_id]
        mol_id = f"{DATASET_KEY}_{row_index:06d}"
        metadata_rows.append(
            {
                "mol_id": mol_id,
                "Drug_ID": drug_id,
                "original_smiles": record.original_smiles,
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "input_canonical_smiles": record.input_canonical_smiles,
                "input_inchikey": record.input_inchikey,
                "standardized_input_smiles": record.canonical_smiles,
                "standardized_input_inchikey": record.inchikey,
                "source_inchikey": str(row.inchikey),
                "source_parent_key": str(row.parent_key),
                "label": record.label,
            }
        )
        molecules.append(molecule)
    return pd.DataFrame(metadata_rows), molecules


def _validate_splits(
    path: Path,
    processed: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, int]]]:
    splits = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"{path}: expected exactly {list(SPLIT_NAMES)}")
    n_molecules = len(processed)
    expected_indices = set(range(n_molecules))
    mol_ids = metadata["mol_id"].tolist()
    canonical_by_id = dict(zip(mol_ids, metadata["canonical_smiles"]))
    assignments: dict[str, dict[str, str]] = {mol_id: {} for mol_id in mol_ids}
    counts: dict[str, dict[str, int]] = {}

    for split_name in SPLIT_NAMES:
        split = splits[split_name]
        if not isinstance(split, dict) or not {"train", "val", "test", "test_novelty"}.issubset(split):
            raise ValueError(f"{split_name}: missing train/val/test or test_novelty")
        index_sets: dict[str, set[int]] = {}
        for partition in ("train", "val", "test"):
            indices = split[partition]
            if not isinstance(indices, list) or any(type(index) is not int for index in indices):
                raise ValueError(f"{split_name}: {partition} indices must be integers")
            if len(indices) != len(set(indices)):
                raise ValueError(f"{split_name}: duplicate indices in {partition}")
            if any(index < 0 or index >= n_molecules for index in indices):
                raise ValueError(f"{split_name}: out-of-range index in {partition}")
            index_sets[partition] = set(indices)
        partitions = list(index_sets.values())
        if any(partitions[left] & partitions[right] for left in range(3) for right in range(left + 1, 3)):
            raise ValueError(f"{split_name}: train, val, and test overlap")
        if set.union(*partitions) != expected_indices:
            raise ValueError(f"{split_name}: partitions do not cover the current curated dataset")
        novelty = split["test_novelty"]
        if not isinstance(novelty, list) or len(novelty) != len(split["test"]):
            raise ValueError(f"{split_name}: test_novelty is not aligned with test indices")
        novelty_values = np.asarray(novelty, dtype=float)
        if not np.isfinite(novelty_values).all() or ((novelty_values < 0) | (novelty_values > 1)).any():
            raise ValueError(f"{split_name}: test_novelty contains invalid values")

        structures = {
            partition: {canonical_by_id[mol_ids[index]] for index in indices}
            for partition, indices in split.items()
            if partition in {"train", "val", "test"}
        }
        partition_names = ("train", "val", "test")
        if any(
            structures[partition_names[left]] & structures[partition_names[right]]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError(f"{split_name}: identical canonical structures cross partitions")

        for partition in partition_names:
            for index in split[partition]:
                assignments[mol_ids[index]][split_name] = partition
        counts[split_name] = {partition: len(split[partition]) for partition in partition_names}

    serialized = {
        mol_id: json.dumps(assignment, sort_keys=True, separators=(",", ":"))
        for mol_id, assignment in assignments.items()
    }
    return splits, serialized, counts


def _audit_text(
    raw_path: Path,
    raw: pd.DataFrame,
    processed: pd.DataFrame,
    raw_records: dict[str, StandardizedMolecule],
    invalid: dict[str, str],
    labels: list[int],
    groups: dict[str, dict[str, list[str]]],
    curation: dict[str, Any],
    split_counts: dict[str, dict[str, int]],
    input_checksums: dict[str, str],
) -> str:
    canonical_duplicates = {
        smiles: ids for smiles, ids in groups["canonical"].items() if len(ids) > 1
    }
    input_canonical_duplicates = {
        smiles: ids for smiles, ids in groups["input_canonical"].items() if len(ids) > 1
    }
    input_canonical_duplicate_rows = sum(len(ids) for ids in input_canonical_duplicates.values())
    input_canonical_excess_rows = sum(len(ids) - 1 for ids in input_canonical_duplicates.values())
    canonical_duplicate_rows = sum(len(ids) for ids in canonical_duplicates.values())
    canonical_excess_rows = sum(len(ids) - 1 for ids in canonical_duplicates.values())
    lines = [
        "hERG_Karim molecular feature-generation audit",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Raw source: {raw_path.as_posix()}",
        f"Raw rows: {len(raw)}",
        f"Curated rows used for features: {len(processed)}",
        f"Raw labels: negatives={labels.count(0)}, positives={labels.count(1)}",
        f"Raw missing values: {int(raw.isna().sum().sum())}",
        f"Raw duplicate Drug_ID values: {int(raw['Drug_ID'].duplicated().sum())}",
        f"Raw duplicate original SMILES values: {int(raw['Drug'].duplicated().sum())}",
        f"Valid standardized raw structures: {len(raw_records)}",
        f"Invalid or unstandardizable raw structures: {len(invalid)}",
        f"Raw structures without an InChIKey: {sum(not record.input_inchikey for record in raw_records.values())}",
        f"Duplicate standardized canonical isomeric SMILES groups: {len(canonical_duplicates)}",
        f"Rows in duplicate canonical groups: {canonical_duplicate_rows}",
        f"Excess rows in duplicate canonical groups: {canonical_excess_rows}",
        f"Duplicate original-input canonical isomeric SMILES groups: {len(input_canonical_duplicates)}",
        f"Rows in duplicate original-input canonical groups: {input_canonical_duplicate_rows}",
        f"Excess rows in duplicate original-input canonical groups: {input_canonical_excess_rows}",
        f"Same-label parent groups merged by existing curation: {len(curation['duplicate_parent_groups'])}",
        f"Raw rows merged into those parent representatives: {curation['merged_rows']}",
        f"Conflicting-label parent groups excluded by existing curation: {len(curation['conflicting_parent_groups'])}",
        f"Raw rows excluded for label conflicts: {curation['conflict_rows']}",
        f"Raw rows excluded for invalid structures: {curation['invalid_rows']}",
        f"Raw-to-curated canonical SMILES differences: {sum('raw_vs_curated_canonical_smiles' in item['differences'] for item in curation['identity_discrepancies'])}",
        f"Rows with stored-versus-recomputed InChIKey/parent-key discrepancies: {sum(any(value.startswith('stored_vs_recomputed_') for value in item['differences']) for item in curation['identity_discrepancies'])}",
        "No raw labels were altered. Fingerprints/descriptors use frozen curated SMILES; original and original-standardized SMILES/InChIKeys are preserved separately.",
        "No scaling, imputation, PCA, feature selection, or model fitting was performed.",
        "",
        "Frozen scaffold split counts (each curated molecule has one assignment per repeat):",
    ]
    for split_name, values in split_counts.items():
        lines.append(
            f"  {split_name}: train={values['train']}, val={values['val']}, test={values['test']}"
        )
    lines.extend(["", "Input SHA-256 checksums:"])
    lines.extend(f"  {name}: {digest}" for name, digest in input_checksums.items())
    lines.extend(["", "Invalid structures:"])
    if invalid:
        lines.extend(f"  Drug_ID={drug_id}: {reason}" for drug_id, reason in sorted(invalid.items()))
    else:
        lines.append("  None.")
    lines.extend(["", "Standardized canonical-structure duplicate groups:"])
    if canonical_duplicates:
        for smiles, ids in sorted(canonical_duplicates.items()):
            lines.append(f"  Drug_IDs={','.join(ids)} canonical_smiles={smiles}")
    else:
        lines.append("  None.")
    lines.extend(["", "Original-input canonical-structure duplicate groups:"])
    if input_canonical_duplicates:
        for smiles, ids in sorted(input_canonical_duplicates.items()):
            lines.append(f"  Drug_IDs={','.join(ids)} canonical_smiles={smiles}")
    else:
        lines.append("  None.")
    lines.extend(["", "Same-label parent merge groups (first Drug_ID is retained):"])
    if curation["duplicate_parent_groups"]:
        lines.extend(
            f"  retained={ids[0]} merged={','.join(ids[1:])}"
            for ids in curation["duplicate_parent_groups"]
        )
    else:
        lines.append("  None.")
    lines.extend(["", "Conflicting-label parent groups (all listed source rows were excluded):"])
    if curation["conflicting_parent_groups"]:
        for ids in curation["conflicting_parent_groups"]:
            group_labels = sorted({raw_records[drug_id].label for drug_id in ids if drug_id in raw_records})
            lines.append(f"  Drug_IDs={','.join(ids)} labels={group_labels}")
    else:
        lines.append("  None.")
    lines.extend(["", "Raw-to-curated standardized identity warnings:"])
    if curation["identity_discrepancies"]:
        lines.extend(
            f"  Drug_ID={item['Drug_ID']}: {', '.join(item['differences'])}"
            for item in curation["identity_discrepancies"]
        )
    else:
        lines.append("  None.")
    return "\n".join(lines) + "\n"


def generate_features(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Generate feature files from existing hERG_Karim data and frozen splits."""
    root = root.resolve()
    config = _read_config(root, config_path)
    raw_path = _required_file(root, config["source_file"])
    processed_path = _required_file(root, config["processed_file"])
    log_path = _required_file(root, config["curation_log"])
    split_path = _required_file(root, config["split_file"])
    config_file = Path(config["_config_path"])
    inputs = {
        "raw_dataset": raw_path,
        "curated_dataset": processed_path,
        "curation_log": log_path,
        "frozen_splits": split_path,
        "feature_config": config_file,
    }
    input_checksums = {name: _sha256(path) for name, path in inputs.items()}

    raw = _read_dataset(raw_path, RAW_COLUMNS, {"Drug_ID": str, "Drug": str})
    processed = _read_dataset(processed_path, PROCESSED_COLUMNS, {"orig_id": str})
    log = pd.read_csv(log_path)
    if not {"action", "reason", "n"}.issubset(log.columns):
        raise ValueError(f"{log_path}: curation log must contain action, reason, and n")

    raw_records, invalid, labels, groups = _prepare_raw_records(raw)
    curation = _validate_curation(raw, processed, log, raw_records, invalid, labels, groups)
    metadata, molecules = _make_metadata(
        processed,
        raw_records,
        curation["curated_structures"],
    )
    _, assignments, split_counts = _validate_splits(split_path, processed, metadata)
    metadata["split_assignment"] = metadata["mol_id"].map(assignments)

    fingerprint_config = config["fingerprint"]
    fingerprints = fingerprint_matrix(
        molecules,
        radius=fingerprint_config["radius"],
        n_bits=fingerprint_config["n_bits"],
        use_chirality=fingerprint_config["use_chirality"],
    )
    if fingerprints.shape != (len(metadata), fingerprint_config["n_bits"]):
        raise ValueError(f"Unexpected fingerprint matrix shape: {fingerprints.shape}")
    descriptors = descriptor_frame(molecules, metadata["mol_id"].tolist())
    if descriptors["mol_id"].tolist() != metadata["mol_id"].tolist():
        raise ValueError("Descriptor molecule order does not match metadata")

    chemprop = metadata[
        ["mol_id", "original_smiles", "canonical_smiles", "label", "split_assignment"]
    ].copy()
    molformer = metadata[
        ["mol_id", "canonical_smiles", "label", "split_assignment"]
    ].copy()
    if not (
        metadata["label"].equals(chemprop["label"])
        and metadata["label"].equals(molformer["label"])
        and np.isfinite(descriptors.drop(columns="mol_id").to_numpy(dtype=float)).all()
    ):
        raise ValueError("Feature representations failed label or finite-value consistency checks")

    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "molecule_metadata.csv", index=False)
    np.savez_compressed(
        output_dir / "ecfp4_2048.npz",
        fingerprints=fingerprints,
        mol_ids=metadata["mol_id"].to_numpy(dtype=str),
    )
    descriptors.to_csv(output_dir / "rdkit_descriptors.csv", index=False)
    chemprop.to_csv(output_dir / "chemprop_input.csv", index=False)
    molformer.to_csv(output_dir / "molformer_input.csv", index=False)

    manifest = {
        "dataset": "hERG_Karim",
        "dataset_key": DATASET_KEY,
        "source_file": config["source_file"],
        "source_dataset_row_count": len(raw),
        "curated_file": config["processed_file"],
        "curated_dataset_row_count": len(processed),
        "feature_molecule_count": len(metadata),
        "feature_structure_source": "frozen curated processed SMILES; raw originals and original-standardized identities are also preserved",
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "fingerprint": {
            "type": "Morgan/ECFP4",
            "radius": fingerprint_config["radius"],
            "n_bits": fingerprint_config["n_bits"],
            "use_chirality": fingerprint_config["use_chirality"],
            "dtype": str(fingerprints.dtype),
            "npz_keys": ["fingerprints", "mol_ids"],
        },
        "descriptor_names": [name for name, _ in DESCRIPTOR_FUNCTIONS],
        "valid_raw_structures": len(raw_records),
        "invalid_raw_structures": len(invalid),
        "raw_structures_without_inchikey": sum(
            not record.input_inchikey for record in raw_records.values()
        ),
        "canonical_duplicate_groups": sum(len(ids) > 1 for ids in groups["canonical"].values()),
        "canonical_duplicate_rows": sum(
            len(ids) for ids in groups["canonical"].values() if len(ids) > 1
        ),
        "canonical_duplicate_excess_rows": sum(
            len(ids) - 1 for ids in groups["canonical"].values() if len(ids) > 1
        ),
        "original_canonical_duplicate_groups": sum(
            len(ids) > 1 for ids in groups["input_canonical"].values()
        ),
        "original_canonical_duplicate_rows": sum(
            len(ids) for ids in groups["input_canonical"].values() if len(ids) > 1
        ),
        "original_canonical_duplicate_excess_rows": sum(
            len(ids) - 1 for ids in groups["input_canonical"].values() if len(ids) > 1
        ),
        "curation": {
            "same_label_parent_groups_merged": len(curation["duplicate_parent_groups"]),
            "raw_rows_merged": curation["merged_rows"],
            "conflicting_parent_groups_excluded": len(curation["conflicting_parent_groups"]),
            "raw_rows_excluded_for_label_conflicts": curation["conflict_rows"],
            "raw_rows_excluded_for_invalid_structures": curation["invalid_rows"],
        },
        "identity_discrepancies": curation["identity_discrepancies"],
        "split_counts": split_counts,
        "input_sha256": input_checksums,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = _audit_text(
        raw_path,
        raw,
        processed,
        raw_records,
        invalid,
        labels,
        groups,
        curation,
        split_counts,
        input_checksums,
    )
    (output_dir / "feature_generation_audit.txt").write_text(audit, encoding="utf-8")

    LOGGER.info(
        "Generated %d hERG_Karim molecules: fingerprints=%s, descriptors=%s",
        len(metadata),
        fingerprints.shape,
        (len(descriptors), len(DESCRIPTOR_FUNCTIONS)),
    )
    if curation["identity_discrepancies"]:
        LOGGER.warning(
            "%d raw-to-curated standardized identity discrepancies are documented in the audit",
            len(curation["identity_discrepancies"]),
        )
    return {
        "output_dir": output_dir,
        "source_rows": len(raw),
        "feature_rows": len(metadata),
        "fingerprint_shape": fingerprints.shape,
        "descriptor_shape": (len(descriptors), len(DESCRIPTOR_FUNCTIONS)),
        "invalid_raw_structures": len(invalid),
        "canonical_duplicate_groups": manifest["canonical_duplicate_groups"],
        "canonical_duplicate_excess_rows": manifest["canonical_duplicate_excess_rows"],
        "split_counts": split_counts,
    }
