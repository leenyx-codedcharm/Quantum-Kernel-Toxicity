"""Read-only integrity checks for source data and frozen splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .curate import standardise

DATASETS = ("herg", "herg_karim", "dili")
NOVELTY_BIN_NAMES = ("familiar", "intermediate", "novel")
RAW_COLUMNS = {"Drug", "Y", "Drug_ID"}
PROCESSED_COLUMNS = {
    "smiles",
    "inchikey",
    "parent_key",
    "scaffold",
    "label",
    "acyclic",
}


def novelty_bin_indices(values: list[float] | np.ndarray, edges: list[float]) -> np.ndarray:
    """Return bin numbers for lower-inclusive first bin and right-closed intervals."""
    return np.digitize(np.asarray(values, dtype=float), edges[1:-1], right=True)


def _check(condition: bool, message: str, issues: list[str]) -> None:
    if not condition:
        issues.append(message)


def _read_yaml(path: Path, issues: list[str]) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict):
            raise ValueError("configuration root must be a mapping")
        return config
    except (OSError, yaml.YAMLError, ValueError) as exc:
        issues.append(f"{path}: cannot read configuration ({exc})")
        return None


def _check_manifest(root: Path, manifest: dict[str, Any], dataset: str, issues: list[str]) -> None:
    raw_path = root / "data" / "raw" / f"{dataset}.csv"
    record = manifest.get("files", {}).get(dataset)
    _check(isinstance(record, dict), f"manifest has no record for {dataset}", issues)
    if not isinstance(record, dict) or not raw_path.is_file():
        return

    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    _check(record.get("sha256") == digest, f"{dataset}: raw checksum differs from manifest", issues)
    try:
        raw = pd.read_csv(raw_path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        issues.append(f"{raw_path}: cannot read raw dataset ({exc})")
        return

    _check(record.get("rows") == len(raw), f"{dataset}: raw row count differs from manifest", issues)
    counts = {str(key): int(value) for key, value in raw["Y"].value_counts().items()} if "Y" in raw else {}
    _check(record.get("label_counts") == counts, f"{dataset}: raw label counts differ from manifest", issues)


def _check_curation(
    root: Path,
    dataset: str,
    raw: pd.DataFrame,
    processed: pd.DataFrame,
    issues: list[str],
) -> None:
    log_path = root / "data" / "processed" / f"{dataset}_curation_log.csv"
    if not log_path.is_file():
        issues.append(f"{log_path}: curation log is missing")
        return
    try:
        log = pd.read_csv(log_path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        issues.append(f"{log_path}: cannot read curation log ({exc})")
        return

    required = {"action", "reason", "n"}
    _check(required.issubset(log.columns), f"{dataset}: curation log missing {sorted(required - set(log.columns))}", issues)
    if not required.issubset(log.columns):
        return

    merges = log[log["reason"] == "duplicate_parent"]
    conflicts = log[log["reason"] == "label_conflict"]
    failures = log[log["reason"].isin(["parse_fail", "standardise_fail"])]
    _check((merges["action"] == "merge").all(), f"{dataset}: duplicate-parent entries are not merges", issues)
    _check((pd.to_numeric(merges["n"], errors="coerce") >= 2).all(), f"{dataset}: invalid duplicate-parent group size", issues)
    _check((conflicts["action"] == "drop").all(), f"{dataset}: label-conflict entries are not drops", issues)
    _check((pd.to_numeric(conflicts["n"], errors="coerce") >= 2).all(), f"{dataset}: invalid conflict group size", issues)
    _check((failures["action"] == "drop").all(), f"{dataset}: parse/standardization failures are not drops", issues)

    removed_by_logs = (
        pd.to_numeric(merges["n"], errors="coerce").sub(1).sum()
        + pd.to_numeric(conflicts["n"], errors="coerce").sum()
        + len(failures)
    )
    _check(
        int(removed_by_logs) == len(raw) - len(processed),
        f"{dataset}: curation log does not account for raw-to-processed row change",
        issues,
    )
    _check(
        not processed["parent_key"].duplicated().any(),
        f"{dataset}: processed rows contain duplicate parent_key values",
        issues,
    )


def _check_molecule_consistency(dataset: str, processed: pd.DataFrame, issues: list[str]) -> None:
    labels = set(pd.to_numeric(processed["label"], errors="coerce").dropna().unique())
    _check(labels.issubset({0, 1}) and processed["label"].notna().all(), f"{dataset}: processed labels are not binary 0/1", issues)

    mismatches: list[str] = []
    for row_index, row in processed.iterrows():
        if not isinstance(row["smiles"], str) or not row["smiles"]:
            mismatches.append(f"row {row_index}: missing or invalid SMILES value")
            continue
        canonical, inchikey, scaffold = standardise(row["smiles"])
        if canonical is None:
            mismatches.append(f"row {row_index}: invalid SMILES")
            continue
        expected = {
            "smiles": canonical,
            "inchikey": inchikey,
            "parent_key": inchikey.split("-")[0],
            "scaffold": scaffold,
            "acyclic": scaffold == "",
        }
        for column, value in expected.items():
            observed = row[column]
            if column == "acyclic":
                matches = bool(observed) == value
            elif column == "scaffold" and value == "":
                matches = pd.isna(observed) or observed == ""
            else:
                matches = str(observed) == str(value)
            if not matches:
                mismatches.append(f"row {row_index}: {column} differs from standardized molecule")
                break
    if mismatches:
        issues.append(f"{dataset}: molecule/scaffold consistency failed ({mismatches[0]}; {len(mismatches)} row(s))")


def _check_split(
    dataset: str,
    split_name: str,
    split: Any,
    processed: pd.DataFrame,
    edges: list[float],
    issues: list[str],
) -> None:
    prefix = f"{dataset}/{split_name}"
    if not isinstance(split, dict):
        issues.append(f"{prefix}: split entry is not an object")
        return
    required = {"train", "val", "test", "test_novelty"}
    _check(required.issubset(split), f"{prefix}: missing split keys {sorted(required - set(split))}", issues)
    if not required.issubset(split):
        return

    indices: dict[str, list[int]] = {}
    valid = True
    for part in ("train", "val", "test"):
        values = split[part]
        if not isinstance(values, list) or any(type(index) is not int for index in values):
            issues.append(f"{prefix}: {part} indices must be a list of integers")
            valid = False
            continue
        indices[part] = values
        _check(len(values) == len(set(values)), f"{prefix}: duplicate indices in {part}", issues)
        _check(all(0 <= index < len(processed) for index in values), f"{prefix}: out-of-range {part} index", issues)

    if not valid or len(indices) != 3:
        return
    parts = [set(indices[name]) for name in ("train", "val", "test")]
    _check(
        not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]),
        f"{prefix}: split partitions overlap",
        issues,
    )
    _check(
        set.union(*parts) == set(range(len(processed))),
        f"{prefix}: split partitions do not cover every processed row exactly once",
        issues,
    )
    if any(index >= len(processed) or index < 0 for group in indices.values() for index in group):
        return

    role_rows = [processed.iloc[indices[name]] for name in ("train", "val", "test")]
    parent_sets = [set(rows["parent_key"]) for rows in role_rows]
    _check(
        not (parent_sets[0] & parent_sets[1] or parent_sets[0] & parent_sets[2] or parent_sets[1] & parent_sets[2]),
        f"{prefix}: duplicate parent appears in multiple partitions",
        issues,
    )

    scaffold_sets = []
    for rows, row_indices in zip(role_rows, indices.values()):
        keys = [
            f"ACYCLIC_{index}" if bool(row["acyclic"]) else str(row["scaffold"])
            for index, (_, row) in zip(row_indices, rows.iterrows())
        ]
        scaffold_sets.append(set(keys))
    _check(
        not (scaffold_sets[0] & scaffold_sets[1] or scaffold_sets[0] & scaffold_sets[2] or scaffold_sets[1] & scaffold_sets[2]),
        f"{prefix}: scaffold appears in multiple partitions",
        issues,
    )

    novelty = split["test_novelty"]
    if not isinstance(novelty, list) or len(novelty) != len(indices["test"]):
        issues.append(f"{prefix}: test_novelty is not aligned with test indices")
        return
    try:
        values = np.asarray(novelty, dtype=float)
    except (TypeError, ValueError):
        issues.append(f"{prefix}: test_novelty must contain numeric values")
        return
    _check(np.isfinite(values).all(), f"{prefix}: test_novelty contains non-finite values", issues)
    _check(((values >= edges[0]) & (values <= edges[-1])).all(), f"{prefix}: test_novelty outside configured range", issues)
    counts = np.bincount(novelty_bin_indices(values, edges), minlength=len(NOVELTY_BIN_NAMES))
    print(f"  {split_name}: train={len(indices['train'])} val={len(indices['val'])} test={len(indices['test'])}; novelty bins={counts.tolist()}")


def run_audit(root: Path) -> list[str]:
    """Run integrity checks and return any errors; never writes to the repository."""
    root = root.resolve()
    issues: list[str] = []
    config = _read_yaml(root / "configs" / "data.yaml", issues)
    if config is None:
        return issues

    expected_edges = [0.0, 0.4, 0.6, 1.0]
    edges = config.get("novelty_bins")
    _check(edges == expected_edges, f"novelty_bins must be {expected_edges}", issues)
    if edges != expected_edges:
        edges = expected_edges
    _check(
        tuple(NOVELTY_BIN_NAMES) == ("familiar", "intermediate", "novel"),
        "internal novelty-bin labels are inconsistent",
        issues,
    )

    manifest_path = root / "data" / "raw" / "MANIFEST.json"
    manifest: dict[str, Any] = {}
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        _check(isinstance(manifest_data, dict), "raw manifest root must be an object", issues)
        if isinstance(manifest_data, dict):
            manifest = manifest_data
            _check(isinstance(manifest.get("files"), dict), "raw manifest has no files mapping", issues)
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"{manifest_path}: cannot read raw manifest ({exc})")

    for dataset in DATASETS:
        print(f"\n===== {dataset} =====")
        raw_path = root / "data" / "raw" / f"{dataset}.csv"
        processed_path = root / "data" / "processed" / f"{dataset}.csv"
        split_path = root / "data" / "splits" / f"{dataset}_scaffold.json"
        cluster_path = root / "data" / "splits" / f"{dataset}_clusters.csv"
        log_path = root / "data" / "processed" / f"{dataset}_curation_log.csv"

        for path in (raw_path, processed_path, split_path, cluster_path, log_path):
            _check(path.is_file(), f"{path.relative_to(root)} is missing", issues)
        if not all(path.is_file() for path in (raw_path, processed_path)):
            continue

        try:
            raw = pd.read_csv(raw_path)
            processed = pd.read_csv(processed_path)
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            issues.append(f"{dataset}: cannot read dataset ({exc})")
            continue

        _check(RAW_COLUMNS.issubset(raw.columns), f"{dataset}: raw columns missing {sorted(RAW_COLUMNS - set(raw.columns))}", issues)
        missing_processed = PROCESSED_COLUMNS - set(processed.columns)
        _check(not missing_processed, f"{dataset}: processed columns missing {sorted(missing_processed)}", issues)
        if not RAW_COLUMNS.issubset(raw.columns) or missing_processed:
            continue

        _check(raw["Y"].notna().all() and set(raw["Y"].unique()).issubset({0, 1, 0.0, 1.0}), f"{dataset}: raw labels are not binary 0/1", issues)
        _check(raw["Drug"].notna().all(), f"{dataset}: raw SMILES contain missing values", issues)
        _check(processed["smiles"].notna().all(), f"{dataset}: processed SMILES contain missing values", issues)
        _check(processed["parent_key"].notna().all(), f"{dataset}: processed parent keys contain missing values", issues)
        print(f"  raw rows={len(raw)} processed rows={len(processed)}")

        if manifest:
            _check_manifest(root, manifest, dataset, issues)
        _check_curation(root, dataset, raw, processed, issues)
        _check_molecule_consistency(dataset, processed, issues)

        if split_path.is_file():
            try:
                splits = json.loads(split_path.read_text(encoding="utf-8-sig"))
                if not isinstance(splits, dict):
                    issues.append(f"{split_path}: frozen split root must be an object")
                else:
                    expected_splits = {f"scaffold_r{repeat}" for repeat in range(config.get("splits", {}).get("n_repeats", 5))}
                    _check(set(splits) == expected_splits, f"{dataset}: scaffold split names differ from configured repeats", issues)
                    for split_name, split in splits.items():
                        _check_split(dataset, split_name, split, processed, edges, issues)
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"{split_path}: cannot read frozen split ({exc})")

        if cluster_path.is_file():
            try:
                clusters = pd.read_csv(cluster_path)
                _check("cluster" in clusters.columns, f"{dataset}: cluster file has no cluster column", issues)
                index_columns = [column for column in clusters.columns if column in {"Unnamed: 0", "index", "row_index"}]
                _check(bool(index_columns), f"{dataset}: cluster file has no row-index column", issues)
                if "cluster" in clusters.columns and index_columns:
                    row_indices = pd.to_numeric(clusters[index_columns[0]], errors="coerce")
                    expected_indices = set(range(len(processed)))
                    observed_indices = set(row_indices.dropna().astype(int))
                    _check(
                        len(clusters) == len(processed)
                        and observed_indices == expected_indices
                        and row_indices.notna().all()
                        and (row_indices % 1 == 0).all(),
                        f"{dataset}: cluster assignments do not cover processed rows exactly once",
                        issues,
                    )
                    _check(not clusters["cluster"].isna().any(), f"{dataset}: missing cluster assignment", issues)
                    _check(
                        pd.to_numeric(clusters["cluster"], errors="coerce").notna().all(),
                        f"{dataset}: cluster assignments must be numeric",
                        issues,
                    )
                    print(f"  cluster rows={len(clusters)} unique clusters={clusters['cluster'].nunique()}")
            except (OSError, pd.errors.ParserError, UnicodeError) as exc:
                issues.append(f"{cluster_path}: cannot read cluster assignments ({exc})")

    if issues:
        print(f"\nAUDIT FAILED: {len(issues)} issue(s)")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nAUDIT PASSED: all configured source datasets, frozen splits, and cluster assignments checked.")
    return issues
