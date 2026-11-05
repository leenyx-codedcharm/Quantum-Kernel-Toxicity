"""Reproducible scientific reports and plots from saved hERG_Karim predictions."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import re
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import t as student_t
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from qktox.baselines import ExperimentData, load_experiment_data

LOGGER = logging.getLogger(__name__)
METRICS = (
    "f1_positive",
    "f1_macro",
    "f1_weighted",
    "precision",
    "recall",
    "auroc",
    "auprc",
    "balanced_accuracy",
    "mcc",
    "brier_score",
    "log_loss",
)
MODEL_ORDER = (
    "lightgbm_descriptors",
    "lightgbm_fingerprints",
    "lightgbm_combined",
    "random_forest_descriptors",
    "random_forest_fingerprints",
    "random_forest_combined",
    "tanimoto_svm_fingerprints",
)
CLASS_LABELS = ("Non-blocker (0)", "Blocker (1)")
DEFAULT_SIMILARITY_SEED = 49110
DEFAULT_RANDOM_PAIR_COUNT = 50_000
DEFAULT_NEIGHBOR_SAMPLE_COUNT = 512
FIGURE_DPI = 300


def calculate_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate probability and thresholded metrics from held-out predictions."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must be aligned one-dimensional arrays")
    if not len(labels) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be a non-empty binary array")
    if len(np.unique(labels)) != 2:
        raise ValueError("Evaluation partitions must contain both binary classes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted class-1 probabilities must be finite values in [0, 1]")
    if not math.isfinite(float(threshold)) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be finite and in [0, 1]")

    predictions = probabilities >= threshold
    return {
        "f1_positive": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def aggregate_split_metrics(
    per_split: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Summarize split-level metrics with t intervals; one-run CIs remain undefined."""
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be strictly between zero and one")
    required = {"experiment", "model", "model_family", "representation", "n_test", *METRICS}
    if not required.issubset(per_split.columns):
        raise ValueError(f"Per-split metrics are missing columns: {sorted(required - set(per_split.columns))}")

    rows: list[dict[str, Any]] = []
    for (experiment, model), group in per_split.groupby(["experiment", "model"], sort=False):
        row: dict[str, Any] = {
            "experiment": experiment,
            "model": model,
            "model_family": group["model_family"].iloc[0],
            "representation": group["representation"].iloc[0],
            "n_splits": len(group),
            "n_eval_total": int(group["n_test"].sum()),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            half_width = (
                float(student_t.ppf((1 + confidence_level) / 2, len(values) - 1) * sd / math.sqrt(len(values)))
                if len(values) > 1
                else math.nan
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = sd
            row[f"{metric}_ci95_low"] = mean - half_width if math.isfinite(half_width) else math.nan
            row[f"{metric}_ci95_high"] = mean + half_width if math.isfinite(half_width) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_original_results(root: Path) -> dict[str, str]:
    preserved_roots = (
        root / "results" / "classical_baselines" / "herg_karim",
        root / "results" / "herg_karim",
    )
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for preserved_root in preserved_roots
        if preserved_root.exists()
        for path in sorted(preserved_root.rglob("*"))
        if path.is_file()
    }


def _validate_selection_trials(
    run_name: str,
    results: pd.DataFrame,
    trials: pd.DataFrame,
) -> None:
    required = {"split", "model", "selected", "validation_auroc", "parameters"}
    if not required.issubset(trials.columns):
        raise ValueError(f"{run_name} tuning trials are missing columns: {sorted(required - set(trials.columns))}")
    expected_keys = set(zip(results["split"], results["model"]))
    observed_keys = set(zip(trials["split"], trials["model"]))
    if expected_keys != observed_keys:
        raise ValueError(f"{run_name} tuning trials do not cover the result split/model pairs")

    for result in results.itertuples(index=False):
        group = trials[(trials["split"] == result.split) & (trials["model"] == result.model)]
        selected = group[group["selected"].astype(bool)]
        if len(selected) != 1:
            raise ValueError(f"{run_name}/{result.split}/{result.model}: expected one validation-selected trial")
        selected_trial = selected.iloc[0]
        best_score = float(group["validation_auroc"].max())
        if not np.isclose(float(selected_trial["validation_auroc"]), best_score, rtol=0, atol=1e-12):
            raise ValueError(f"{run_name}/{result.split}/{result.model}: selected trial is not best by validation AUROC")
        if not np.isclose(float(result.validation_selection_auroc), best_score, rtol=0, atol=1e-12):
            raise ValueError(f"{run_name}/{result.split}/{result.model}: saved selection score disagrees with tuning trials")
        if json.loads(str(selected_trial["parameters"])) != json.loads(str(result.selected_parameters)):
            raise ValueError(f"{run_name}/{result.split}/{result.model}: selected parameters disagree with tuning trials")
        if not math.isfinite(float(result.validation_threshold)) or not 0 <= result.validation_threshold <= 1:
            raise ValueError(f"{run_name}/{result.split}/{result.model}: invalid saved validation threshold")


def _validate_manifest_inputs(
    manifest: dict[str, Any],
    current_checksums: dict[str, str],
    run_name: str,
) -> None:
    key_map = {
        "feature_metadata": "feature_metadata",
        "fingerprints": "fingerprints",
        "descriptors": "descriptors",
        "feature_manifest": "feature_manifest",
        "frozen_splits": "frozen_splits",
        "experiment_config": "experiment_config",
    }
    recorded = manifest.get("input_sha256", {})
    if not recorded:
        raise ValueError(f"{run_name} manifest has no input checksums")
    for recorded_key, current_key in key_map.items():
        if recorded.get(recorded_key) != current_checksums[current_key]:
            raise ValueError(f"{run_name} baseline input checksum changed: {recorded_key}")


def _load_run_predictions(
    root: Path,
    data: ExperimentData,
    run_name: str,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_dir = root / "results" / "classical_baselines" / "herg_karim" / run_name
    manifest_path = run_dir / "experiment_manifest.json"
    results_path = run_dir / "per_split_results.csv"
    trials_path = run_dir / "tuning_trials.csv"
    for path in (manifest_path, results_path, trials_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required existing {run_name} artifact is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_type") != run_name:
        raise ValueError(f"{manifest_path}: run_type does not match its directory")
    _validate_manifest_inputs(manifest, data.input_checksums, run_name)
    if "Final evaluation only" not in str(manifest.get("test_set_use", "")):
        raise ValueError(f"{manifest_path}: missing held-out evaluation restriction")

    results = pd.read_csv(results_path)
    trials = pd.read_csv(trials_path)
    if set(results["model"]) != set(MODEL_ORDER):
        raise ValueError(f"{run_name}: expected all seven configured model tracks")
    expected_split_count = 1 if run_name == "pilot" else len(data.splits)
    if results["split"].nunique() != expected_split_count:
        raise ValueError(f"{run_name}: unexpected number of split results")
    _validate_selection_trials(run_name, results, trials)

    pilot_config = data.config["pilot"]
    pilot_source_split = pilot_config["source_split"]
    pilot_source_partition = pilot_config["source_partition"]
    prediction_root = Path(data.config["prediction_root"])
    metadata = data.metadata
    label_by_id = metadata.set_index("mol_id")["label"].astype(int)
    row_by_id = {mol_id: index for index, mol_id in enumerate(metadata["mol_id"].astype(str))}
    split_metrics: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for result in results.itertuples(index=False):
        split_name = str(result.split)
        model_name = str(result.model)
        prediction_path = prediction_root / split_name / f"{model_name}.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"{run_name} prediction file is missing: {prediction_path}")
        predictions = pd.read_csv(
            prediction_path,
            dtype={"mol_id": str},
            float_precision="round_trip",
        )
        if predictions.columns.tolist() != ["mol_id", "y_true", "p_pred"]:
            raise ValueError(f"{prediction_path}: prediction schema is not the shared molecule-level schema")
        if predictions["mol_id"].duplicated().any() or len(predictions) != int(result.n_test):
            raise ValueError(f"{prediction_path}: duplicated IDs or an unexpected prediction count")
        ids = predictions["mol_id"].astype(str)
        if not ids.isin(label_by_id.index).all():
            raise ValueError(f"{prediction_path}: contains molecule IDs absent from feature metadata")
        actual_labels = ids.map(label_by_id).to_numpy(dtype=np.int8)
        saved_labels = pd.to_numeric(predictions["y_true"], errors="coerce").to_numpy()
        if not np.array_equal(saved_labels, actual_labels):
            raise ValueError(f"{prediction_path}: saved labels disagree with curated molecule metadata")
        probabilities = pd.to_numeric(predictions["p_pred"], errors="coerce").to_numpy(dtype=np.float64)
        threshold = float(result.validation_threshold)

        if run_name == "full":
            if split_name not in data.splits:
                raise ValueError(f"{split_name}: unknown frozen scaffold split")
            expected_ids = set(metadata.iloc[data.splits[split_name]["test"]]["mol_id"].astype(str))
            if set(ids) != expected_ids:
                raise ValueError(f"{prediction_path}: molecule IDs do not equal the frozen test partition")
            novelty = dict(
                zip(
                    metadata.iloc[data.splits[split_name]["test"]]["mol_id"].astype(str),
                    data.splits[split_name]["test_novelty"],
                    strict=True,
                )
            )
            evaluation_partition = "official_frozen_test"
        else:
            if split_name != pilot_config["output_name"]:
                raise ValueError(f"{run_name}: unexpected pilot split name {split_name}")
            source_indices = data.splits[pilot_source_split][pilot_source_partition]
            source_ids = set(metadata.iloc[source_indices]["mol_id"].astype(str))
            if not set(ids).issubset(source_ids):
                raise ValueError("Pilot holdout contains molecules outside its frozen training source partition")
            if len(ids) != int(pilot_config["holdout_size"]):
                raise ValueError("Pilot holdout count disagrees with its recorded configuration")
            novelty = {}
            evaluation_partition = "pilot_holdout_within_frozen_train"

        metrics = calculate_classification_metrics(actual_labels, probabilities, threshold)
        # Recompute the baseline's original metrics to detect altered prediction artifacts.
        original_metric_names = ("auroc", "auprc", "balanced_accuracy", "mcc", "brier_score", "log_loss")
        if any(
            not np.isclose(metrics[name], float(getattr(result, name)), rtol=0, atol=1e-12)
            for name in original_metric_names
        ):
            raise ValueError(f"{prediction_path}: recomputed metrics disagree with the original split result")

        predicted_labels = (probabilities >= threshold).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(actual_labels, predicted_labels, labels=[0, 1]).ravel()
        confusion_rows.append(
            {
                "experiment": run_name,
                "split": split_name,
                "model": model_name,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "threshold_from_validation": threshold,
                "n_test": len(ids),
            }
        )
        split_row = {
            "experiment": run_name,
            "split": split_name,
            "model": model_name,
            "model_family": result.model_family,
            "representation": result.representation,
            "n_train": int(result.n_train),
            "n_validation": int(result.n_validation),
            "n_test": len(ids),
            "validation_selection_auroc": float(result.validation_selection_auroc),
            "validation_threshold": threshold,
            **metrics,
        }
        split_metrics.append(split_row)
        output = predictions[["mol_id", "y_true", "p_pred"]].copy()
        output.insert(0, "experiment", run_name)
        output.insert(1, "split", split_name)
        output.insert(2, "evaluation_partition", evaluation_partition)
        output.insert(3, "model", model_name)
        output["decision_threshold_from_validation"] = threshold
        output["y_pred"] = predicted_labels
        output["structural_novelty"] = output["mol_id"].map(novelty)
        output["novelty_bin"] = output["structural_novelty"].map(_novelty_bin)
        prediction_rows.append(output)

        if result.checkpoint_path and not Path(result.checkpoint_path).is_file():
            raise FileNotFoundError(f"{run_name} checkpoint is missing: {result.checkpoint_path}")

    split_frame = pd.DataFrame(split_metrics)
    confusion_frame = pd.DataFrame(confusion_rows)
    prediction_frame = pd.concat(prediction_rows, ignore_index=True)
    return split_frame, confusion_frame, prediction_frame, manifest


def _novelty_bin(value: Any) -> str | None:
    if pd.isna(value):
        return None
    score = float(value)
    if 0.0 <= score <= 0.4:
        return "familiar"
    if 0.4 < score <= 0.6:
        return "intermediate"
    if 0.6 < score <= 1.0:
        return "novel"
    raise ValueError(f"Frozen structural novelty must be in [0, 1], got {score}")


def _figure_style() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }


def _save_figure(
    figure: plt.Figure,
    path: Path,
    figures: list[dict[str, Any]],
    category: str,
    title: str,
    description: str,
    experiment: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    figures.append(
        {
            "category": category,
            "experiment": experiment,
            "title": title,
            "path": path.as_posix(),
            "dpi": FIGURE_DPI,
            "description": description,
        }
    )


def _scaffold_counts(curated: pd.DataFrame) -> pd.DataFrame:
    scaffold = curated["scaffold"].astype("string")
    scaffold = scaffold.fillna("").str.strip().replace("", "Acyclic/no scaffold")
    counts = scaffold.value_counts(dropna=False).rename_axis("scaffold").reset_index(name="molecule_count")
    return counts


def _generate_dataset_figures(
    root: Path,
    data: ExperimentData,
    destination: Path,
    figures: list[dict[str, Any]],
) -> dict[str, Any]:
    curated_path = root / "data" / "processed" / "herg_karim.csv"
    if not curated_path.is_file():
        raise FileNotFoundError(f"Curated hERG_Karim data required for scaffold analysis is missing: {curated_path}")
    curated = pd.read_csv(curated_path)
    metadata = data.metadata
    if len(curated) != len(metadata):
        raise ValueError("Curated dataset row count does not align with frozen feature metadata")
    if not np.array_equal(curated["orig_id"].astype(str).to_numpy(), metadata["Drug_ID"].astype(str).to_numpy()):
        raise ValueError("Curated dataset row identity does not align with feature metadata")
    if not np.array_equal(pd.to_numeric(curated["label"]).to_numpy(), data.labels):
        raise ValueError("Curated dataset labels do not align with feature metadata")
    descriptors_path = Path(data.config["feature_dir"]) / "rdkit_descriptors.csv"
    descriptors = pd.read_csv(descriptors_path, dtype={"mol_id": str})
    if descriptors["mol_id"].astype(str).tolist() != metadata["mol_id"].astype(str).tolist():
        raise ValueError("Descriptor IDs do not align with feature metadata")
    descriptors = descriptors.loc[:, data.descriptor_names]

    dataset_dir = destination / "dataset_analysis"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_figure_style()):
        label_counts = pd.Series(data.labels).value_counts().reindex([0, 1], fill_value=0)
        figure, axis = plt.subplots(figsize=(5.4, 4.1), constrained_layout=True)
        bars = axis.bar(CLASS_LABELS, label_counts.to_numpy(), color=["#4C78A8", "#E45756"])
        axis.bar_label(bars, fmt="%d", padding=3)
        axis.set_ylabel("Molecules")
        axis.set_title("Curated hERG_Karim class distribution")
        axis.grid(axis="y", alpha=0.25)
        _save_figure(
            figure,
            dataset_dir / "class_distribution.png",
            figures,
            "dataset_analysis",
            "Curated class distribution",
            "Counts of curated non-blockers and blockers from the frozen hERG_Karim feature metadata.",
        )

        descriptor_count = len(data.descriptor_names)
        figure, axes = plt.subplots(4, 4, figsize=(12.5, 10), constrained_layout=True)
        for index, axis in enumerate(axes.flat):
            if index >= descriptor_count:
                axis.set_visible(False)
                continue
            name = data.descriptor_names[index]
            for label, color, label_text in (
                (0, "#4C78A8", "Non-blocker"),
                (1, "#E45756", "Blocker"),
            ):
                values = descriptors.loc[data.labels == label, name].to_numpy(dtype=np.float64)
                axis.hist(values, bins=32, density=True, alpha=0.58, color=color, label=label_text)
            axis.set_title(name)
            axis.set_ylabel("Density")
            axis.grid(axis="y", alpha=0.2)
        axes.flat[0].legend(frameon=False)
        figure.suptitle("RDKit descriptor distributions by class")
        _save_figure(
            figure,
            dataset_dir / "descriptor_distributions.png",
            figures,
            "dataset_analysis",
            "RDKit descriptor distributions by class",
            "All documented RDKit descriptor values, shown separately by class; no feature scaling or selection is applied.",
        )

        scaffold_counts = _scaffold_counts(curated)
        scaffold_counts.to_csv(root / "results" / "reports" / "herg_karim" / "scaffold_counts.csv", index=False)
        named = scaffold_counts[scaffold_counts["scaffold"] != "Acyclic/no scaffold"]
        top = named.head(20).copy()
        omitted = int(named.iloc[20:]["molecule_count"].sum())
        chart = top.iloc[::-1].copy()
        labels = [
            f"Scaffold {len(top) - index} (n={count})"
            for index, count in enumerate(chart["molecule_count"])
        ]
        figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        axis.barh(labels, chart["molecule_count"], color="#72B7B2")
        if omitted:
            axis.barh(["Other scaffolds"], [omitted], color="#B279A2")
        no_scaffold = int(scaffold_counts.loc[scaffold_counts["scaffold"] == "Acyclic/no scaffold", "molecule_count"].sum())
        if no_scaffold:
            axis.barh(["Acyclic/no scaffold"], [no_scaffold], color="#F2CF5B")
        axis.set_xlabel("Molecules")
        axis.set_title("Bemis–Murcko scaffold distribution (most frequent)")
        axis.grid(axis="x", alpha=0.25)
        _save_figure(
            figure,
            dataset_dir / "scaffold_distribution.png",
            figures,
            "dataset_analysis",
            "Scaffold distribution",
            "Top 20 non-empty scaffold groups, a combined long-tail group, and molecules without a scaffold. Exact scaffold counts are in scaffold_counts.csv.",
        )

        split_rows: list[dict[str, Any]] = []
        for split_name, split in data.splits.items():
            for partition in ("train", "val", "test"):
                labels = data.labels[np.asarray(split[partition], dtype=np.int64)]
                split_rows.append(
                    {
                        "split": split_name,
                        "partition": partition,
                        "n_molecules": len(labels),
                        "n_negative": int((labels == 0).sum()),
                        "n_positive": int((labels == 1).sum()),
                        "positive_rate": float(labels.mean()),
                    }
                )
        split_balance = pd.DataFrame(split_rows)
        split_balance.to_csv(root / "results" / "reports" / "herg_karim" / "class_balance_by_split.csv", index=False)
        figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), constrained_layout=True)
        for axis, partition in zip(axes, ("train", "val", "test")):
            frame = split_balance[split_balance["partition"] == partition]
            x = np.arange(len(frame))
            negative = frame["n_negative"].to_numpy()
            positive = frame["n_positive"].to_numpy()
            axis.bar(x, negative, color="#4C78A8", label="Class 0")
            axis.bar(x, positive, bottom=negative, color="#E45756", label="Class 1")
            for position, row in enumerate(frame.itertuples(index=False)):
                axis.text(position, row.n_molecules + max(frame["n_molecules"]) * 0.015, f"{row.positive_rate:.1%}", ha="center", fontsize=7)
            axis.set_xticks(x, [name.replace("scaffold_", "") for name in frame["split"]], rotation=30, ha="right")
            axis.set_title(partition.capitalize())
            axis.set_ylabel("Molecules")
            axis.grid(axis="y", alpha=0.2)
        axes[0].legend(frameon=False)
        figure.suptitle("Class balance in frozen scaffold partitions (positive-rate labels)")
        _save_figure(
            figure,
            dataset_dir / "class_balance_by_partition.png",
            figures,
            "dataset_analysis",
            "Class balance across train, validation, and test",
            "Counts and positive rates for each of the five frozen scaffold repeats. The evaluation partitions are shown for descriptive analysis only.",
        )

        similarity_info = _sample_molecular_similarities(
            data,
            dataset_dir,
            root / "results" / "reports" / "herg_karim",
        )
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
        axes[0].hist(similarity_info["pair_values"], bins=np.linspace(0, 1, 41), color="#4C78A8", alpha=0.9)
        axes[0].set_title(f"Random unique molecule pairs (n={len(similarity_info['pair_values']):,})")
        axes[0].set_xlabel("Morgan Tanimoto similarity")
        axes[0].set_ylabel("Pair count")
        axes[1].hist(similarity_info["nearest_values"], bins=np.linspace(0, 1, 41), color="#F58518", alpha=0.9)
        axes[1].set_title(f"Sampled molecules vs. nearest dataset neighbor (n={len(similarity_info['nearest_values']):,})")
        axes[1].set_xlabel("Nearest-neighbor Tanimoto similarity")
        axes[1].set_ylabel("Molecule count")
        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle("Molecular fingerprint similarity and neighborhood diversity")
        _save_figure(
            figure,
            dataset_dir / "molecular_similarity_distributions.png",
            figures,
            "dataset_analysis",
            "Molecular similarity distributions",
            "Chirality-aware ECFP4 (radius 2, 2048 bits): a deterministic sample of unique random molecule pairs and nearest-neighbor similarities for a deterministic query sample. Sampling details and raw values are recorded in molecular_similarity_sample.csv.",
        )

    return {
        "curated_molecule_count": len(curated),
        "class_counts": {str(key): int(value) for key, value in label_counts.items()},
        "descriptor_names": list(data.descriptor_names),
        "scaffold_count_nonempty": int((scaffold_counts["scaffold"] != "Acyclic/no scaffold").sum()),
        "molecules_without_scaffold": int(no_scaffold),
        "similarity_sampling": {
            "seed": DEFAULT_SIMILARITY_SEED,
            "unique_random_pairs": len(similarity_info["pair_values"]),
            "nearest_neighbor_queries": len(similarity_info["nearest_values"]),
            "fingerprint": data.feature_manifest["fingerprint"],
        },
    }


def _sample_molecular_similarities(
    data: ExperimentData,
    dataset_dir: Path,
    reports_dir: Path,
) -> dict[str, np.ndarray]:
    metadata = data.metadata
    fingerprint_config = data.feature_manifest["fingerprint"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(fingerprint_config["radius"]),
        fpSize=int(fingerprint_config["n_bits"]),
        includeChirality=bool(fingerprint_config["use_chirality"]),
    )
    fingerprints = []
    for mol_id, smiles in zip(metadata["mol_id"], metadata["canonical_smiles"], strict=True):
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise ValueError(f"{mol_id}: canonical SMILES failed RDKit parsing during similarity analysis")
        fingerprints.append(generator.GetFingerprint(molecule))

    rng = np.random.default_rng(DEFAULT_SIMILARITY_SEED)
    pair_target = min(DEFAULT_RANDOM_PAIR_COUNT, len(fingerprints) * (len(fingerprints) - 1) // 2)
    unique_pairs: set[tuple[int, int]] = set()
    while len(unique_pairs) < pair_target:
        left, right = rng.integers(0, len(fingerprints), size=(2, max(4096, pair_target - len(unique_pairs))))
        unique_pairs.update((int(min(i, j)), int(max(i, j))) for i, j in zip(left, right, strict=True) if i != j)
        if len(unique_pairs) > pair_target:
            pair_list = sorted(unique_pairs)
            sampled_indices = rng.choice(len(pair_list), size=pair_target, replace=False)
            unique_pairs = {pair_list[int(index)] for index in sampled_indices}
    pairs = sorted(unique_pairs)
    pair_values = np.fromiter(
        (DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right]) for left, right in pairs),
        dtype=np.float64,
        count=len(pairs),
    )
    pair_frame = pd.DataFrame(
        {
            "similarity_type": "random_unique_pair",
            "mol_id": [metadata.iloc[left]["mol_id"] for left, _ in pairs],
            "reference_mol_id": [metadata.iloc[right]["mol_id"] for _, right in pairs],
            "tanimoto_similarity": pair_values,
        }
    )

    query_count = min(DEFAULT_NEIGHBOR_SAMPLE_COUNT, len(fingerprints))
    query_indices = np.sort(rng.choice(len(fingerprints), size=query_count, replace=False))
    nearest_rows: list[dict[str, Any]] = []
    nearest_values: list[float] = []
    for query_index in query_indices:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprints[int(query_index)], fingerprints),
            dtype=np.float64,
        )
        similarities[int(query_index)] = -1.0
        neighbor_index = int(np.argmax(similarities))
        value = float(similarities[neighbor_index])
        nearest_values.append(value)
        nearest_rows.append(
            {
                "similarity_type": "nearest_dataset_neighbor",
                "mol_id": metadata.iloc[int(query_index)]["mol_id"],
                "reference_mol_id": metadata.iloc[neighbor_index]["mol_id"],
                "tanimoto_similarity": value,
            }
        )
    raw = pd.concat([pair_frame, pd.DataFrame(nearest_rows)], ignore_index=True)
    raw.to_csv(reports_dir / "molecular_similarity_sample.csv", index=False)
    return {
        "pair_values": pair_values,
        "nearest_values": np.asarray(nearest_values, dtype=np.float64),
    }


def _plot_metric_comparison(
    aggregate: pd.DataFrame,
    experiment: str,
    metric: str,
    destination: Path,
    figures: list[dict[str, Any]],
) -> None:
    frame = aggregate[aggregate["experiment"] == experiment].set_index("model").reindex(MODEL_ORDER).reset_index()
    means = frame[f"{metric}_mean"].to_numpy(dtype=np.float64)
    lows = frame[f"{metric}_ci95_low"].to_numpy(dtype=np.float64)
    highs = frame[f"{metric}_ci95_high"].to_numpy(dtype=np.float64)
    errors = np.vstack((means - lows, highs - means))
    errors[~np.isfinite(errors)] = 0.0
    with plt.rc_context(_figure_style()):
        figure, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
        x = np.arange(len(frame))
        axis.bar(x, means, color=["#4C78A8"] * 3 + ["#F58518"] * 3 + ["#54A24B"], alpha=0.88)
        if np.isfinite(lows).any():
            axis.errorbar(x, means, yerr=errors, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
        axis.set_xticks(x, frame["model"], rotation=35, ha="right")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title(_experiment_title(experiment, f"{metric.replace('_', ' ').upper()} comparison"))
        axis.grid(axis="y", alpha=0.25)
        if metric in {"auroc", "auprc", "f1_positive", "f1_macro", "f1_weighted"}:
            axis.set_ylim(0, 1)
        _save_figure(
            figure,
            destination,
            figures,
            "model_comparison",
            f"{experiment.capitalize()} {metric.replace('_', ' ').upper()} comparison",
            f"Mean held-out {metric} by model with descriptive t-based 95% confidence intervals across repeats; pilot intervals are undefined for a single holdout.",
            experiment,
        )


def _experiment_title(experiment: str, title: str) -> str:
    if experiment == "pilot":
        return f"PILOT — training-partition holdout only; not independent test performance\n{title}"
    return f"hERG_Karim frozen scaffold repeats\n{title}"


def _plot_f1_comparison(
    aggregate: pd.DataFrame,
    experiment: str,
    output: Path,
    figures: list[dict[str, Any]],
) -> None:
    frame = aggregate[aggregate["experiment"] == experiment].set_index("model").reindex(MODEL_ORDER).reset_index()
    metrics = ("f1_positive", "f1_macro", "f1_weighted")
    labels = ("Positive-class F1", "Macro-F1", "Weighted F1")
    with plt.rc_context(_figure_style()):
        figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), constrained_layout=True, sharey=True)
        x = np.arange(len(frame))
        for axis, metric, label in zip(axes, metrics, labels, strict=True):
            means = frame[f"{metric}_mean"].to_numpy(dtype=np.float64)
            lows = frame[f"{metric}_ci95_low"].to_numpy(dtype=np.float64)
            highs = frame[f"{metric}_ci95_high"].to_numpy(dtype=np.float64)
            errors = np.vstack((means - lows, highs - means))
            errors[~np.isfinite(errors)] = 0.0
            axis.bar(x, means, color="#4C78A8", alpha=0.88)
            if np.isfinite(lows).any():
                axis.errorbar(x, means, yerr=errors, fmt="none", ecolor="#222222", capsize=2, linewidth=0.9)
            axis.set_title(label)
            axis.set_xticks(x, frame["model"], rotation=80, ha="right")
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("F1 score")
        figure.suptitle(_experiment_title(experiment, "F1 metrics"))
        _save_figure(
            figure,
            output,
            figures,
            "model_comparison",
            f"{experiment.capitalize()} F1-score comparison",
            "Positive-class, macro, and weighted F1 computed from validation-selected thresholds; bars show means and descriptive 95% t intervals where repeats permit.",
            experiment,
        )


def _plot_curves(
    prediction_frame: pd.DataFrame,
    experiment: str,
    destination_dir: Path,
    figures: list[dict[str, Any]],
) -> None:
    rows = prediction_frame[prediction_frame["experiment"] == experiment]
    with plt.rc_context(_figure_style()):
        fpr_grid = np.linspace(0, 1, 201)
        figure, axis = plt.subplots(figsize=(7.1, 5.8), constrained_layout=True)
        for model in MODEL_ORDER:
            group = rows[rows["model"] == model]
            curves = []
            aucs = []
            for _, split in group.groupby("split", sort=False):
                fpr, tpr, _ = roc_curve(split["y_true"], split["p_pred"])
                curves.append(np.interp(fpr_grid, fpr, tpr))
                aucs.append(roc_auc_score(split["y_true"], split["p_pred"]))
            values = np.asarray(curves)
            mean_curve = values.mean(axis=0)
            line, = axis.plot(fpr_grid, mean_curve, linewidth=1.6, label=f"{model} (AUROC={np.mean(aucs):.3f})")
            if len(values) > 1:
                sd = values.std(axis=0, ddof=1)
                axis.fill_between(fpr_grid, np.clip(mean_curve - sd, 0, 1), np.clip(mean_curve + sd, 0, 1), color=line.get_color(), alpha=0.12)
        axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1, label="Chance")
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
        axis.set_title(_experiment_title(experiment, "ROC curves"))
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, loc="lower right", fontsize=7)
        _save_figure(
            figure,
            destination_dir / "roc_curves.png",
            figures,
            "model_comparison",
            f"{experiment.capitalize()} ROC curves",
            "Each curve is the mean interpolated ROC across held-out split predictions; full-run ribbons show one split standard deviation. Pilot curves use only the training-partition holdout.",
            experiment,
        )

        recall_grid = np.linspace(0, 1, 201)
        figure, axis = plt.subplots(figsize=(7.1, 5.8), constrained_layout=True)
        for model in MODEL_ORDER:
            group = rows[rows["model"] == model]
            curves = []
            aps = []
            for _, split in group.groupby("split", sort=False):
                precision, recall, _ = precision_recall_curve(split["y_true"], split["p_pred"])
                curves.append(np.interp(recall_grid, recall[::-1], precision[::-1]))
                aps.append(average_precision_score(split["y_true"], split["p_pred"]))
            values = np.asarray(curves)
            mean_curve = values.mean(axis=0)
            line, = axis.plot(recall_grid, mean_curve, linewidth=1.6, label=f"{model} (AUPRC={np.mean(aps):.3f})")
            if len(values) > 1:
                sd = values.std(axis=0, ddof=1)
                axis.fill_between(recall_grid, np.clip(mean_curve - sd, 0, 1), np.clip(mean_curve + sd, 0, 1), color=line.get_color(), alpha=0.12)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
        axis.set_title(_experiment_title(experiment, "Precision–recall curves"))
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, loc="lower left", fontsize=7)
        _save_figure(
            figure,
            destination_dir / "precision_recall_curves.png",
            figures,
            "model_comparison",
            f"{experiment.capitalize()} precision-recall curves",
            "Each curve is the mean interpolated precision-recall curve across held-out split predictions; full-run ribbons show one split standard deviation. Pilot curves are not independent test performance.",
            experiment,
        )


def _plot_confusion_matrices(
    confusion_rows: pd.DataFrame,
    experiment: str,
    output: Path,
    figures: list[dict[str, Any]],
) -> None:
    frame = confusion_rows[confusion_rows["experiment"] == experiment]
    summed = frame.groupby("model", sort=False)[["tn", "fp", "fn", "tp"]].sum().reindex(MODEL_ORDER)
    with plt.rc_context(_figure_style()):
        figure, axes = plt.subplots(2, 4, figsize=(13.5, 6.7), constrained_layout=True)
        for axis, model in zip(axes.flat, MODEL_ORDER, strict=False):
            counts = summed.loc[model, ["tn", "fp", "fn", "tp"]].to_numpy(dtype=np.int64).reshape(2, 2)
            image = axis.imshow(counts, cmap="Blues")
            for (row, column), value in np.ndenumerate(counts):
                axis.text(column, row, f"{value:,}", ha="center", va="center", color="#111111", fontsize=10)
            axis.set_xticks([0, 1], ["Pred 0", "Pred 1"])
            axis.set_yticks([0, 1], ["True 0", "True 1"])
            axis.set_title(model)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axes.flat[-1].set_visible(False)
        figure.suptitle(_experiment_title(experiment, "Confusion counts pooled across available holdouts"))
        _save_figure(
            figure,
            output,
            figures,
            "model_comparison",
            f"{experiment.capitalize()} confusion matrices",
            "Counts are summed across held-out predictions using validation-selected thresholds. Full-repeat test molecules overlap across repeats, so pooled counts are descriptive, not independent observations.",
            experiment,
        )


def _plot_split_metrics(
    per_split: pd.DataFrame,
    experiment: str,
    output: Path,
    figures: list[dict[str, Any]],
) -> None:
    frame = per_split[per_split["experiment"] == experiment]
    metrics = ("auroc", "auprc", "f1_positive", "mcc")
    titles = ("AUROC", "AUPRC", "Positive-class F1", "MCC")
    with plt.rc_context(_figure_style()):
        figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
        split_names = frame["split"].drop_duplicates().tolist()
        x = np.arange(len(split_names))
        for axis, metric, title in zip(axes.flat, metrics, titles, strict=True):
            for model in MODEL_ORDER:
                group = frame[frame["model"] == model].set_index("split").reindex(split_names)
                axis.plot(x, group[metric], marker="o", linewidth=1.25, markersize=3.5, label=model)
            axis.set_title(title)
            axis.set_xticks(x, [re.sub(r"^scaffold_", "", value).replace("pilot_", "pilot ") for value in split_names], rotation=25, ha="right")
            axis.grid(alpha=0.25)
        axes[0, 0].set_ylabel("Score")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].legend(frameon=False, fontsize=7, loc="best")
        figure.suptitle(_experiment_title(experiment, "Split-level held-out metrics"))
        _save_figure(
            figure,
            output,
            figures,
            "split_analysis",
            f"{experiment.capitalize()} split-level performance",
            "Each point is a split-level held-out metric calculated from saved molecule predictions. The pilot panel represents one training-partition holdout only.",
            experiment,
        )


def _write_advanced_analysis_inputs(
    reports_dir: Path,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    input_path = reports_dir / "advanced_analysis_inputs.csv"
    predictions.to_csv(input_path, index=False)
    calibration_template = reports_dir / "conformal_calibration_template.csv"
    pd.DataFrame(
        columns=[
            "experiment",
            "split",
            "model",
            "mol_id",
            "y_true",
            "p_pred",
            "calibration_role",
            "nonconformity_score",
        ]
    ).to_csv(calibration_template, index=False)

    checkpoint_paths = {
        f"{run_name}/{row.split}/{row.model}": row.checkpoint_path
        for run_name in ("pilot", "full")
        for row in pd.read_csv(
            reports_dir.parent.parent / "classical_baselines" / "herg_karim" / run_name / "per_split_results.csv"
        ).itertuples(index=False)
        if row.checkpoint_path and Path(row.checkpoint_path).is_file()
    }
    manifest = {
        "status": "prepared_inputs_only_no_advanced_analysis_performed",
        "prediction_table": input_path.name,
        "prediction_columns": predictions.columns.tolist(),
        "model_checkpoints": checkpoint_paths,
        "feature_metadata": "data/processed/features/hERG_Karim/molecule_metadata.csv",
        "descriptors": "data/processed/features/hERG_Karim/rdkit_descriptors.csv",
        "fingerprints": "data/processed/features/hERG_Karim/ecfp4_2048.npz",
        "full_run_novelty_source": "frozen scaffold test_novelty values from data/splits/herg_karim_scaffold.json",
        "shap": "Use saved split checkpoint and its matching feature representation; no SHAP values were computed.",
        "calibration": "Saved test probabilities and labels are for evaluation only; fit calibration with validation/training data, never with test labels.",
        "conformal": {
            "template": calibration_template.name,
            "status": "empty_template",
            "instruction": "Generate dedicated train/validation calibration predictions before estimating nonconformity quantiles. Do not populate from held-out test labels.",
        },
        "novelty_vs_performance": "Full-run rows include frozen structural_novelty and novelty_bin; pilot rows have null novelty because their holdout was sampled within scaffold_r0 training.",
        "selection_integrity": "No test metric was used for hyperparameter selection or threshold choice; selection audit was checked against validation tuning_trials.csv.",
        "confidence_intervals": "Across-repeat intervals use the five dependent scaffold repeats descriptively; pilot has one holdout and no interval.",
    }
    (reports_dir / "advanced_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_markdown_report(
    report_path: Path,
    figures: list[dict[str, Any]],
    per_split: pd.DataFrame,
    aggregate: pd.DataFrame,
    dataset_summary: dict[str, Any],
) -> None:
    lines = [
        "# hERG_Karim scientific visualisation and metric report",
        "",
        "This report was generated from the existing molecule-level predictions. No baseline model was retrained, no feature or split was regenerated, and no test result was used for model selection.",
        "",
        "The full experiment contains five repeated frozen scaffold splits. Its 95% t intervals are descriptive because the repeats share molecules. The pilot is a single 512/128/128 training/validation/holdout experiment sampled only from the frozen `scaffold_r0` training partition; pilot results are **not independent test performance** and have no across-repeat confidence interval.",
        "",
        "## Dataset summary",
        "",
        f"- Curated molecules: {dataset_summary['curated_molecule_count']:,}.",
        f"- Class counts (0/1): {dataset_summary['class_counts'].get('0', 0):,} / {dataset_summary['class_counts'].get('1', 0):,}.",
        f"- Non-empty scaffold groups: {dataset_summary['scaffold_count_nonempty']:,}; molecules without a scaffold: {dataset_summary['molecules_without_scaffold']:,}.",
        f"- Similarity sample: {dataset_summary['similarity_sampling']['unique_random_pairs']:,} unique random pairs and {dataset_summary['similarity_sampling']['nearest_neighbor_queries']:,} nearest-neighbor queries using the frozen ECFP4 settings.",
        "",
        "## Figures",
        "",
    ]
    for figure in figures:
        lines.extend(
            [
                f"### {figure['title']}",
                "",
                f"- File: `{figure['path']}`",
                f"- Description: {figure['description']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Metric definitions and integrity",
            "",
            "- Precision, recall, positive-class F1, macro-F1, weighted F1, balanced accuracy, and MCC use the saved threshold selected on validation by balanced accuracy.",
            "- AUROC, average precision (AUPRC), Brier score, and log loss use saved class-1 probabilities.",
            "- Per-split metrics are recomputed and checked against the baseline's original six metrics; prediction IDs and labels are checked against curated metadata and frozen test membership.",
            "- Hyperparameter selections are checked against the best recorded validation-AUROC tuning trial. The thresholds are taken only from the baseline's recorded validation-selected thresholds.",
            "- Pilot held-out IDs are verified to belong to the frozen `scaffold_r0` training partition. Its predictions are presented separately from official test results.",
            "- See `complete_metric_comparison.csv`, `per_split_metrics.csv`, and `confusion_matrices.csv` for numeric tables.",
            "",
            "## Prepared advanced-analysis inputs (not analyses)",
            "",
            "The `advanced_analysis_inputs.csv` and `advanced_analysis_manifest.json` provide compatible prediction, feature, checkpoint, and frozen novelty references. No SHAP values, calibration curves, conformal sets, or novelty-performance analyses were produced. The conformal template is intentionally empty; test labels must not be used to estimate conformal quantiles.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def generate_scientific_reports(root: Path) -> dict[str, Any]:
    """Create metrics and publication figures from frozen inputs and saved predictions."""
    started = time.perf_counter()
    root = root.resolve()
    preserved_before = _snapshot_original_results(root)
    data = load_experiment_data(root)
    reports_dir = root / "results" / "reports" / "herg_karim"
    visualization_dir = root / "results" / "visualizations" / "herg_karim"
    reports_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    split_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    run_manifests: dict[str, dict[str, Any]] = {}
    for run_name in ("pilot", "full"):
        split_frame, confusion_frame, prediction_frame, manifest = _load_run_predictions(
            root,
            data,
            run_name,
            float(data.config["tuning"]["confidence_level"]),
        )
        split_frames.append(split_frame)
        confusion_frames.append(confusion_frame)
        prediction_frames.append(prediction_frame)
        run_manifests[run_name] = manifest

    per_split = pd.concat(split_frames, ignore_index=True)
    confusion = pd.concat(confusion_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    aggregate = aggregate_split_metrics(
        per_split,
        confidence_level=float(data.config["tuning"]["confidence_level"]),
    )
    per_split.to_csv(reports_dir / "per_split_metrics.csv", index=False)
    aggregate.to_csv(reports_dir / "complete_metric_comparison.csv", index=False)
    confusion.to_csv(reports_dir / "confusion_matrices.csv", index=False)
    predictions.to_csv(reports_dir / "advanced_analysis_inputs.csv", index=False)

    confusion_root = reports_dir / "confusion_matrices"
    for experiment in ("full", "pilot"):
        for row in confusion[confusion["experiment"] == experiment].itertuples(index=False):
            model_matrix = pd.DataFrame(
                [[row.tn, row.fp], [row.fn, row.tp]],
                index=["true_0", "true_1"],
                columns=["predicted_0", "predicted_1"],
            )
            path = confusion_root / experiment / f"{row.split}__{row.model}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            model_matrix.to_csv(path, index_label="actual_class")

    figures: list[dict[str, Any]] = []
    dataset_summary = _generate_dataset_figures(root, data, visualization_dir, figures)
    for experiment in ("full", "pilot"):
        comparison_dir = visualization_dir / "model_comparison" / experiment
        comparison_dir.mkdir(parents=True, exist_ok=True)
        _plot_curves(predictions, experiment, comparison_dir, figures)
        _plot_metric_comparison(
            aggregate,
            experiment,
            "auroc",
            comparison_dir / "auroc_comparison.png",
            figures,
        )
        _plot_metric_comparison(
            aggregate,
            experiment,
            "auprc",
            comparison_dir / "auprc_comparison.png",
            figures,
        )
        _plot_f1_comparison(
            aggregate,
            experiment,
            comparison_dir / "f1_comparison.png",
            figures,
        )
        _plot_metric_comparison(
            aggregate,
            experiment,
            "mcc",
            comparison_dir / "mcc_comparison.png",
            figures,
        )
        _plot_confusion_matrices(
            confusion,
            experiment,
            comparison_dir / "confusion_matrices.png",
            figures,
        )
        split_destination = visualization_dir / "split_analysis" / experiment
        _plot_split_metrics(
            per_split,
            experiment,
            split_destination / "split_level_performance.png",
            figures,
        )

    advanced_manifest = _write_advanced_analysis_inputs(
        reports_dir,
        predictions,
    )
    reporting_input_checksums = {
        "curated_processed_dataset": _sha256(root / "data" / "processed" / "herg_karim.csv")
    }
    preserved_after = _snapshot_original_results(root)
    if preserved_before != preserved_after:
        changed = sorted(
            key
            for key in set(preserved_before) | set(preserved_after)
            if preserved_before.get(key) != preserved_after.get(key)
        )
        raise RuntimeError(f"Existing baseline results changed during report generation: {changed}")
    figure_manifest = {
        "dataset": "hERG_Karim",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "generator": "scripts/09_generate_scientific_visualizations.py",
        "figure_dpi": FIGURE_DPI,
        "input_sha256": data.input_checksums,
        "reporting_input_sha256": reporting_input_checksums,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "rdkit": rdBase.rdkitVersion,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "preserved_original_results": {
            "status": "unchanged",
            "file_count": len(preserved_after),
            "sha256_of_path_digest_manifest": hashlib.sha256(
                json.dumps(preserved_after, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "report_files": {
            "complete_metric_comparison": "results/reports/herg_karim/complete_metric_comparison.csv",
            "per_split_metrics": "results/reports/herg_karim/per_split_metrics.csv",
            "confusion_matrices": "results/reports/herg_karim/confusion_matrices.csv",
            "scientific_visualization_report": "results/reports/herg_karim/scientific_visualization_report.md",
            "advanced_analysis_manifest": "results/reports/herg_karim/advanced_analysis_manifest.json",
        },
        "baseline_runs": {
            name: {
                "run_type": manifest["run_type"],
                "generated_at_utc": manifest["generated_at_utc"],
                "test_set_use": manifest["test_set_use"],
            }
            for name, manifest in run_manifests.items()
        },
        "dataset_summary": dataset_summary,
        "metric_names": list(METRICS),
        "confidence_interval": {
            "level": float(data.config["tuning"]["confidence_level"]),
            "method": "two-sided Student-t interval across split-level metrics",
            "caveat": "Full repeats share molecules; intervals are descriptive, not superiority tests. A single pilot holdout has undefined SD and CI.",
        },
        "selection_audit": {
            "status": "passed",
            "method": "For each model and split, the recorded selected trial was verified to attain the maximum validation AUROC and match the saved selected parameters; saved thresholds are the baseline's validation-selected thresholds.",
            "test_role": "Test predictions and labels were read only to calculate final metrics and figures.",
        },
        "advanced_analysis": advanced_manifest,
        "figures": figures,
    }
    (reports_dir / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _build_markdown_report(
        reports_dir / "scientific_visualization_report.md",
        figures,
        per_split,
        aggregate,
        dataset_summary,
    )

    preserved_final = _snapshot_original_results(root)
    if preserved_before != preserved_final:
        changed = sorted(
            key
            for key in set(preserved_before) | set(preserved_final)
            if preserved_before.get(key) != preserved_final.get(key)
        )
        raise RuntimeError(f"Existing baseline results changed during report generation: {changed}")
    LOGGER.info(
        "Generated %d figures, %d per-split metric rows, and %d confusion matrices in %.1f seconds",
        len(figures),
        len(per_split),
        len(confusion),
        time.perf_counter() - started,
    )
    return {
        "reports_dir": reports_dir,
        "visualization_dir": visualization_dir,
        "per_split_metrics": per_split,
        "aggregate_metrics": aggregate,
        "confusion_matrices": confusion,
        "predictions": predictions,
        "figures": figures,
        "preserved_result_count":         len(preserved_final),
        "input_checksums": data.input_checksums,
    }
