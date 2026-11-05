"""Validation-selected classical hERG_Karim baselines on frozen scaffold splits."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.stats import t as student_t
from rdkit import rdBase
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    log_loss,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

LOGGER = logging.getLogger(__name__)
MODEL_REPRESENTATIONS = {
    "lightgbm": ("descriptors", "fingerprints", "combined"),
    "random_forest": ("descriptors", "fingerprints", "combined"),
    "tanimoto_svm": ("fingerprints",),
}
METRICS = (
    "auroc",
    "auprc",
    "balanced_accuracy",
    "mcc",
    "brier_score",
    "log_loss",
)


@dataclass(frozen=True)
class ExperimentData:
    metadata: pd.DataFrame
    labels: np.ndarray
    fingerprints: np.ndarray
    descriptors: np.ndarray
    descriptor_names: tuple[str, ...]
    splits: dict[str, dict[str, list[int]]]
    feature_manifest: dict[str, Any]
    config: dict[str, Any]
    input_checksums: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required baseline input is missing: {path}")
    return path


def _read_splits(path: Path, split_names: list[str], n_rows: int) -> dict[str, Any]:
    splits = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(splits, dict) or set(splits) != set(split_names):
        raise ValueError(f"{path}: expected exactly these splits: {split_names}")
    for split_name, split in splits.items():
        if not isinstance(split, dict) or not {"train", "val", "test", "test_novelty"}.issubset(split):
            raise ValueError(f"{split_name}: expected train, val, test, and test_novelty")
        partitions = []
        for partition in ("train", "val", "test"):
            indices = split[partition]
            if not isinstance(indices, list) or any(type(index) is not int for index in indices):
                raise ValueError(f"{split_name}/{partition}: expected integer row positions")
            if len(indices) != len(set(indices)) or any(index < 0 or index >= n_rows for index in indices):
                raise ValueError(f"{split_name}/{partition}: duplicate or out-of-range indices")
            partitions.append(set(indices))
        if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"{split_name}: train, validation, and test overlap")
        if set.union(*partitions) != set(range(n_rows)):
            raise ValueError(f"{split_name}: does not partition the feature rows exactly")
        if len(split["test_novelty"]) != len(split["test"]):
            raise ValueError(f"{split_name}: test novelty values are not aligned to test rows")
        novelty = np.asarray(split["test_novelty"], dtype=float)
        if not np.isfinite(novelty).all() or ((novelty < 0) | (novelty > 1)).any():
            raise ValueError(f"{split_name}: test novelty values must be finite in [0, 1]")
    return splits


def load_experiment_data(root: Path, config_path: Path | None = None) -> ExperimentData:
    """Load and align existing feature artifacts; never regenerate them."""
    root = root.resolve()
    config_file = config_path or root / "configs" / "classical_baselines.yaml"
    if not config_file.is_absolute():
        config_file = root / config_file
    config = _read_yaml(config_file)
    if config.get("dataset_key") != "herg_karim":
        raise ValueError("Classical baselines are configured for hERG_Karim only")
    for path_key in ("feature_dir", "split_file", "output_root", "prediction_root"):
        config[path_key] = str((root / config[path_key]).resolve())

    feature_dir = root / config["feature_dir"]
    metadata_path = _require_file(feature_dir / "molecule_metadata.csv")
    fingerprints_path = _require_file(feature_dir / "ecfp4_2048.npz")
    descriptors_path = _require_file(feature_dir / "rdkit_descriptors.csv")
    feature_manifest_path = _require_file(feature_dir / "feature_manifest.json")
    split_path = _require_file(root / config["split_file"])
    metadata = pd.read_csv(metadata_path, dtype={"mol_id": str, "Drug_ID": str})
    descriptors_frame = pd.read_csv(descriptors_path, dtype={"mol_id": str})
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    descriptor_names = tuple(feature_manifest["descriptor_names"])
    with np.load(fingerprints_path, allow_pickle=False) as archive:
        fingerprints = archive["fingerprints"]
        fingerprint_ids = archive["mol_ids"].astype(str)

    required_metadata = {"mol_id", "label", "canonical_smiles", "split_assignment"}
    if not required_metadata.issubset(metadata.columns):
        raise ValueError(f"{metadata_path}: missing columns {sorted(required_metadata - set(metadata.columns))}")
    molecule_ids = metadata["mol_id"].astype(str).to_numpy()
    if len(set(molecule_ids)) != len(molecule_ids) or not np.array_equal(fingerprint_ids, molecule_ids):
        raise ValueError("Feature molecule IDs are duplicated or fingerprint rows are misaligned")
    if descriptors_frame["mol_id"].astype(str).tolist() != molecule_ids.tolist():
        raise ValueError("Descriptor rows do not follow the metadata molecule order")
    if not set(descriptor_names).issubset(descriptors_frame.columns):
        raise ValueError("Descriptor file does not contain every descriptor in the feature manifest")
    descriptors = descriptors_frame.loc[:, descriptor_names].to_numpy(dtype=np.float32)
    if fingerprints.ndim != 2 or fingerprints.shape[0] != len(metadata):
        raise ValueError("Fingerprint matrix has an unexpected shape")
    if fingerprints.shape[1] != feature_manifest["fingerprint"]["n_bits"]:
        raise ValueError("Fingerprint bit dimension disagrees with feature manifest")
    if not np.isin(fingerprints, (0, 1)).all():
        raise ValueError("Morgan fingerprint matrix must contain only binary values")
    if not np.isfinite(descriptors).all():
        raise ValueError("Descriptor matrix contains NaN or infinite values")

    numeric_labels = pd.to_numeric(metadata["label"], errors="coerce")
    if numeric_labels.isna().any() or not set(numeric_labels.unique()).issubset({0, 1}):
        raise ValueError("Metadata labels must be binary 0/1")
    labels = numeric_labels.to_numpy(dtype=np.int8)
    split_names = config["split_names"]
    splits = _read_splits(split_path, split_names, len(metadata))
    for split_name, split in splits.items():
        for partition in ("train", "val", "test"):
            for index in split[partition]:
                assignment = json.loads(metadata.iloc[index]["split_assignment"])
                if assignment.get(split_name) != partition:
                    raise ValueError(f"{molecule_ids[index]}: metadata split assignment mismatch")
        canonical = metadata["canonical_smiles"].astype(str).to_numpy()
        structure_sets = [
            {canonical[index] for index in split[partition]}
            for partition in ("train", "val", "test")
        ]
        if any(structure_sets[i] & structure_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"{split_name}: canonical structures leak across partitions")

    input_checksums = {
        "feature_metadata": _sha256(metadata_path),
        "fingerprints": _sha256(fingerprints_path),
        "descriptors": _sha256(descriptors_path),
        "feature_manifest": _sha256(feature_manifest_path),
        "frozen_splits": _sha256(split_path),
        "experiment_config": _sha256(config_file),
    }
    return ExperimentData(
        metadata=metadata,
        labels=labels,
        fingerprints=fingerprints,
        descriptors=descriptors,
        descriptor_names=descriptor_names,
        splits=splits,
        feature_manifest=feature_manifest,
        config=config,
        input_checksums=input_checksums,
    )


def tanimoto_kernel_matrix(
    query: np.ndarray,
    reference: np.ndarray,
    block_size: int = 256,
) -> np.ndarray:
    """Build a dense binary Tanimoto kernel in row blocks.

    The result occupies 8 * n_query * n_reference bytes. A temporary float32
    product is limited to block_size * n_reference values.
    """
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("Tanimoto inputs must be 2-D matrices with the same bit dimension")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not np.isin(query, (0, 1)).all() or not np.isin(reference, (0, 1)).all():
        raise ValueError("Tanimoto inputs must contain only binary fingerprint values")
    query_float = np.asarray(query, dtype=np.float32)
    reference_float = np.asarray(reference, dtype=np.float32)
    query_counts = query_float.sum(axis=1, dtype=np.float32)
    reference_counts = reference_float.sum(axis=1, dtype=np.float32)
    result = np.empty((len(query_float), len(reference_float)), dtype=np.float64)
    for start in range(0, len(query_float), block_size):
        stop = min(start + block_size, len(query_float))
        intersections = query_float[start:stop] @ reference_float.T
        unions = (
            query_counts[start:stop, None]
            + reference_counts[None, :]
            - intersections
        )
        np.divide(intersections, unions, out=intersections, where=unions > 0)
        intersections[unions == 0] = 1.0
        result[start:stop] = intersections
    return result


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Select the validation threshold with best balanced accuracy."""
    above_maximum = np.nextafter(float(probabilities.max()), math.inf)
    candidates = np.unique(np.concatenate(([0.0, above_maximum], probabilities)))
    scores = [
        balanced_accuracy_score(labels, probabilities >= threshold)
        for threshold in candidates
    ]
    best = max(
        range(len(candidates)),
        key=lambda index: (
            scores[index],
            -abs(float(candidates[index]) - 0.5),
            -float(candidates[index]),
        ),
    )
    return float(candidates[best]), float(scores[best])


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted class-1 probabilities must be finite values in [0, 1]")
    if len(np.unique(labels)) != 2:
        raise ValueError("All evaluation partitions must contain both binary classes")
    hard_predictions = probabilities >= threshold
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, hard_predictions)),
        "mcc": float(matthews_corrcoef(labels, hard_predictions)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def _feature_matrix(data: ExperimentData, representation: str) -> np.ndarray:
    if representation == "descriptors":
        return data.descriptors
    if representation == "fingerprints":
        return data.fingerprints
    if representation == "combined":
        return np.concatenate((data.fingerprints, data.descriptors), axis=1)
    raise ValueError(f"Unknown feature representation: {representation}")


def _feature_names(data: ExperimentData, representation: str) -> list[str]:
    fingerprint_names = [
        f"ecfp4_{bit_index:04d}"
        for bit_index in range(data.fingerprints.shape[1])
    ]
    if representation == "descriptors":
        return list(data.descriptor_names)
    if representation == "fingerprints":
        return fingerprint_names
    if representation == "combined":
        return [*fingerprint_names, *data.descriptor_names]
    raise ValueError(f"Unknown feature representation: {representation}")


def _forest_candidates(config: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    candidates = [dict(candidate) for candidate in config["tuning"]["forest"]]
    return candidates[:limit] if limit is not None else candidates


def _lightgbm_candidates(config: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    candidates = [dict(candidate) for candidate in config["tuning"]["lightgbm"]]
    return candidates[:limit] if limit is not None else candidates


def _svm_candidates(config: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    svm_config = config["tuning"]["tanimoto_svm"]
    candidates = [
        {"C": float(c_value), "class_weight": class_weight}
        for c_value in svm_config["c_values"]
        for class_weight in svm_config["class_weights"]
    ]
    return candidates[:limit] if limit is not None else candidates


def _make_model(
    model_family: str,
    params: dict[str, Any],
    seed: int,
    config: dict[str, Any],
) -> Any:
    if model_family == "random_forest":
        return RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=seed,
            n_jobs=config.get("n_jobs", 4),
        )
    if model_family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required. Install the declared dependency with "
                ".venv\\Scripts\\python.exe -m pip install lightgbm==4.6.0"
            ) from exc
        return LGBMClassifier(
            **params,
            objective="binary",
            random_state=seed,
            n_jobs=config.get("n_jobs", 4),
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    if model_family == "tanimoto_svm":
        svm_config = config["tuning"]["tanimoto_svm"]
        return SVC(
            C=params["C"],
            kernel="precomputed",
            class_weight=params["class_weight"],
            random_state=seed,
            cache_size=svm_config["cache_size_mb"],
        )
    raise ValueError(f"Unknown model family: {model_family}")


def _fit_svm_with_training_calibration(
    train_kernel: np.ndarray,
    validation_kernel: np.ndarray,
    train_labels: np.ndarray,
    params: dict[str, Any],
    seed: int,
    config: dict[str, Any],
) -> tuple[SVC, LogisticRegression, np.ndarray, float]:
    """Fit an SVC and Platt calibrator using out-of-fold training margins only."""
    svm_config = config["tuning"]["tanimoto_svm"]
    splitter = StratifiedKFold(
        n_splits=svm_config["calibration_folds"],
        shuffle=True,
        random_state=seed,
    )
    oof_margins = np.empty(len(train_labels), dtype=np.float64)
    started = time.perf_counter()
    with threadpool_limits(limits=config.get("n_jobs", 4)):
        for fit_indices, calibration_indices in splitter.split(train_kernel, train_labels):
            fold_train_kernel = train_kernel[np.ix_(fit_indices, fit_indices)]
            fold_validation_kernel = train_kernel[np.ix_(calibration_indices, fit_indices)]
            fold_model = _make_model("tanimoto_svm", params, seed, config)
            fold_model.fit(fold_train_kernel, train_labels[fit_indices])
            oof_margins[calibration_indices] = fold_model.decision_function(
                fold_validation_kernel
            )
            del fold_model, fold_train_kernel, fold_validation_kernel

        calibrator = LogisticRegression(
            C=svm_config["calibration_logistic_c"],
            random_state=seed,
            solver="lbfgs",
        )
        calibrator.fit(oof_margins.reshape(-1, 1), train_labels)
        estimator = _make_model("tanimoto_svm", params, seed, config)
        estimator.fit(train_kernel, train_labels)
        validation_margins = estimator.decision_function(validation_kernel)
        validation_probabilities = calibrator.predict_proba(
            validation_margins.reshape(-1, 1)
        )[:, 1]
    elapsed = time.perf_counter() - started
    return estimator, calibrator, validation_probabilities, elapsed


def _pilot_partitions(
    split: dict[str, Any],
    labels: np.ndarray,
    pilot_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    source = np.asarray(split[pilot_config["source_partition"]], dtype=np.int64)
    sample_size = pilot_config["sample_size"]
    if sample_size > len(source):
        raise ValueError(f"Pilot sample size {sample_size} exceeds the source partition size {len(source)}")
    sampled, _ = train_test_split(
        source,
        train_size=sample_size,
        random_state=seed,
        stratify=labels[source],
    )
    pilot_train, remainder = train_test_split(
        sampled,
        train_size=pilot_config["train_size"],
        random_state=seed + 1,
        stratify=labels[sampled],
    )
    pilot_val, pilot_test = train_test_split(
        remainder,
        train_size=pilot_config["validation_size"],
        random_state=seed + 2,
        stratify=labels[remainder],
    )
    if len(pilot_val) != pilot_config["validation_size"] or len(pilot_test) != pilot_config["holdout_size"]:
        raise ValueError("Pilot partition sizes do not match configuration")
    return {
        "train": pilot_train.tolist(),
        "val": pilot_val.tolist(),
        "test": pilot_test.tolist(),
        "source_partition": pilot_config["source_partition"],
    }


def _kernel_memory_estimate_mb(
    n_train: int,
    n_val: int,
    n_test: int,
    cache_size_mb: int,
) -> float:
    """Conservative estimate for matrices, SVC copies, and solver cache."""
    kernel_bytes = 8 * n_train * (n_train + max(n_val, n_test))
    return kernel_bytes * 1.75 / (1024**2) + cache_size_mb


def _fit_evaluate_model(
    data: ExperimentData,
    split_name: str,
    partition_indices: dict[str, Any],
    model_family: str,
    representation: str,
    candidates: list[dict[str, Any]],
    save_model: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    config = data.config
    train_indices = np.asarray(partition_indices["train"], dtype=np.int64)
    val_indices = np.asarray(partition_indices["val"], dtype=np.int64)
    test_indices = np.asarray(partition_indices["test"], dtype=np.int64)
    y_train = data.labels[train_indices]
    y_val = data.labels[val_indices]
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError(f"{split_name}: train and validation must each contain both classes")

    if model_family == "tanimoto_svm":
        svm_config = config["tuning"]["tanimoto_svm"]
        memory_estimate = _kernel_memory_estimate_mb(
            len(train_indices),
            len(val_indices),
            len(test_indices),
            svm_config["cache_size_mb"],
        )
        if memory_estimate > svm_config["max_kernel_memory_mb"]:
            raise MemoryError(
                f"{split_name}: estimated Tanimoto-SVM kernel memory "
                f"{memory_estimate:.1f} MiB exceeds configured limit "
                f"{svm_config['max_kernel_memory_mb']} MiB"
            )
        train_x = data.fingerprints[train_indices]
        val_x = data.fingerprints[val_indices]
        kernel_construction_started = time.perf_counter()
        train_kernel = tanimoto_kernel_matrix(
            train_x,
            train_x,
            svm_config["kernel_block_size"],
        )
        val_kernel = tanimoto_kernel_matrix(
            val_x,
            train_x,
            svm_config["kernel_block_size"],
        )
        kernel_construction_seconds = time.perf_counter() - kernel_construction_started
        x_train = train_kernel
    else:
        kernel_construction_seconds = 0.0
        matrix = _feature_matrix(data, representation)
        if model_family == "lightgbm":
            feature_names = _feature_names(data, representation)
            x_train = pd.DataFrame(
                matrix[train_indices],
                columns=feature_names,
                copy=False,
            )
            x_val = pd.DataFrame(
                matrix[val_indices],
                columns=feature_names,
                copy=False,
            )
        else:
            x_train = matrix[train_indices]
            x_val = matrix[val_indices]
        train_kernel = None
        val_kernel = None

    tuning_rows: list[dict[str, Any]] = []
    best_score = -math.inf
    best_params: dict[str, Any] | None = None
    best_candidate_fit_seconds = 0.0
    tuning_fit_seconds = 0.0
    model_seed = int(config["seed"]) + int(split_name[-1])

    for candidate_index, params in enumerate(candidates):
        estimator = _make_model(model_family, params, model_seed, config)
        started = time.perf_counter()
        with threadpool_limits(limits=config.get("n_jobs", 4)):
            estimator.fit(x_train, y_train)
            if model_family == "tanimoto_svm":
                validation_scores = estimator.decision_function(val_kernel)
            else:
                validation_scores = estimator.predict_proba(x_val)[:, 1]
        fit_seconds = time.perf_counter() - started
        tuning_fit_seconds += fit_seconds
        score = float(roc_auc_score(y_val, validation_scores))
        tuning_rows.append(
            {
                "split": split_name,
                "model": f"{model_family}_{representation}",
                "candidate_index": candidate_index,
                "parameters": json.dumps(params, sort_keys=True),
                "validation_auroc": score,
                "fit_seconds": fit_seconds,
                "selected": False,
            }
        )
        if score > best_score:
            best_score = score
            best_params = dict(params)
            best_candidate_fit_seconds = fit_seconds
        del estimator

    if best_params is None:
        raise ValueError(f"{split_name}: no candidates configured for {model_family}")
    selected_index = next(
        index for index, row in enumerate(tuning_rows)
        if json.loads(row["parameters"]) == best_params
    )
    tuning_rows[selected_index]["selected"] = True

    calibration_fit_seconds = 0.0
    calibrator: LogisticRegression | None = None
    if model_family == "tanimoto_svm":
        estimator, calibrator, validation_probabilities, calibration_fit_seconds = (
            _fit_svm_with_training_calibration(
                train_kernel,
                val_kernel,
                y_train,
                best_params,
                model_seed,
                config,
            )
        )
    else:
        estimator = _make_model(model_family, best_params, model_seed, config)
        started = time.perf_counter()
        with threadpool_limits(limits=config.get("n_jobs", 4)):
            estimator.fit(x_train, y_train)
        calibration_fit_seconds = time.perf_counter() - started
        validation_probabilities = estimator.predict_proba(x_val)[:, 1]
    threshold, validation_balanced_accuracy = select_threshold(y_val, validation_probabilities)

    test_labels = data.labels[test_indices]
    if len(np.unique(test_labels)) != 2:
        raise ValueError(f"{split_name}: test must contain both classes")
    prediction_started = time.perf_counter()
    if model_family == "tanimoto_svm":
        test_kernel = tanimoto_kernel_matrix(
            data.fingerprints[test_indices],
            data.fingerprints[train_indices],
            config["tuning"]["tanimoto_svm"]["kernel_block_size"],
        )
        test_margins = estimator.decision_function(test_kernel)
        if calibrator is None:
            raise RuntimeError("Tanimoto-SVM calibration model was not fitted")
        test_probabilities = calibrator.predict_proba(test_margins.reshape(-1, 1))[:, 1]
    else:
        x_test_values = _feature_matrix(data, representation)[test_indices]
        x_test = (
            pd.DataFrame(
                x_test_values,
                columns=_feature_names(data, representation),
                copy=False,
            )
            if model_family == "lightgbm"
            else x_test_values
        )
        test_probabilities = estimator.predict_proba(x_test)[:, 1]
    prediction_seconds = time.perf_counter() - prediction_started
    test_metrics = calculate_metrics(test_labels, test_probabilities, threshold)

    model_name = f"{model_family}_{representation}"
    prediction_frame = pd.DataFrame(
        {
            "mol_id": data.metadata.iloc[test_indices]["mol_id"].astype(str).to_numpy(),
            "y_true": test_labels.astype(int),
            "p_pred": test_probabilities.astype(float),
        }
    )
    if list(prediction_frame.columns) != ["mol_id", "y_true", "p_pred"]:
        raise ValueError("Prediction output does not match the shared prediction schema")

    checkpoint_path = ""
    if save_model:
        model_dir = Path(config["output_root"]) / split_name / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "estimator": estimator,
            "model": model_name,
            "representation": representation,
            "descriptor_names": data.descriptor_names if representation != "fingerprints" else (),
            "fingerprint_bits": data.fingerprints.shape[1] if representation != "descriptors" else 0,
            "training_mol_ids": data.metadata.iloc[train_indices]["mol_id"].astype(str).tolist(),
            "decision_threshold": threshold,
            "selected_parameters": best_params,
            "seed": model_seed,
        }
        if model_family == "tanimoto_svm":
            checkpoint["training_fingerprints"] = data.fingerprints[train_indices]
            checkpoint["probability_calibrator"] = calibrator
        checkpoint_file = model_dir / f"{model_name}.joblib"
        joblib.dump(checkpoint, checkpoint_file, compress=3)
        checkpoint_path = checkpoint_file.as_posix()

    row: dict[str, Any] = {
        "split": split_name,
        "model": model_name,
        "model_family": model_family,
        "representation": representation,
        "n_train": len(train_indices),
        "n_validation": len(val_indices),
        "n_test": len(test_indices),
        "test_positive_rate": float(test_labels.mean()),
        "selected_parameters": json.dumps(best_params, sort_keys=True),
        "validation_selection_auroc": best_score,
        "validation_threshold": threshold,
        "validation_balanced_accuracy": validation_balanced_accuracy,
        "training_time_seconds": (
            tuning_fit_seconds + calibration_fit_seconds + kernel_construction_seconds
        ),
        "selected_candidate_fit_seconds": best_candidate_fit_seconds,
        "final_fit_seconds": calibration_fit_seconds,
        "probability_refit_seconds": calibration_fit_seconds if model_family == "tanimoto_svm" else 0.0,
        "tuning_fit_seconds": tuning_fit_seconds,
        "kernel_construction_time_seconds": kernel_construction_seconds,
        "prediction_time_seconds": prediction_seconds,
        "checkpoint_path": checkpoint_path,
    }
    row.update(test_metrics)
    if model_family == "tanimoto_svm":
        row["estimated_peak_kernel_memory_mib"] = _kernel_memory_estimate_mb(
            len(train_indices),
            len(val_indices),
            len(test_indices),
            config["tuning"]["tanimoto_svm"]["cache_size_mb"],
        )

    prediction_dir = Path(config["prediction_root"]) / split_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(prediction_dir / f"{model_name}.csv", index=False)
    del estimator, x_train
    if model_family == "tanimoto_svm":
        del train_kernel, val_kernel, test_kernel
    else:
        del x_val
    return row, tuning_rows, prediction_frame


def _aggregate_results(results: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric_fields = [
        *METRICS,
        "training_time_seconds",
        "prediction_time_seconds",
        "kernel_construction_time_seconds",
        "validation_selection_auroc",
    ]
    for model, group in results.groupby("model", sort=False):
        row: dict[str, Any] = {
            "model": model,
            "model_family": group["model_family"].iloc[0],
            "representation": group["representation"].iloc[0],
            "n_repeats": len(group),
            "n_test_total_across_repeats": int(group["n_test"].sum()),
        }
        for field in numeric_fields:
            values = group[field].to_numpy(dtype=float)
            mean = float(np.mean(values))
            standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            if len(values) > 1:
                half_width = float(
                    student_t.ppf((1 + confidence_level) / 2, df=len(values) - 1)
                    * standard_deviation
                    / math.sqrt(len(values))
                )
            else:
                half_width = math.nan
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = standard_deviation
            row[f"{field}_ci95_low"] = mean - half_width if math.isfinite(half_width) else math.nan
            row[f"{field}_ci95_high"] = mean + half_width if math.isfinite(half_width) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _make_performance_plot(aggregate: pd.DataFrame, output_path: Path) -> None:
    metrics = ("auroc", "auprc", "balanced_accuracy", "mcc", "brier_score", "log_loss")
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics):
        means = aggregate[f"{metric}_mean"].to_numpy(dtype=float)
        lower = aggregate[f"{metric}_ci95_low"].to_numpy(dtype=float)
        upper = aggregate[f"{metric}_ci95_high"].to_numpy(dtype=float)
        errors = np.vstack((means - lower, upper - means))
        errors[~np.isfinite(errors)] = 0.0
        labels = aggregate["model"].tolist()
        axis.bar(np.arange(len(labels)), means, color="#4472C4", alpha=0.85)
        axis.errorbar(
            np.arange(len(labels)),
            means,
            yerr=errors,
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=1,
        )
        axis.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right")
        axis.set_title(metric.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("hERG_Karim classical baselines: repeated scaffold splits")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rdkit": rdBase.rdkitVersion,
        "scikit_learn": sklearn.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
    }
    try:
        versions["lightgbm"] = version("lightgbm")
    except PackageNotFoundError:
        versions["lightgbm"] = "missing"
    return versions


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_baselines(
    root: Path,
    config_path: Path | None = None,
    pilot: bool = False,
) -> dict[str, Any]:
    """Run validation-only tuning and then produce held-out predictions."""
    root = root.resolve()
    config_file = config_path or root / "configs" / "classical_baselines.yaml"
    if not config_file.is_absolute():
        config_file = root / config_file
    data = load_experiment_data(root, config_file)
    config = data.config
    split_names = config["split_names"]
    output_root = Path(config["output_root"])
    prediction_root = Path(config["prediction_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    selected_splits = [config["pilot"]["source_split"]] if pilot else split_names
    if pilot:
        split_name = selected_splits[0]
        partitions = _pilot_partitions(
            data.splits[split_name],
            data.labels,
            config["pilot"],
            int(config["seed"]),
        )
        split_entries = [(config["pilot"]["output_name"], partitions)]
        candidate_limit = int(config["pilot"]["candidates_per_track"])
    else:
        split_entries = [(name, data.splits[name]) for name in selected_splits]
        candidate_limit = None

    per_split_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    model_configs: dict[str, dict[str, Any]] = {}
    for split_name, partitions in split_entries:
        LOGGER.info(
            "%s: train=%d validation=%d held-out=%d%s",
            split_name,
            len(partitions["train"]),
            len(partitions["val"]),
            len(partitions["test"]),
            " (sampled only from the frozen training partition)" if pilot else "",
        )
        for model_family, representations in MODEL_REPRESENTATIONS.items():
            if model_family == "lightgbm":
                candidates = _lightgbm_candidates(config, candidate_limit)
            elif model_family == "random_forest":
                candidates = _forest_candidates(config, candidate_limit)
            else:
                candidates = _svm_candidates(config, candidate_limit)
            for representation in representations:
                model_name = f"{model_family}_{representation}"
                result, trials, _ = _fit_evaluate_model(
                    data,
                    split_name,
                    partitions,
                    model_family,
                    representation,
                    candidates,
                    save_model=True,
                )
                result["experiment"] = "pilot" if pilot else "full"
                per_split_rows.append(result)
                tuning_rows.extend(trials)
                model_configs[model_name] = {
                    "candidate_count_per_representation": len(candidates),
                    "candidates": candidates,
                    "selected_parameters_by_split": {
                        **model_configs.get(model_name, {}).get("selected_parameters_by_split", {}),
                        split_name: result["selected_parameters"],
                    },
                }
                LOGGER.info(
                    "%s %s: validation AUROC=%.4f, test AUROC=%.4f",
                    split_name,
                    model_name,
                    result["validation_selection_auroc"],
                    result["auroc"],
                )

    results = pd.DataFrame(per_split_rows)
    trials_frame = pd.DataFrame(tuning_rows)
    aggregate = _aggregate_results(
        results,
        float(config["tuning"]["confidence_level"]),
    )
    result_dir = output_root / ("pilot" if pilot else "full")
    result_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(result_dir / "per_split_results.csv", index=False)
    trials_frame.to_csv(result_dir / "tuning_trials.csv", index=False)
    aggregate.to_csv(result_dir / "aggregate_results.csv", index=False)
    _make_performance_plot(aggregate, result_dir / "performance.png")

    current_versions = _package_versions()
    feature_versions = data.feature_manifest.get("versions", {})
    version_warnings = {
        name: {
            "feature_generation": feature_versions.get(name),
            "baseline_runtime": current_versions.get(name),
        }
        for name in ("numpy", "pandas", "rdkit")
        if feature_versions.get(name) is not None
        and current_versions.get(name) is not None
        and feature_versions.get(name) != current_versions.get(name)
    }
    manifest = {
        "experiment": "classical_hERG_Karim_baselines",
        "run_type": "pilot" if pilot else "full",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "dataset": "hERG_Karim",
        "feature_molecule_count": len(data.metadata),
        "split_counts": {
            name: {key: len(data.splits[name][key]) for key in ("train", "val", "test")}
            for name in split_names
        },
        "run_split_counts": {
            name: {key: len(parts[key]) for key in ("train", "val", "test")}
            for name, parts in split_entries
        },
        "input_sha256": data.input_checksums,
        "feature_generation_versions": feature_versions,
        "baseline_runtime_versions": current_versions,
        "runtime_version_warnings": version_warnings,
        "fingerprint": data.feature_manifest["fingerprint"],
        "descriptor_names": list(data.descriptor_names),
        "selection_metric": config["tuning"]["selection_metric"],
        "threshold_metric": config["tuning"]["decision_threshold_metric"],
        "confidence_interval": {
            "method": config["tuning"]["ci_method"],
            "level": config["tuning"]["confidence_level"],
            "caveat": "Descriptive across five dependent repeated scaffold splits; not a superiority test.",
        },
        "tuning_budget": {
            "lightgbm": len(config["tuning"]["lightgbm"]) * 3,
            "random_forest": len(config["tuning"]["forest"]) * 3,
            "tanimoto_svm": len(_svm_candidates(config, None)),
            "pilot_candidates_per_track": candidate_limit if pilot else None,
        },
        "model_fixed_parameters": {
            "lightgbm": {
                "objective": "binary",
                "n_jobs": config["n_jobs"],
                "deterministic": True,
                "force_col_wise": True,
                "random_state": "seed + repeat index",
            },
            "random_forest": {
                "class_weight": "balanced",
                "bootstrap": True,
                "n_jobs": config["n_jobs"],
                "random_state": "seed + repeat index",
            },
            "tanimoto_svm": {
                "kernel": "precomputed Tanimoto",
                "probability_calibration": config["tuning"]["tanimoto_svm"][
                    "probability_calibration"
                ],
                "calibration_folds": config["tuning"]["tanimoto_svm"]["calibration_folds"],
                "calibration_logistic_c": config["tuning"]["tanimoto_svm"][
                    "calibration_logistic_c"
                ],
                "cache_size_mib": config["tuning"]["tanimoto_svm"]["cache_size_mb"],
            },
        },
        "model_configurations": model_configs,
        "svm_kernel_memory": {
            "formula_bytes": "8 * n_train * (n_train + max(n_validation, n_test))",
            "conservative_peak_multiplier": 1.75,
            "cache_mib": config["tuning"]["tanimoto_svm"]["cache_size_mb"],
            "configured_limit_mib": config["tuning"]["tanimoto_svm"]["max_kernel_memory_mb"],
            "kernel_block_size": config["tuning"]["tanimoto_svm"]["kernel_block_size"],
        },
        "outputs": {
            "per_split_results": (result_dir / "per_split_results.csv").as_posix(),
            "tuning_trials": (result_dir / "tuning_trials.csv").as_posix(),
            "aggregate_results": (result_dir / "aggregate_results.csv").as_posix(),
            "performance_plot": (result_dir / "performance.png").as_posix(),
            "predictions_root": prediction_root.as_posix(),
            "checkpoints_root": output_root.as_posix(),
        },
        "test_set_use": "Final evaluation only; no test metric or label is used for model selection or threshold choice.",
    }
    (result_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved baseline results to %s", result_dir)
    return {
        "results": results,
        "aggregate": aggregate,
        "manifest_path": result_dir / "experiment_manifest.json",
        "result_dir": result_dir,
    }
