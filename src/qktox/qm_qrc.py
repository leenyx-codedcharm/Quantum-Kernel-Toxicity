"""Classical readouts and matched random-feature controls for QM-QRC pilots."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC, SVC


@dataclass
class ClassicalRandomReservoir:
    """Fixed tanh recurrent map with an output dimension matched to a QRC layer."""

    input_dimension: int
    output_dimension_per_layer: int = 15
    seed: int = 20261119
    recurrent_scale: float = 0.5
    input_weights: np.ndarray = field(init=False, repr=False)
    recurrent_weights: np.ndarray = field(init=False, repr=False)
    bias: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.input_dimension <= 0 or self.output_dimension_per_layer <= 0:
            raise ValueError("random-reservoir dimensions must be positive")
        if self.recurrent_scale < 0:
            raise ValueError("recurrent_scale must be non-negative")
        rng = np.random.default_rng(self.seed)
        self.input_weights = rng.normal(
            0.0,
            1.0 / np.sqrt(self.input_dimension),
            size=(self.input_dimension, self.output_dimension_per_layer),
        )
        self.recurrent_weights = rng.normal(
            0.0,
            self.recurrent_scale / np.sqrt(self.output_dimension_per_layer),
            size=(self.output_dimension_per_layer, self.output_dimension_per_layer),
        )
        self.bias = rng.uniform(-np.pi, np.pi, size=self.output_dimension_per_layer)

    def transform(self, features: np.ndarray, layers: int) -> np.ndarray:
        """Return all recurrent states, concatenated to match the requested depth."""
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.input_dimension:
            raise ValueError("features have an unexpected random-reservoir input dimension")
        if not np.isfinite(values).all() or not len(values):
            raise ValueError("features must be non-empty and finite")
        if layers not in (2, 4, 6):
            raise ValueError("layers must be one of 2, 4, or 6")

        state = np.zeros((len(values), self.output_dimension_per_layer), dtype=np.float64)
        states = []
        input_drive = values @ self.input_weights + self.bias
        for _ in range(layers):
            state = np.tanh(input_drive + state @ self.recurrent_weights)
            states.append(state.copy())
        return np.concatenate(states, axis=1)

    def configuration(self) -> dict[str, Any]:
        """Return reproducible fixed weights and dimensions."""
        return {
            "architecture": "classical_random_recurrent_reservoir",
            "input_dimension": self.input_dimension,
            "output_dimension_per_layer": self.output_dimension_per_layer,
            "seed": self.seed,
            "recurrent_scale": self.recurrent_scale,
            "input_weights": self.input_weights.tolist(),
            "recurrent_weights": self.recurrent_weights.tolist(),
            "bias": self.bias.tolist(),
            "trainable_reservoir_parameters": False,
        }


def build_readout(name: str, regularization: float, seed: int) -> Any:
    """Construct a readout with training-only probability calibration as needed."""
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if name == "ridge":
        estimator = RidgeClassifier(alpha=1.0 / regularization)
    elif name == "logistic_regression":
        return LogisticRegression(
            C=regularization,
            max_iter=2000,
            random_state=seed,
            solver="liblinear",
        )
    elif name == "rbf_svm":
        return SVC(
            C=regularization,
            kernel="rbf",
            probability=True,
            random_state=seed,
        )
    elif name == "linear_svm":
        estimator = LinearSVC(C=regularization, random_state=seed, max_iter=10000)
    else:
        raise ValueError(f"unsupported readout: {name}")

    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=5)


def positive_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    """Extract validated class-1 probabilities from a fitted readout."""
    values = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if values.shape != (len(features), len(classes)) or len(positive) != 1:
        raise ValueError("readout returned an unexpected probability shape/classes")
    probabilities = values[:, int(positive[0])]
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("readout probabilities must be finite values in [0, 1]")
    return probabilities


def select_validation_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Maximize validation balanced accuracy with deterministic tie breaking."""
    y_true = _validate_binary_labels(labels, "validation labels")
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.shape != y_true.shape or not np.isfinite(scores).all():
        raise ValueError("validation probabilities must be finite and aligned with labels")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("validation probabilities must be in [0, 1]")

    candidates = np.unique(
        np.concatenate(([0.0, np.nextafter(float(scores.max()), np.inf)], scores))
    )
    accuracies = [
        balanced_accuracy_score(y_true, scores >= threshold)
        for threshold in candidates
    ]
    best = max(
        range(len(candidates)),
        key=lambda index: (
            accuracies[index],
            -abs(float(candidates[index]) - 0.5),
            -float(candidates[index]),
        ),
    )
    return float(candidates[best]), float(accuracies[best])


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate binary classification and probability metrics."""
    y_true = _validate_binary_labels(labels, "evaluation labels")
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.shape != y_true.shape or not np.isfinite(scores).all():
        raise ValueError("probabilities must be finite and aligned with labels")
    if ((scores < 0) | (scores > 1)).any() or not 0 <= threshold <= 1:
        raise ValueError("probabilities and threshold must be in [0, 1]")
    predicted = scores >= threshold
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "positive_f1": float(f1_score(y_true, predicted, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "cohen_kappa": float(cohen_kappa_score(y_true, predicted)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores, labels=[0, 1])),
    }


def tune_readout(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    readout_names: tuple[str, ...],
    regularization_values: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    """Select regularization and readout by validation AUROC only."""
    y_train = _validate_binary_labels(train_labels, "training labels")
    y_validation = _validate_binary_labels(validation_labels, "validation labels")
    x_train = _validate_matrix(train_features, "training features")
    x_validation = _validate_matrix(validation_features, "validation features")
    if x_train.shape[1] != x_validation.shape[1] or len(x_train) != len(y_train):
        raise ValueError("training features and labels are not aligned")
    if len(x_validation) != len(y_validation):
        raise ValueError("validation features and labels are not aligned")
    if len(np.unique(y_train, return_counts=True)[1]) < 2:
        raise ValueError("training data must contain at least two examples per class for calibration")
    if not readout_names or not regularization_values:
        raise ValueError("at least one readout and regularization value are required")

    best: dict[str, Any] | None = None
    trials = []
    for name in readout_names:
        for regularization in regularization_values:
            model = build_readout(name, regularization, seed)
            start = perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = perf_counter() - start
            probabilities = positive_probabilities(model, x_validation)
            validation_auc = float(roc_auc_score(y_validation, probabilities))
            trial = {
                "readout": name,
                "regularization": float(regularization),
                "validation_auroc": validation_auc,
                "fit_seconds": fit_seconds,
            }
            trials.append(trial)
            if best is None or validation_auc > best["validation_auroc"]:
                threshold, threshold_score = select_validation_threshold(
                    y_validation, probabilities
                )
                best = {
                    "readout": name,
                    "regularization": float(regularization),
                    "validation_auroc": validation_auc,
                    "threshold": threshold,
                    "validation_balanced_accuracy": threshold_score,
                    "validation_metrics": calculate_metrics(
                        y_validation, probabilities, threshold
                    ),
                    "validation_probabilities": probabilities,
                    "model": model,
                }
    if best is None:
        raise RuntimeError("readout tuning produced no candidate")
    best["trials"] = trials
    best["tuning_seconds"] = float(sum(item["fit_seconds"] for item in trials))
    return best


def _validate_binary_labels(labels: np.ndarray, description: str) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or not len(values) or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{description} must be a non-empty 1-D binary array")
    if set(np.unique(values)) != {0, 1}:
        raise ValueError(f"{description} must contain both binary classes")
    return values.astype(np.int8, copy=False)


def _validate_matrix(features: np.ndarray, description: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{description} must be a non-empty finite 2-D matrix")
    return values
