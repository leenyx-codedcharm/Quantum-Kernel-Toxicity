"""Numerical helpers for the isolated fidelity-kernel pilot."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def sample_partition_indices(
    indices: list[int],
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """Choose sorted, reproducible row positions from one frozen partition."""
    if sample_size <= 0 or sample_size > len(indices):
        raise ValueError("sample_size must be positive and no larger than the partition")
    if len(indices) != len(set(indices)):
        raise ValueError("partition indices must be unique")
    return np.sort(
        np.random.default_rng(seed).choice(indices, size=sample_size, replace=False)
    ).astype(np.int64)


def fit_train_preprocessor(
    train_features: np.ndarray,
    n_components: int = 4,
) -> dict[str, np.ndarray]:
    """Fit scaling and PCA on training rows only and return reusable parameters."""
    features = np.asarray(train_features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("training features must be a finite 2-D matrix")
    if n_components <= 0 or n_components >= min(features.shape):
        raise ValueError("n_components must be positive and smaller than each training dimension")

    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    pca = PCA(n_components=n_components, svd_solver="full").fit(scaled)
    return {
        "scaler_mean": scaler.mean_.copy(),
        "scaler_scale": scaler.scale_.copy(),
        "pca_mean": pca.mean_.copy(),
        "pca_components": pca.components_.copy(),
        "pca_explained_variance": pca.explained_variance_.copy(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.copy(),
        "pca_singular_values": pca.singular_values_.copy(),
    }


def apply_train_preprocessor(
    features: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply previously saved scaling/PCA parameters without fitting."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("features must be a finite 2-D matrix")
    if values.shape[1] != len(parameters["scaler_mean"]):
        raise ValueError("feature dimension does not match the saved preprocessing parameters")
    scaled = (values - parameters["scaler_mean"]) / parameters["scaler_scale"]
    return (scaled - parameters["pca_mean"]) @ parameters["pca_components"].T


def select_validation_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Select the threshold maximizing validation balanced accuracy."""
    y_true = np.asarray(labels, dtype=np.int8)
    p_pred = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or p_pred.shape != y_true.shape or len(y_true) == 0:
        raise ValueError("labels and probabilities must be non-empty 1-D arrays of equal length")
    if not np.isfinite(p_pred).all() or ((p_pred < 0) | (p_pred > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("threshold selection requires both binary classes")
    candidates = np.unique(
        np.concatenate(
            ([0.0, np.nextafter(float(p_pred.max()), np.inf)], p_pred)
        )
    )
    scores = [
        balanced_accuracy_score(y_true, p_pred >= threshold)
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


def calculate_pilot_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate the preregistered binary pilot metrics."""
    y_true = np.asarray(labels, dtype=np.int8)
    p_pred = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or p_pred.shape != y_true.shape or len(y_true) == 0:
        raise ValueError("labels and probabilities must be non-empty 1-D arrays of equal length")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("pilot metrics require both binary classes")
    if not np.isfinite(p_pred).all() or ((p_pred < 0) | (p_pred > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    y_pred = p_pred >= threshold
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, p_pred)),
        "auprc": float(average_precision_score(y_true, p_pred)),
        "positive_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, p_pred)),
        "log_loss": float(log_loss(y_true, p_pred, labels=[0, 1])),
    }


def validate_square_kernel(
    matrix: np.ndarray,
    *,
    diagonal_value: float = 1.0,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Check a square Gram matrix for finite values, symmetry, diagonal, and PSD."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not len(values):
        raise ValueError("kernel Gram matrix must be a non-empty square matrix")
    if not np.isfinite(values).all():
        raise ValueError("kernel Gram matrix contains non-finite values")
    symmetric = bool(np.allclose(values, values.T, rtol=0.0, atol=tolerance))
    diagonal_ok = bool(
        np.allclose(np.diag(values), diagonal_value, rtol=0.0, atol=tolerance)
    )
    min_eigenvalue = float(np.linalg.eigvalsh((values + values.T) / 2.0).min())
    positive_semidefinite = min_eigenvalue >= -tolerance
    if not symmetric or not diagonal_ok or not positive_semidefinite:
        raise ValueError(
            "invalid kernel Gram matrix: "
            f"symmetric={symmetric}, diagonal_ok={diagonal_ok}, "
            f"min_eigenvalue={min_eigenvalue}"
        )
    return {
        "symmetric": symmetric,
        "diagonal_value": diagonal_value,
        "diagonal_ok": diagonal_ok,
        "minimum_eigenvalue": min_eigenvalue,
        "positive_semidefinite": positive_semidefinite,
        "tolerance": tolerance,
    }


def validate_cross_kernel(matrix: np.ndarray) -> None:
    """Check that a rectangular train-query kernel has finite values."""
    values = np.asarray(matrix)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("cross-kernel matrix must be a finite 2-D matrix")
