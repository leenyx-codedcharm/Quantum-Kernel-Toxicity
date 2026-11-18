import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.kernels import tanimoto_kernel_matrix
from qktox.quantum_pilot import (
    apply_train_preprocessor,
    calculate_pilot_metrics,
    fit_train_preprocessor,
    sample_partition_indices,
    select_validation_threshold,
    validate_square_kernel,
)


def test_partition_sampling_is_reproducible_and_confined():
    partition = list(range(100, 200))
    sample = sample_partition_indices(partition, 24, seed=19)
    np.testing.assert_array_equal(sample, sample_partition_indices(partition, 24, seed=19))
    assert len(sample) == 24
    assert set(sample).issubset(partition)
    assert np.all(np.diff(sample) > 0)


def test_tanimoto_kernel_is_exact_symmetric_and_handles_empty_fingerprints():
    fingerprints = np.array(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    kernel = tanimoto_kernel_matrix(fingerprints, fingerprints, block_size=2)
    np.testing.assert_allclose(
        kernel,
        np.array([[1.0, 1 / 3, 0.0], [1 / 3, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(kernel, kernel.T, rtol=0, atol=0)


def test_scaling_and_pca_parameters_are_fit_from_training_rows_only():
    train = np.arange(60, dtype=np.float64).reshape(10, 6)
    parameters = fit_train_preprocessor(train, n_components=4)
    assert parameters["scaler_mean"].tolist() == train.mean(axis=0).tolist()
    validation = np.full((2, 6), 1e9)
    transformed = apply_train_preprocessor(validation, parameters)
    assert transformed.shape == (2, 4)
    assert np.isfinite(transformed).all()
    assert parameters["pca_components"].shape == (4, 6)


def test_validation_threshold_and_metrics_match_selected_binary_cutoff():
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, score = select_validation_threshold(labels, probabilities)
    assert score == 1.0
    metrics = calculate_pilot_metrics(labels, probabilities, threshold)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["positive_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["brier_score"] < 0.1
    assert metrics["log_loss"] < 0.5


def test_square_kernel_validation_checks_psd_and_diagonal():
    diagnostics = validate_square_kernel(np.array([[1.0, 0.25], [0.25, 1.0]]))
    assert diagnostics["symmetric"]
    assert diagnostics["diagonal_ok"]
    assert diagnostics["positive_semidefinite"]
    with np.testing.assert_raises_regex(ValueError, "min_eigenvalue"):
        validate_square_kernel(np.array([[1.0, 2.0], [2.0, 1.0]]))
