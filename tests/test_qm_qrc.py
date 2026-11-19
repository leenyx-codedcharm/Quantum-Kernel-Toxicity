import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.qm_qrc import (
    ClassicalRandomReservoir,
    calculate_metrics,
    positive_probabilities,
    select_validation_threshold,
    tune_readout,
)


def test_classical_random_reservoir_is_fixed_and_dimension_matched():
    features = np.arange(20, dtype=np.float64).reshape(5, 4) / 20
    reservoir = ClassicalRandomReservoir(input_dimension=4, seed=31)
    first = reservoir.transform(features, layers=2)
    np.testing.assert_array_equal(first, reservoir.transform(features, layers=2))
    assert first.shape == (5, 30)
    assert reservoir.transform(features, layers=6).shape == (5, 90)
    assert reservoir.configuration()["trainable_reservoir_parameters"] is False


def test_threshold_and_metrics_cover_f1_mcc_kappa_and_probability_scores():
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    threshold, balanced_accuracy = select_validation_threshold(labels, probabilities)
    metrics = calculate_metrics(labels, probabilities, threshold)

    assert balanced_accuracy == 1.0
    assert metrics["positive_f1"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["cohen_kappa"] == 1.0


def test_validation_tuning_uses_training_fits_and_returns_probability_model():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(60, 5))
    labels = (features[:, 0] > 0).astype(np.int8)
    validation = rng.normal(size=(30, 5))
    validation_labels = (validation[:, 0] > 0).astype(np.int8)

    result = tune_readout(
        features,
        labels,
        validation,
        validation_labels,
        readout_names=("logistic_regression",),
        regularization_values=(0.1, 1.0),
        seed=11,
    )

    assert len(result["trials"]) == 2
    assert result["readout"] == "logistic_regression"
    assert positive_probabilities(result["model"], validation).shape == (30,)
    assert set(result["validation_metrics"]) >= {
        "positive_f1",
        "auroc",
        "auprc",
        "mcc",
        "cohen_kappa",
    }


def test_binary_metric_inputs_reject_invalid_values():
    with pytest.raises(ValueError, match="both binary classes"):
        select_validation_threshold(np.array([0, 0]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        calculate_metrics(np.array([0, 1]), np.array([0.2, 1.2]), 0.5)
