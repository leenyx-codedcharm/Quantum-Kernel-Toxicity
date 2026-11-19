import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.qm_qrc import (
    ClassicalRandomReservoir,
    MolecularFeatureFusion,
    QM_FEATURE_COLUMNS,
    calculate_metrics,
    positive_probabilities,
    select_validation_threshold,
    tune_readout,
)


def _fusion_inputs(row_count: int, seed: int = 5):
    rng = np.random.default_rng(seed)
    molecule_ids = [f"herg_karim_{index:06d}" for index in range(row_count)]
    checksums = [f"{index + 1:064x}" for index in range(row_count)]
    return {
        "molecule_ids": molecule_ids,
        "structure_checksums": checksums,
        "rdkit_descriptors": rng.normal(size=(row_count, 14)),
        "qm_molecule_ids": molecule_ids.copy(),
        "qm_structure_checksums": checksums.copy(),
        "qm_descriptors": rng.normal(size=(row_count, len(QM_FEATURE_COLUMNS))),
    }


@pytest.mark.parametrize("output_dimensions", [4, 8])
def test_qm_feature_fusion_fits_only_training_rows(output_dimensions):
    training = _fusion_inputs(20)
    fusion = MolecularFeatureFusion("rdkit_qm", output_dimensions)
    train_features = fusion.fit(partition="train", **training)
    parameters_before_validation = {
        name: values.copy()
        for name, values in fusion.preprocessing_parameters.items()
    }

    validation = _fusion_inputs(3, seed=19)
    validation["rdkit_descriptors"] *= 1e6
    validation["qm_descriptors"] *= 1e6
    validation_features = fusion.transform(partition="val", **validation)

    assert train_features.shape == (20, output_dimensions)
    assert validation_features.shape == (3, output_dimensions)
    for name, values in fusion.preprocessing_parameters.items():
        np.testing.assert_array_equal(values, parameters_before_validation[name])
    assert fusion.configuration()["preprocessing_fit_partition"] == "train"
    assert fusion.configuration()["absolute_total_energy_used"] is False


def test_ecfp_rdkit_qm_fusion_checks_molecule_identity_and_structure():
    training = _fusion_inputs(20)
    training["ecfp4"] = np.zeros((20, 2048), dtype=np.uint8)
    training["ecfp4"][:, 0] = 1
    fusion = MolecularFeatureFusion("ecfp4_rdkit_qm", 4)
    fusion.fit(partition="train", **training)
    assert fusion.configuration()["input_dimensions"] == 2048 + 14 + len(
        QM_FEATURE_COLUMNS
    )

    mismatched = _fusion_inputs(2, seed=23)
    mismatched["ecfp4"] = np.zeros((2, 2048), dtype=np.uint8)
    mismatched["qm_molecule_ids"][0] = "herg_karim_999999"
    with pytest.raises(ValueError, match="molecule IDs"):
        fusion.transform(partition="val", **mismatched)

    mismatched = _fusion_inputs(2, seed=29)
    mismatched["ecfp4"] = np.zeros((2, 2048), dtype=np.uint8)
    mismatched["qm_structure_checksums"][0] = "f" * 64
    with pytest.raises(ValueError, match="checksums"):
        fusion.transform(partition="val", **mismatched)


def test_feature_fusion_rejects_nontraining_fit_and_unspecified_representation():
    with pytest.raises(ValueError, match="unsupported molecular representation"):
        MolecularFeatureFusion("selected_descriptors", 4)
    with pytest.raises(ValueError, match="only be fitted on the train"):
        MolecularFeatureFusion("rdkit_qm", 4).fit(
            partition="val",
            **_fusion_inputs(20),
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
