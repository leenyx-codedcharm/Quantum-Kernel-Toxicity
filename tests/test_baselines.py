import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.baselines import (
    ExperimentData,
    _fit_evaluate_model,
    _kernel_memory_estimate_mb,
    calculate_metrics,
    select_threshold,
    tanimoto_kernel_matrix,
)


def test_tanimoto_kernel_is_exact_symmetric_and_blocked():
    fingerprints = np.array(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    kernel = tanimoto_kernel_matrix(fingerprints, fingerprints, block_size=2)
    expected = np.array(
        [
            [1.0, 1 / 3, 0.0, 1.0],
            [1 / 3, 1.0, 0.0, 1 / 3],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1 / 3, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(kernel, expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(kernel, kernel.T, rtol=0, atol=0)


def test_tanimoto_kernel_rejects_nonbinary_input():
    with np.testing.assert_raises(ValueError):
        tanimoto_kernel_matrix(np.array([[0, 2]], dtype=np.uint8), np.array([[0, 1]], dtype=np.uint8))


def test_validation_threshold_and_metrics():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, validation_score = select_threshold(labels, probabilities)
    assert validation_score == 1.0
    metrics = calculate_metrics(labels, probabilities, threshold)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["brier_score"] < 0.1
    assert metrics["log_loss"] < 0.5
    _, all_negative_score = select_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    assert all_negative_score == 0.5


def test_kernel_memory_estimate_includes_dense_matrices_and_solver_cache():
    small = _kernel_memory_estimate_mb(100, 20, 40, 32)
    larger = _kernel_memory_estimate_mb(200, 20, 40, 32)
    assert small > 32
    assert larger > small


def test_tanimoto_svm_fits_train_only_and_writes_test_schema(tmp_path):
    rng = np.random.default_rng(42)
    fingerprints = rng.integers(0, 2, size=(48, 16), dtype=np.uint8)
    labels = np.tile([0, 1], 24).astype(np.int8)
    metadata = pd.DataFrame(
        {
            "mol_id": [f"mol_{index:03d}" for index in range(48)],
            "label": labels,
        }
    )
    config = {
        "seed": 42,
        "n_jobs": 1,
        "output_root": str(tmp_path / "checkpoints"),
        "prediction_root": str(tmp_path / "predictions"),
        "tuning": {
            "tanimoto_svm": {
                "cache_size_mb": 16,
                "kernel_block_size": 4,
                "max_kernel_memory_mb": 128,
                "calibration_folds": 5,
                "calibration_logistic_c": 1.0,
            }
        },
    }
    data = ExperimentData(
        metadata=metadata,
        labels=labels,
        fingerprints=fingerprints,
        descriptors=np.zeros((48, 1), dtype=np.float32),
        descriptor_names=("dummy",),
        splits={},
        feature_manifest={},
        config=config,
        input_checksums={},
    )
    partitions = {
        "train": list(range(24)),
        "val": list(range(24, 36)),
        "test": list(range(36, 48)),
    }
    result, trials, predictions = _fit_evaluate_model(
        data,
        "scaffold_r0",
        partitions,
        "tanimoto_svm",
        "fingerprints",
        [{"C": 1.0, "class_weight": None}],
        save_model=True,
    )
    assert len(trials) == 1 and trials[0]["selected"]
    assert result["n_train"] == 24 and result["n_validation"] == 12 and result["n_test"] == 12
    assert predictions.columns.tolist() == ["mol_id", "y_true", "p_pred"]
    assert predictions["mol_id"].tolist() == metadata.iloc[36:]["mol_id"].tolist()
    assert result["checkpoint_path"]
    assert Path(result["checkpoint_path"]).is_file()
    assert (tmp_path / "predictions" / "scaffold_r0" / "tanimoto_svm_fingerprints.csv").is_file()
