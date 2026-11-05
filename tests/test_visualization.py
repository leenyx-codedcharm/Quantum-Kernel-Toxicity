import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.visualization import (
    METRICS,
    aggregate_split_metrics,
    calculate_classification_metrics,
    _novelty_bin,
    _validate_selection_trials,
)


def test_extended_metrics_use_validation_selected_threshold():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])
    metrics = calculate_classification_metrics(labels, probabilities, threshold=0.5)

    assert set(metrics) == set(METRICS)
    assert metrics["f1_positive"] == pytest.approx(0.5)
    assert metrics["f1_macro"] == pytest.approx(0.5)
    assert metrics["f1_weighted"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["auprc"] == pytest.approx(5 / 6)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["mcc"] == pytest.approx(0.0)
    assert metrics["brier_score"] == pytest.approx(0.255)
    assert metrics["log_loss"] > 0


@pytest.mark.parametrize(
    ("labels", "probabilities", "threshold"),
    [
        ([0, 1], [0.2], 0.5),
        ([0, 1], [0.2, float("nan")], 0.5),
        ([0, 1], [0.2, 1.1], 0.5),
        ([0, 0], [0.2, 0.3], 0.5),
        ([0, 1], [0.2, 0.8], 1.1),
    ],
)
def test_extended_metrics_reject_invalid_inputs(labels, probabilities, threshold):
    with pytest.raises(ValueError):
        calculate_classification_metrics(np.asarray(labels), np.asarray(probabilities), threshold)


def test_aggregate_metrics_reports_t_interval_and_single_run_nan():
    base = {
        "experiment": "full",
        "model": "lightgbm_descriptors",
        "model_family": "lightgbm",
        "representation": "descriptors",
        "n_test": 100,
    }
    rows = []
    for value in (0.7, 0.8, 0.9):
        rows.append({**base, **{metric: value for metric in METRICS}})
    full = aggregate_split_metrics(pd.DataFrame(rows))
    assert len(full) == 1
    assert full.loc[0, "n_splits"] == 3
    assert full.loc[0, "f1_positive_mean"] == pytest.approx(0.8)
    assert full.loc[0, "f1_positive_std"] == pytest.approx(0.1)
    assert full.loc[0, "f1_positive_ci95_low"] < 0.8
    assert full.loc[0, "f1_positive_ci95_high"] > 0.8

    pilot = aggregate_split_metrics(pd.DataFrame([{**base, "experiment": "pilot", **{metric: 0.8 for metric in METRICS}}]))
    assert np.isnan(pilot.loc[0, "f1_positive_std"])
    assert np.isnan(pilot.loc[0, "f1_positive_ci95_low"])
    assert np.isnan(pilot.loc[0, "f1_positive_ci95_high"])


def test_selection_audit_requires_best_validation_trial_and_matching_parameters():
    results = pd.DataFrame(
        [
            {
                "split": "scaffold_r0",
                "model": "lightgbm_descriptors",
                "validation_selection_auroc": 0.8,
                "selected_parameters": '{"depth": 2}',
                "validation_threshold": 0.5,
            }
        ]
    )
    trials = pd.DataFrame(
        [
            {"split": "scaffold_r0", "model": "lightgbm_descriptors", "selected": False, "validation_auroc": 0.7, "parameters": '{"depth": 1}'},
            {"split": "scaffold_r0", "model": "lightgbm_descriptors", "selected": True, "validation_auroc": 0.8, "parameters": '{"depth": 2}'},
        ]
    )
    _validate_selection_trials("full", results, trials)

    trials.loc[1, "validation_auroc"] = 0.6
    with pytest.raises(ValueError, match="not best by validation AUROC"):
        _validate_selection_trials("full", results, trials)


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.0, "familiar"), (0.4, "familiar"), (0.4001, "intermediate"), (0.6, "intermediate"), (0.6001, "novel"), (1.0, "novel")],
)
def test_novelty_bin_matches_frozen_boundaries(score, expected):
    assert _novelty_bin(score) == expected


def test_pilot_novelty_is_not_invented():
    assert _novelty_bin(np.nan) is None
