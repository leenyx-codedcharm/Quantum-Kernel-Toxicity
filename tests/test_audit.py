import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.audit import _check_split, novelty_bin_indices


def test_novelty_bins_follow_protocol_boundaries():
    values = [0.0, 0.4, 0.40001, 0.6, 0.60001, 1.0]
    assert novelty_bin_indices(values, [0.0, 0.4, 0.6, 1.0]).tolist() == [0, 0, 1, 1, 2, 2]


def test_novelty_bins_accept_numpy_values():
    result = novelty_bin_indices(np.array([0.39, 0.5, 0.9]), [0.0, 0.4, 0.6, 1.0])
    assert result.tolist() == [0, 1, 2]


def test_split_audit_detects_duplicate_parent_leakage():
    processed = pd.DataFrame(
        {
            "parent_key": ["shared", "shared", "other"],
            "scaffold": ["A", "B", "C"],
            "acyclic": [False, False, False],
        }
    )
    split = {
        "train": [0],
        "val": [2],
        "test": [1],
        "test_novelty": [0.6],
    }
    issues = []
    _check_split("toy", "scaffold_r0", split, processed, [0.0, 0.4, 0.6, 1.0], issues)
    assert any("duplicate parent" in issue for issue in issues)
