import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "18_generate_qm_descriptor_coverage.py"
SPEC = importlib.util.spec_from_file_location("generate_qm_descriptor_coverage", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load coverage script: {SCRIPT_PATH}")
COVERAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COVERAGE
SPEC.loader.exec_module(COVERAGE)


def _splits():
    return {
        f"scaffold_r{repeat}": {
            "train": list(range(20)),
            "val": list(range(20, 25)),
            "test": list(range(25, 30)),
        }
        for repeat in range(5)
    }


def test_coverage_plan_is_deterministic_nested_and_train_only():
    split_names = [f"scaffold_r{index}" for index in range(5)]
    plan = COVERAGE.build_coverage_plan(
        _splits(), split_names, training_sizes=[2, 4, 8], seed=51
    )
    repeated = COVERAGE.build_coverage_plan(
        _splits(), split_names, training_sizes=[2, 4, 8], seed=51
    )

    assert plan == repeated
    expected_rows = set()
    for split_name in split_names:
        split_plan = plan["splits"][split_name]
        training = split_plan["train_samples"]
        subsets = [set(training[str(size)]) for size in (2, 4, 8)]
        assert subsets[0] < subsets[1] < subsets[2]
        assert not subsets[-1] & set(range(20, 30))
        expected_rows.update(subsets[-1])
        expected_rows.update(range(20, 30))
        assert split_plan["val"] == [
            f"herg_karim_{index:06d}" for index in range(20, 25)
        ]
        assert split_plan["test"] == [
            f"herg_karim_{index:06d}" for index in range(25, 30)
        ]
    assert set(plan["unique_row_indices"]) == expected_rows


def test_coverage_plan_rejects_missing_repeat_or_oversized_training_subset():
    with pytest.raises(ValueError, match="each frozen scaffold"):
        COVERAGE.build_coverage_plan(
            _splits(), ["scaffold_r0"], training_sizes=[2], seed=1
        )
    with pytest.raises(ValueError, match="smaller than requested"):
        COVERAGE.build_coverage_plan(
            _splits(),
            [f"scaffold_r{index}" for index in range(5)],
            training_sizes=[64],
            seed=1,
        )


def test_coverage_loader_reads_only_requested_smiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset_path = tmp_path / "curated.csv"
    pd.DataFrame(
        {
            "smiles": ["C", "N", "O", "F"],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(dataset_path, index=False)
    requested_columns = []
    real_read_csv = pd.read_csv

    def tracked_read_csv(*args, **kwargs):
        requested_columns.extend(kwargs["usecols"])
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(COVERAGE.pd, "read_csv", tracked_read_csv)
    smiles = COVERAGE._read_selected_smiles(dataset_path, [0, 2])

    assert smiles == ["C", "O"]
    assert requested_columns == ["smiles"]


def test_coverage_loader_requires_sorted_unique_positions(tmp_path: Path):
    dataset_path = tmp_path / "curated.csv"
    dataset_path.write_text("smiles\nC\nN\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique, and sorted"):
        COVERAGE._read_selected_smiles(dataset_path, [1, 0])
