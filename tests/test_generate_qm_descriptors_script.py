import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "16_generate_qm_descriptors.py"
SPEC = importlib.util.spec_from_file_location("generate_qm_descriptors_pilot", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load pilot script: {SCRIPT_PATH}")
PILOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PILOT
SPEC.loader.exec_module(PILOT)


def test_training_loader_reads_only_selected_train_smiles(
    tmp_path: Path, monkeypatch
):
    dataset_path = tmp_path / "curated.csv"
    pd.DataFrame(
        {
            "smiles": ["C", "N", "O", "F"],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(dataset_path, index=False)
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "scaffold_r0": {
                    "train": [0, 1],
                    "val": [2],
                    "test": [3],
                }
            }
        ),
        encoding="utf-8",
    )
    real_read_csv = pd.read_csv
    requested_columns = []

    def tracked_read_csv(*args, **kwargs):
        requested_columns.extend(kwargs["usecols"])
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(PILOT.pd, "read_csv", tracked_read_csv)
    molecule_ids, positions, smiles, checksums = PILOT._read_training_smiles(
        dataset_path,
        split_path,
        "scaffold_r0",
        sample_size=2,
        seed=3,
    )

    assert positions == [0, 1]
    assert molecule_ids == ["herg_karim_000000", "herg_karim_000001"]
    assert smiles == ["C", "N"]
    assert checksums == [
        hashlib.sha256(value.encode("utf-8")).hexdigest() for value in smiles
    ]
    assert requested_columns == ["smiles"]
