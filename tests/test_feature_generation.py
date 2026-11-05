import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.curate import curate
from qktox.feature_generation import (
    DESCRIPTOR_FUNCTIONS,
    descriptor_frame,
    fingerprint_matrix,
    generate_features,
)


def test_fingerprints_are_2048_bits_and_encode_chirality():
    molecules = [
        Chem.MolFromSmiles("N[C@@H](C)C(=O)O"),
        Chem.MolFromSmiles("N[C@H](C)C(=O)O"),
    ]
    fingerprints = fingerprint_matrix(molecules)
    assert fingerprints.shape == (2, 2048)
    assert fingerprints.dtype == np.uint8
    assert not np.array_equal(fingerprints[0], fingerprints[1])


def test_descriptor_frame_has_expected_finite_columns():
    molecules = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
    descriptors = descriptor_frame(molecules, ["mol_a", "mol_b"])
    assert descriptors.columns.tolist() == [
        "mol_id",
        *(name for name, _ in DESCRIPTOR_FUNCTIONS),
    ]
    assert descriptors.shape == (2, 15)
    assert np.isfinite(descriptors.drop(columns="mol_id").to_numpy(dtype=float)).all()


def test_generate_features_maps_raw_ids_and_preserves_every_curation_decision(tmp_path):
    root = tmp_path
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    split_dir = root / "data" / "splits"
    config_dir = root / "configs"
    for directory in (raw_dir, processed_dir, split_dir, config_dir):
        directory.mkdir(parents=True)

    raw = pd.DataFrame(
        {
            "Drug_ID": ["ethanol_a", "ethanol_b", "ethylamine", "propane", "acid_a",
                        "acid_b", "benzene", "cyclohexane", "phenol", "invalid"],
            "Drug": ["CCO", "OCC", "CCN", "CCC", "CC(=O)O", "OC(=O)C",
                     "c1ccccc1", "C1CCCCC1", "Oc1ccccc1", "not_a_smiles"],
            "Y": [1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        }
    )
    curated, curation_log = curate(raw)
    curated.loc[curated["orig_id"] == "ethylamine", "inchikey"] = "STALE-KEY"
    curated.loc[curated["orig_id"] == "ethylamine", "parent_key"] = "STALE"
    raw.to_csv(raw_dir / "herg_karim.csv", index=False)
    curated.to_csv(processed_dir / "herg_karim.csv", index=False)
    curation_log.to_csv(processed_dir / "herg_karim_curation_log.csv", index=False)

    train = [0, 1]
    val = [2, 3]
    test = [4, 5]
    split = {
        f"scaffold_r{repeat}": {
            "train": train,
            "val": val,
            "test": test,
            "test_novelty": [0.2, 0.8],
        }
        for repeat in range(5)
    }
    (split_dir / "herg_karim_scaffold.json").write_text(
        json.dumps(split), encoding="utf-8"
    )
    config = {
        "dataset_key": "herg_karim",
        "source_file": "data/raw/herg_karim.csv",
        "processed_file": "data/processed/herg_karim.csv",
        "curation_log": "data/processed/herg_karim_curation_log.csv",
        "split_file": "data/splits/herg_karim_scaffold.json",
        "output_dir": "data/processed/features/hERG_Karim",
        "fingerprint": {"radius": 2, "n_bits": 2048, "use_chirality": True},
    }
    config_path = config_dir / "features.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = generate_features(root, config_path)
    output_dir = root / config["output_dir"]
    metadata = pd.read_csv(output_dir / "molecule_metadata.csv")
    chemprop = pd.read_csv(output_dir / "chemprop_input.csv")
    molformer = pd.read_csv(output_dir / "molformer_input.csv")
    descriptors = pd.read_csv(output_dir / "rdkit_descriptors.csv")
    with np.load(output_dir / "ecfp4_2048.npz", allow_pickle=False) as archive:
        fingerprints = archive["fingerprints"]
        mol_ids = archive["mol_ids"].tolist()

    assert result["source_rows"] == 10
    assert result["feature_rows"] == 6
    assert result["fingerprint_shape"] == (6, 2048)
    assert fingerprints.shape == (6, 2048)
    assert mol_ids == metadata["mol_id"].tolist()
    assert metadata["Drug_ID"].is_unique
    assert set(metadata["Drug_ID"]).isdisjoint({"acid_a", "acid_b", "invalid"})
    assert "ethanol_a" in set(metadata["Drug_ID"])
    assert metadata["label"].tolist() == chemprop["label"].tolist()
    assert metadata["label"].tolist() == molformer["label"].tolist()
    assert metadata["mol_id"].tolist() == descriptors["mol_id"].tolist()
    assert json.loads(metadata.iloc[0]["split_assignment"]) == {
        f"scaffold_r{repeat}": "train" for repeat in range(5)
    }
    manifest = json.loads((output_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    assert manifest["invalid_raw_structures"] == 1
    assert manifest["curation"]["same_label_parent_groups_merged"] == 1
    assert manifest["curation"]["conflicting_parent_groups_excluded"] == 1
    assert manifest["identity_discrepancies"] == [
        {
            "Drug_ID": "ethylamine",
            "differences": [
                "stored_vs_recomputed_inchikey",
                "stored_vs_recomputed_parent_key",
            ],
        }
    ]
    assert metadata.loc[metadata["Drug_ID"] == "ethylamine", "inchikey"].iloc[0] != "STALE-KEY"
    assert metadata.loc[metadata["Drug_ID"] == "ethylamine", "source_inchikey"].iloc[0] == "STALE-KEY"
    assert (output_dir / "feature_generation_audit.txt").is_file()
