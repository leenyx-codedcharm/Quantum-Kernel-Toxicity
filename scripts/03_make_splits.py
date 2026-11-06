"""Generate repeated scaffold splits and Butina clusters."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qktox.novelty import novelty
from qktox.output_safety import refuse_existing_outputs
from qktox.splits import butina_clusters, repeated_scaffold_splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write split JSON and cluster CSVs under this directory instead of data/splits.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or ROOT / "data" / "splits"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    with (ROOT / "configs" / "data.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    dataset_keys = [
        key
        for key in ("herg", "herg_karim", "dili")
        if (ROOT / "data" / "processed" / f"{key}.csv").is_file()
    ]
    output_files = [
        output_dir / filename
        for key in dataset_keys
        for filename in (f"{key}_scaffold.json", f"{key}_clusters.csv")
    ]
    refuse_existing_outputs(output_files)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_config = config["splits"]
    for key in ("herg", "herg_karim", "dili"):
        processed_file = ROOT / "data" / "processed" / f"{key}.csv"
        if not processed_file.is_file():
            continue
        data = pd.read_csv(processed_file)
        splits = repeated_scaffold_splits(
            data,
            split_config["n_repeats"],
            split_config["test_frac"],
            split_config["val_frac"],
            config["seed"],
        )
        for name, indices in splits.items():
            train, test = data.loc[indices["train"]], data.loc[indices["test"]]
            assert not set(train.scaffold[~train.acyclic]) & set(
                test.scaffold[~test.acyclic]
            ), "scaffold leak!"
            assert not set(train.parent_key) & set(test.parent_key), "duplicate leak!"
            test_novelty = novelty(test.smiles, train.smiles)
            indices["test_novelty"] = test_novelty.tolist()
            print(
                f"{key} {name}: train {len(train)} val {len(indices['val'])} "
                f"test {len(test)} | mean novelty {test_novelty.mean():.2f}"
            )
        with (output_dir / f"{key}_scaffold.json").open(
            "w", encoding="utf-8"
        ) as split_file:
            json.dump(splits, split_file)
        butina_clusters(data, split_config["cluster_cutoff"]).to_csv(
            output_dir / f"{key}_clusters.csv"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
