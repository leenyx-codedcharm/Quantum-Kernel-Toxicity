"""Curate available raw datasets into processed CSVs."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qktox.curate import curate
from qktox.output_safety import refuse_existing_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write curated datasets under this directory instead of data/processed.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or ROOT / "data" / "processed"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    with (ROOT / "configs" / "data.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    dataset_keys = [
        key
        for key in config["datasets"]
        if (ROOT / "data" / "raw" / f"{key}.csv").is_file()
    ]
    output_files = [
        output_dir / filename
        for key in dataset_keys
        for filename in (f"{key}.csv", f"{key}_curation_log.csv")
    ]
    refuse_existing_outputs(output_files)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in config["datasets"]:
        source_file = ROOT / "data" / "raw" / f"{key}.csv"
        if not source_file.is_file():
            print("skip (missing):", key)
            continue
        raw_data = pd.read_csv(source_file)
        clean, log = curate(raw_data)
        clean.to_csv(output_dir / f"{key}.csv", index=False)
        log.to_csv(output_dir / f"{key}_curation_log.csv", index=False)
        print(
            f"{key}: {len(raw_data)} -> {len(clean)} | log rows {len(log)} | "
            f"pos rate {clean.label.mean():.2f} | acyclic {int(clean.acyclic.sum())}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
