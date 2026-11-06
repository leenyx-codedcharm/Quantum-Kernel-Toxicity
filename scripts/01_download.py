"""Download TDC datasets; freeze with checksums + package versions."""
import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import yaml
from tdc.single_pred import Tox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.output_safety import refuse_existing_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write raw CSVs and the manifest under this directory instead of data/raw.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or root / "data" / "raw"
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    with (root / "configs" / "data.yaml").open(encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    datasets = [
        (key, dataset)
        for key, dataset in cfg["datasets"].items()
        if "tdc_name" in dataset
    ]
    output_files = [output_dir / f"{key}.csv" for key, _ in datasets]
    output_files.append(output_dir / "MANIFEST.json")
    refuse_existing_outputs(output_files)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "python": sys.version,
        "tdc_version": version("PyTDC"),
        "files": {},
    }

    for key, dataset in datasets:
        df = Tox(name=dataset["tdc_name"], path=str(output_dir)).get_data()
        output_file = output_dir / f"{key}.csv"
        df.to_csv(output_file, index=False)
        checksum = hashlib.sha256(output_file.read_bytes()).hexdigest()
        manifest["files"][key] = {
            "rows": len(df),
            "sha256": checksum,
            "label_counts": {
                str(label): int(count)
                for label, count in df["Y"].value_counts().items()
            },
        }
        print(key, len(df), checksum[:12])

    with (output_dir / "MANIFEST.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
