"""Build extended metrics and scientific figures from existing hERG_Karim results."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.visualization import generate_scientific_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (defaults to the repository containing this script).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = generate_scientific_reports(args.root)
    print(f"Reports: {result['reports_dir']}")
    print(f"Visualizations: {result['visualization_dir']}")
    print(f"Figures: {len(result['figures'])}")
    print(f"Per-split metric rows: {len(result['per_split_metrics'])}")
    print(f"Baseline result files verified unchanged: {result['preserved_result_count']}")
    print(result["aggregate_metrics"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
