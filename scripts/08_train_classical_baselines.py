"""Tune and evaluate the classical hERG_Karim baseline models."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.baselines import run_baselines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run a small holdout pilot sampled only from scaffold_r0 training rows.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML configuration path (defaults to configs/classical_baselines.yaml).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(__file__).resolve().parents[1]
    result = run_baselines(root, config_path=args.config, pilot=args.pilot)
    print(f"Results: {result['result_dir']}")
    print(result["aggregate"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
