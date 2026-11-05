"""Run read-only checks on source datasets, curation, frozen splits, and clusters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.audit import run_audit


if __name__ == "__main__":
    failures = run_audit(Path(__file__).resolve().parents[1])
    raise SystemExit(1 if failures else 0)
