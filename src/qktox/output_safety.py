"""Helpers for protecting existing pipeline outputs."""

import sys
from pathlib import Path
from collections.abc import Iterable


def refuse_existing_outputs(output_files: Iterable[Path]) -> None:
    """Exit before writing if any declared output already exists."""
    for output_file in output_files:
        if output_file.exists():
            print(f"Refusing to overwrite existing output: {output_file}", file=sys.stderr)
            raise SystemExit(1)
