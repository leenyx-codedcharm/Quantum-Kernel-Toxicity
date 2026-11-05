"""Generate molecular representations for the frozen hERG_Karim dataset."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.feature_generation import generate_features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = generate_features(Path(__file__).resolve().parents[1])
    print(
        "hERG_Karim feature generation complete: "
        f"source_rows={result['source_rows']}, feature_rows={result['feature_rows']}, "
        f"fingerprints={result['fingerprint_shape']}, descriptors={result['descriptor_shape']}, "
        f"invalid={result['invalid_raw_structures']}, "
        f"canonical_duplicate_groups={result['canonical_duplicate_groups']}"
    )
