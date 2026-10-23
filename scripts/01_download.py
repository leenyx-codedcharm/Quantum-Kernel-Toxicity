"""Download TDC datasets; freeze with checksums + package versions."""
import hashlib, json, sys, yaml
from pathlib import Path
import tdc
from tdc.single_pred import Tox

cfg = yaml.safe_load(open("configs/data.yaml"))
raw = Path("data/raw"); raw.mkdir(parents=True, exist_ok=True)
manifest = {"tdc_version": getattr(tdc, "__version__", "unknown"), "python": sys.version, "files": {}}

for key, d in cfg["datasets"].items():
    if "tdc_name" not in d:
        continue
    df = Tox(name=d["tdc_name"], path=str(raw)).get_data()
    out = raw / f"{key}.csv"
    df.to_csv(out, index=False)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest["files"][key] = {"rows": len(df), "sha256": h,
                              "label_counts": {str(k): int(v) for k, v in df["Y"].value_counts().items()}}
    print(key, len(df), h[:12])

json.dump(manifest, open(raw / "MANIFEST.json", "w"), indent=2)
