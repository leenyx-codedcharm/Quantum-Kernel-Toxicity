import sys, yaml, pandas as pd
sys.path.insert(0, "src")
from qktox.curate import curate

cfg = yaml.safe_load(open("configs/data.yaml"))
for key in cfg["datasets"]:
    try:
        df = pd.read_csv(f"data/raw/{key}.csv")
    except FileNotFoundError:
        print("skip (missing):", key); continue
    clean, log = curate(df)
    clean.to_csv(f"data/processed/{key}.csv", index=False)
    log.to_csv(f"data/processed/{key}_curation_log.csv", index=False)
    print(f"{key}: {len(df)} -> {len(clean)} | log rows {len(log)} | "
          f"pos rate {clean.label.mean():.2f} | acyclic {int(clean.acyclic.sum())}")
