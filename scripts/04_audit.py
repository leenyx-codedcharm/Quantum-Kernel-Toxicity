import json, sys, numpy as np, pandas as pd
import yaml
sys.path.insert(0, "src")
cfg = yaml.safe_load(open("configs/data.yaml"))
bins = cfg["novelty_bins"]

for key in ["herg", "herg_karim", "dili"]:
    log = pd.read_csv(f"data/processed/{key}_curation_log.csv")
    print(f"\n===== {key} =====")
    m = log[log.reason == "duplicate_parent"]
    c = log[log.reason == "label_conflict"]
    f = log[(log.action == "drop") & (log.reason != "label_conflict")]
    print("duplicates merged, molecules removed:", int((m.n - 1).sum()))
    print("label-conflict groups dropped:", len(c), "-> molecules removed:", int(c.n.sum()))
    print("parse/standardise failures:", len(f))

    df = pd.read_csv(f"data/processed/{key}.csv")
    sp = json.load(open(f"data/splits/{key}_scaffold.json"))
    for name, s in sp.items():
        nov = np.array(s["test_novelty"])
        counts = pd.cut(nov, bins, include_lowest=True).value_counts().reindex(pd.cut(nov, bins, include_lowest=True).categories, fill_value=0).tolist()
        pos = df.loc[s["test"]].label.mean()
        print(f"{name}: novelty-bin counts {counts} | test positive rate {pos:.2f}")
