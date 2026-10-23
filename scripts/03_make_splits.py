import sys, json, yaml, pandas as pd
sys.path.insert(0, "src")
from qktox.splits import repeated_scaffold_splits, butina_clusters
from qktox.novelty import novelty

cfg = yaml.safe_load(open("configs/data.yaml")); s = cfg["splits"]
for key in ["herg", "herg_karim", "dili"]:
    try:
        df = pd.read_csv(f"data/processed/{key}.csv")
    except FileNotFoundError:
        continue
    sp = repeated_scaffold_splits(df, s["n_repeats"], s["test_frac"], s["val_frac"], cfg["seed"])
    for name, idx in sp.items():
        tr, te = df.loc[idx["train"]], df.loc[idx["test"]]
        assert not set(tr.scaffold[~tr.acyclic]) & set(te.scaffold[~te.acyclic]), "scaffold leak!"
        assert not set(tr.parent_key) & set(te.parent_key), "duplicate leak!"
        nov = novelty(te.smiles, tr.smiles)
        idx["test_novelty"] = nov.tolist()
        print(f"{key} {name}: train {len(tr)} val {len(idx['val'])} test {len(te)} | mean novelty {nov.mean():.2f}")
    json.dump(sp, open(f"data/splits/{key}_scaffold.json", "w"))
    butina_clusters(df, s["cluster_cutoff"]).to_csv(f"data/splits/{key}_clusters.csv")
