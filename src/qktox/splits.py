"""Frozen, group-aware splits (whole scaffold groups go to one side)."""
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from .novelty import fps


def scaffold_groups(df):
    """Acyclic molecules (empty scaffold) become singleton groups -- explicit rule."""
    g = df["scaffold"].astype(object).copy()
    for i in df.index[df["acyclic"]]:
        g[i] = f"ACYCLIC_{i}"
    return g


def _group_split(groups, frac, rng):
    """Randomly assign whole groups to the held-out side until ~frac of molecules."""
    sizes = groups.value_counts()
    order = list(sizes.index)
    rng.shuffle(order)
    held, n = set(), 0
    for g in order:
        if n >= frac * len(groups):
            break
        if sizes[g] > 0.5 * frac * len(groups) * 2:   # skip huge groups (e.g. benzene) -> stay in train
            continue
        held.add(g); n += sizes[g]
    return groups.isin(held).to_numpy()


def repeated_scaffold_splits(df, n_repeats=5, test_frac=0.2, val_frac=0.1, seed=42):
    groups = scaffold_groups(df)
    out = {}
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        is_test = _group_split(groups, test_frac, rng)
        tv = df.index[~is_test]
        is_val = _group_split(groups[tv], val_frac, rng)
        out[f"scaffold_r{r}"] = {"train": tv[~is_val].tolist(),
                                 "val": tv[is_val].tolist(),
                                 "test": df.index[is_test].tolist()}
    return out


def butina_clusters(df, cutoff=0.6):
    """Tanimoto-distance clusters; assign whole clusters to one side for a harder split."""
    f = fps(df["smiles"])
    d = []
    for i in range(1, len(f)):
        d.extend((1 - np.array(DataStructs.BulkTanimotoSimilarity(f[i], f[:i]))).tolist())
    cl = Butina.ClusterData(d, len(f), cutoff, isDistData=True)
    lab = np.zeros(len(f), int)
    for k, c in enumerate(cl):
        lab[list(c)] = k
    return pd.Series(lab, index=df.index, name="cluster")
