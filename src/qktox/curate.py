"""RDKit standardisation + logged curation. Every drop/merge decision is recorded."""
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()
_normalizer = rdMolStandardize.Normalizer()


def standardise(smiles):
    """Return (canonical parent smiles, inchikey, scaffold) or (None, reason, None)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "parse_fail", None
    try:
        mol = _normalizer.normalize(mol)
        mol = _chooser.choose(mol)          # strip salts, keep parent fragment
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "standardise_fail", None
    smi = Chem.MolToSmiles(mol)
    key = inchi.MolToInchiKey(mol)
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)   # '' for acyclic
    except Exception:
        scaf = ""
    return smi, key, scaf


def curate(df, smiles_col="Drug", label_col="Y", name_col="Drug_ID"):
    """df -> (clean_df, log_df). Conflicting labels dropped; duplicates merged."""
    log, rows = [], []
    for i, r in df.iterrows():
        smi, key, scaf = standardise(r[smiles_col])
        if smi is None:
            log.append({"row": i, "action": "drop", "reason": key})
            continue
        rows.append({"orig_id": r.get(name_col, i), "smiles": smi, "inchikey": key,
                     "parent_key": key.split("-")[0], "scaffold": scaf, "label": r[label_col]})
    d = pd.DataFrame(rows)
    out = []
    for pk, g in d.groupby("parent_key"):
        if g["label"].nunique() > 1:
            log.append({"row": pk, "action": "drop", "reason": "label_conflict", "n": len(g)})
            continue
        if len(g) > 1:
            log.append({"row": pk, "action": "merge", "reason": "duplicate_parent", "n": len(g)})
        out.append(g.iloc[0])
    clean = pd.DataFrame(out).reset_index(drop=True)
    clean["scaffold"] = clean["scaffold"].fillna("")
    clean["acyclic"] = clean["scaffold"] == ""     # explicit rule: singleton groups in splits.py
    return clean, pd.DataFrame(log, columns=["row", "action", "reason", "n"])
