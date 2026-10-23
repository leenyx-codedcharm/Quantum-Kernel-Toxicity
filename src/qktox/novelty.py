"""novelty(x) = 1 - max Tanimoto similarity to the TRAINING set."""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg


def _gen(radius=2, nbits=2048, use_features=False):
    if use_features:
        return rfg.GetMorganGenerator(radius=radius, fpSize=nbits,
                                      atomInvariantsGenerator=rfg.GetMorganFeatureAtomInvGen())
    return rfg.GetMorganGenerator(radius=radius, fpSize=nbits)


def fps(smiles, **kw):
    g = _gen(**kw)
    return [g.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def novelty(test_smiles, train_smiles, **kw):
    tr, te = fps(train_smiles, **kw), fps(test_smiles, **kw)
    return np.array([1.0 - max(DataStructs.BulkTanimotoSimilarity(f, tr)) for f in te])
