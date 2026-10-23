import sys; sys.path.insert(0, "src")
import pandas as pd
from qktox.curate import curate

def test_salt_and_conflict():
    df = pd.DataFrame({"Drug": ["CCO", "CCO.[Na+].[Cl-]", "c1ccccc1", "c1ccccc1", "bad_smiles"],
                       "Y": [1, 1, 0, 1, 0], "Drug_ID": list("abcde")})
    clean, log = curate(df)
    assert len(clean) == 1   # ethanol+salt merged; benzene conflict dropped; bad smiles dropped
    assert "label_conflict" in set(log.reason)
