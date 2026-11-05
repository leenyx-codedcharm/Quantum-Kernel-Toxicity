from pathlib import Path
import pandas as pd
from rdkit import Chem

datasets = ["hERG", "hERG_Karim", "DILI"]

for name in datasets:

    path = Path("data/raw") / f"{name}.csv"

    if not path.exists():
        print(f"\nERROR: {path} not found!")
        continue

    df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("DATASET:", name)
    print("=" * 60)

    print("Shape:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nLabel distribution:")
    print(df["Y"].value_counts(dropna=False))

    print("\nLabel percentages:")
    print(df["Y"].value_counts(normalize=True) * 100)

    print("\nDuplicate Drug IDs:",
          df["Drug_ID"].duplicated().sum())

    print("Duplicate SMILES:",
          df["Drug"].duplicated().sum())

    canonical = []
    invalid = 0

    for smiles in df["Drug"]:

        if pd.isna(smiles):
            canonical.append(None)
            invalid += 1
            continue

        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            canonical.append(None)
            invalid += 1
        else:
            canonical.append(
                Chem.MolToSmiles(
                    mol,
                    canonical=True,
                    isomericSmiles=True
                )
            )

    df["Canonical_SMILES"] = canonical

    valid = df.dropna(subset=["Canonical_SMILES"])

    print("\nInvalid SMILES:", invalid)

    print("Canonical duplicate molecules:",
          valid["Canonical_SMILES"].duplicated().sum())

    conflicts = (
        valid.groupby("Canonical_SMILES")["Y"].nunique()
    )

    print("Conflicting label groups:",
          (conflicts > 1).sum())

    print("Unique valid molecules:",
          valid["Canonical_SMILES"].nunique())

print("\n" + "=" * 60)
print("DATASET AUDIT COMPLETED")
print("=" * 60)
