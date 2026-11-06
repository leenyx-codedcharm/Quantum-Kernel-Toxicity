from pathlib import Path
import pandas as pd

path = Path("data/raw/DILIst Supplementary Table.xlsx")

df = pd.read_excel(path, sheet_name="DILIst")

# Clean column names
df.columns = df.columns.astype(str).str.strip()

print("=" * 60)
print("DILIST DATASET AUDIT")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nToxicity distribution:")
print(df["DILIst Classification"].value_counts())

print("\nToxicity percentages:")
print(
    df["DILIst Classification"]
    .value_counts(normalize=True) * 100
)

print("\nDuplicate IDs:")
print(df["DILIST_ID"].duplicated().sum())

print("\nDuplicate compound names:")
names = (
    df["CompoundName"]
    .astype(str)
    .str.strip()
    .str.casefold()
)

print(names.duplicated().sum())

print("\nRoute distribution:")
print(df["Routs of Administration"].value_counts())

# Preserve original records as CSV
output = Path("data/raw/dilist_original.csv")
df.to_csv(output, index=False)

print("\nSaved:", output)
print("\nDILIST AUDIT COMPLETED")
