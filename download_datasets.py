
from tdc.single_pred import Tox
from pathlib import Path

# Create folders
output = Path("data/raw")
output.mkdir(parents=True, exist_ok=True)

# Datasets required for Project 10
datasets = ["hERG", "hERG_Karim", "DILI"]

for name in datasets:

    print(f"\nDownloading {name}...")

    data = Tox(name=name)

    # Download complete dataset
    df = data.get_data()

    # Save CSV
    df.to_csv(output / f"{name}.csv", index=False)

    print("Dataset:", name)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Saved successfully!")

    # Retrieve standard TDC split
    split = data.get_split()

    for split_name, split_df in split.items():

        split_df.to_csv(
            output / f"{name}_{split_name}.csv",
            index=False
        )

        print(split_name, split_df.shape)

print("\nAll datasets downloaded!")
