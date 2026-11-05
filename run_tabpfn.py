"""Evaluate TabPFN on the safe and history feature sets.

In Google Colab, upload metadata_feature_matrix.csv, enable a GPU runtime if
available, install dependencies with `pip install tabpfn scikit-learn pandas numpy`,
then run `python run_tabpfn.py`.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tabpfn import TabPFNClassifier


RANDOM_SEED = 42
INPUT_FILE = "metadata_feature_matrix.csv"
OUTPUT_FILE = "tabpfn_results.csv"
TIER1_FEATURES = [
    "age_years",
    "female",
    "cadaver_length",
    "cadaver_weight",
    "bmi_like",
]


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    required_columns = {"id", "label", *TIER1_FEATURES}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"{INPUT_FILE} is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    if df["id"].duplicated().any():
        raise ValueError(f"{INPUT_FILE} must contain one row per unique id.")
    if df["label"].isna().any():
        raise ValueError(f"{INPUT_FILE} contains missing labels.")

    y = df["label"].to_numpy()
    tier12_features = [column for column in df.columns if column not in ("id", "label")]
    feature_sets = [
        ("Tier 1 safe", TIER1_FEATURES),
        ("Tier 1+2 (history)", tier12_features),
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=2, random_state=RANDOM_SEED
    )
    results = []

    for name, columns in feature_sets:
        X = df[columns].astype(float).to_numpy()
        scores = []
        for train_indices, test_indices in cv.split(X, y):
            # A fresh model is fitted on each training fold only.
            classifier = TabPFNClassifier(device=device, random_state=RANDOM_SEED)
            classifier.fit(X[train_indices], y[train_indices])
            predictions = classifier.predict(X[test_indices])
            scores.append(
                f1_score(y[test_indices], predictions, average="macro")
            )

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        print(f"{name}: macro-F1 = {mean_score:.3f} +/- {std_score:.3f}")
        results.append(
            {
                "feature_set": name,
                "mean_macro_f1": mean_score,
                "std_macro_f1": std_score,
            }
        )

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
