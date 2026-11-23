from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reports/herg_karim/teacher_requirements"
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / "results/quantum_benchmarks/herg_karim/qm_qrc/pilot_scaffold_r0"
PRED = BASE / "test_predictions"
FEATURES = BASE / "features"

def need(p):
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return p

def tsne_plot(X, ids, y, stem, title):
    model = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                 init="pca", random_state=42, max_iter=1000)
    xy = model.fit_transform(X)
    pd.DataFrame({"mol_id": ids, "y_true": y,
                  "class": np.where(y == 1, "Positive", "Negative"),
                  "tsne_1": xy[:, 0], "tsne_2": xy[:, 1]}).to_csv(
                      OUT / f"{stem}_coordinates.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, name in [(0, "Negative"), (1, "Positive")]:
        m = y == label
        ax.scatter(xy[m, 0], xy[m, 1], label=name, alpha=.75, s=28)
    ax.set(title=title, xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=250)
    plt.close(fig)

def main():
    df = pd.read_csv(need(ROOT / "data/processed/herg_karim.csv"))
    counts = df["label"].value_counts().reindex([0, 1], fill_value=0)
    class_df = pd.DataFrame({
        "class": ["Negative", "Positive"], "label": [0, 1],
        "samples": [int(counts[0]), int(counts[1])]
    })
    class_df["percentage"] = class_df.samples / len(df) * 100
    class_df.to_csv(OUT / "class_distribution.csv", index=False)

    split_file = ROOT / "results/reports/herg_karim/class_balance_by_split.csv"
    if split_file.exists():
        pd.read_csv(split_file).to_csv(OUT / "class_distribution_by_split.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(class_df["class"], class_df["samples"])
    ax.set(title="hERG_Karim Class Distribution", ylabel="Number of molecules")
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    for b, n, pct in zip(bars, class_df.samples, class_df.percentage):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+35,
                f"{n:,} ({pct:.2f}%)", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figure1_class_distribution.png", dpi=250)
    plt.close(fig)

    c = pd.read_csv(need(PRED / "classical_2_layers_logistic_regression.csv"))
    q = pd.read_csv(need(PRED / "quantum_2_layers_linear_svm.csv"))
    required = {"mol_id", "y_true", "p_pred"}
    for name, frame in [("classical", c), ("quantum", q)]:
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} prediction columns invalid: {list(frame.columns)}")
    if not c.mol_id.equals(q.mol_id) or not c.y_true.equals(q.y_true):
        raise ValueError("Classical and quantum test IDs/labels do not match.")
    if c.mol_id.duplicated().any():
        raise ValueError("Duplicate test molecule IDs.")

    ids = c.mol_id.astype(str).to_numpy()
    y = c.y_true.astype(int).to_numpy()
    with np.load(need(FEATURES / "test_reservoir_features.npz")) as z:
        cf, qf = z["classical_2"].copy(), z["quantum_2"].copy()
    if len(y) != len(cf) or len(y) != len(qf):
        raise ValueError("Feature rows do not match test prediction rows.")
    if not np.isfinite(cf).all() or not np.isfinite(qf).all():
        raise ValueError("Feature matrix contains non-finite values.")

    tsne_plot(cf, ids, y, "figure2_classical_tsne",
              "Classical Reservoir Features (128 test molecules)")
    tsne_plot(qf, ids, y, "figure3_quantum_tsne",
              "Quantum Reservoir Features (128 test molecules)")

    metrics_file = need(BASE / "comparisons/pilot_metrics.csv")
    metric_table = pd.read_csv(metrics_file).set_index("model")
    specs = [
        ("Classical Reservoir", c, "classical_2_layers_logistic_regression",
         "figure4_classical_confusion_matrix.png", "classical_confusion_matrix.csv"),
        ("Quantum Reservoir", q, "quantum_2_layers_linear_svm",
         "figure5_quantum_confusion_matrix.png", "quantum_confusion_matrix.csv")
    ]
    rows = []
    for name, frame, key, fig_name, csv_name in specs:
        if key not in metric_table.index:
            raise ValueError(f"Missing validation threshold for {key}")
        threshold = float(metric_table.loc[key, "threshold"])
        probs = frame.p_pred.astype(float).to_numpy()
        pred = (probs >= threshold).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "model": name, "validation_threshold": threshold,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "specificity": tn/(tn+fp) if tn+fp else 0,
            "positive_f1": f1_score(y, pred, zero_division=0),
            "macro_f1": f1_score(y, pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "MCC": matthews_corrcoef(y, pred),
            "AUROC": roc_auc_score(y, probs),
            "AUPRC": average_precision_score(y, probs)
        })
        pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"],
                     columns=["Predicted Negative", "Predicted Positive"]).to_csv(OUT/csv_name)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(
            ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(f"{name} (n={len(y)})")
        fig.tight_layout()
        fig.savefig(OUT/fig_name, dpi=250)
        plt.close(fig)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "matched_confusion_metrics.csv", index=False)
    print("CLASS DISTRIBUTION")
    print(class_df.to_string(index=False))
    print("\nMATCHED TEST SET:", len(y), "| Negative:", int((y==0).sum()),
          "| Positive:", int((y==1).sum()))
    print("\nCONFUSION METRICS")
    print(result.round(4).to_string(index=False))
    print("\nOUTPUTS")
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            print(p.name, p.stat().st_size, "bytes")
    print("\nDone. Reservoir figures are exploratory single-split results.")
    print("Separate t-SNE maps have independent coordinate systems; do not compare absolute distances.")

if __name__ == "__main__":
    main()
