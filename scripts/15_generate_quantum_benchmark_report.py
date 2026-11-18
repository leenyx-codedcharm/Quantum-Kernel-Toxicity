"""Generate a unified report from completed matched quantum pilot artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "quantum_benchmarks" / "herg_karim"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    fixed_path = ROOT / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/metrics/test_metrics.json"
    fixed = read_json(fixed_path)
    models = [
        ("Fixed Fidelity Kernel", fixed["fidelity_svc"]["test_metrics"],
         fixed["fidelity_svc"]["n_test"],
         fixed["fidelity_svc"]["training_seconds_final_fit"],
         fixed["fidelity_svc"]["kernel_construction_seconds_total"],
         fixed["fidelity_svc"]["test_prediction_seconds"]),
        ("Tanimoto-SVM", fixed["tanimoto_svm"]["test_metrics"],
         fixed["tanimoto_svm"]["n_test"],
         fixed["tanimoto_svm"]["training_seconds_final_fit"],
         fixed["tanimoto_svm"]["kernel_construction_seconds_total"],
         fixed["tanimoto_svm"]["test_prediction_seconds"]),
    ]
    for model_name, path in (
        ("Trainable Fidelity Kernel", RESULTS / "fidelity_trainable/metrics/test_metrics.json"),
        ("VQC COBYLA", RESULTS / "vqc_cobyla/metrics/test_metrics.json"),
        ("VQC SPSA", RESULTS / "vqc_spsa/metrics/test_metrics.json"),
    ):
        payload = read_json(path)
        models.append((
            model_name,
            payload["test_metrics"],
            payload["n_test"],
            payload.get("kernel_training_seconds", payload.get("training_seconds", 0.0)),
            payload.get("kernel_construction_seconds", 0.0),
            payload.get("test_prediction_seconds", 0.0),
        ))
    rows = []
    for model_name, metrics, n_test, optimization_seconds, kernel_seconds, prediction_seconds in models:
        rows.append({
            "model": model_name,
            "n_test": n_test,
            "accuracy": metrics["accuracy"] if "accuracy" in metrics else "",
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"] if "precision" in metrics else "",
            "recall": metrics["recall"] if "recall" in metrics else "",
            "positive_f1": metrics["positive_f1"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"] if "weighted_f1" in metrics else "",
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "mcc": metrics["mcc"],
            "brier_score": metrics["brier_score"],
            "log_loss": metrics["log_loss"],
            "optimization_or_training_seconds": optimization_seconds,
            "kernel_construction_seconds": kernel_seconds,
            "test_prediction_seconds": prediction_seconds,
        })

    comparison_dir = RESULTS / "comparisons"
    report_dir = RESULTS / "reports"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = comparison_dir / "matched_pilot_results.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = report_dir / "matched_pilot_report.md"
    lines = [
        "# Matched hERG_Karim Scaffold R0 Pilot Report",
        "",
        "All reported models use the same 64 training, 128 validation, and 128 test molecule IDs.",
        "These are descriptive single-split pilot results, not five-repeat benchmark results.",
        "",
        "| Model | AUROC | AUPRC | Positive F1 | Macro-F1 | MCC | Brier | Log loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['auroc']:.4f} | {row['auprc']:.4f} | "
            f"{row['positive_f1']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['mcc']:.4f} | {row['brier_score']:.4f} | {row['log_loss']:.4f} |"
        )
    lines.extend(["", "Runtime breakdown is recorded in the accompanying CSV."])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
