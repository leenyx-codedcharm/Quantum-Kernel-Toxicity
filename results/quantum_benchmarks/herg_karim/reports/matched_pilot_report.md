# Matched hERG_Karim Scaffold R0 Pilot Report

All reported models use the same 64 training, 128 validation, and 128 test molecule IDs.
These are descriptive single-split pilot results, not five-repeat benchmark results.

| Model | AUROC | AUPRC | Positive F1 | Macro-F1 | MCC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Fidelity Kernel | 0.5704 | 0.4499 | 0.4082 | 0.5205 | 0.0411 | 0.2518 | 0.6967 |
| Tanimoto-SVM | 0.6590 | 0.5752 | 0.5225 | 0.5785 | 0.1693 | 0.2361 | 0.6717 |
| Trainable Fidelity Kernel | 0.5495 | 0.4347 | 0.1311 | 0.4297 | -0.0327 | 0.2367 | 0.6663 |
| VQC COBYLA | 0.4598 | 0.3769 | 0.3889 | 0.4715 | -0.0511 | 0.2545 | 0.7016 |
| VQC SPSA | 0.5110 | 0.4139 | 0.5315 | 0.4693 | 0.0734 | 0.2498 | 0.6928 |

Runtime breakdown is recorded in the accompanying CSV.
