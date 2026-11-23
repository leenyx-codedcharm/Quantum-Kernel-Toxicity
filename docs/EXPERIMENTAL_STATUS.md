# Experimental Status

Last updated: 24 November 2026

## Completed

- Curated dataset class counts and split-level class-balance analysis.
- Exploratory classical and quantum reservoir t-SNE visualisations.
- Matched pilot confusion matrices using validation-selected thresholds.
- Qiskit/PennyLane reservoir feature parity for 2, 4, and 6 layers.
- Full 320-molecule structural pilot feature generation in PennyLane.
- Exploratory two-layer PennyLane data-reuploading Hybrid VQC training and held-out pilot evaluation.
- QM descriptor recovery workflow: 42 of 44 original failures recovered.

## Exploratory metrics

Hybrid VQC pilot test AUROC: 0.5448; positive F1: 0.4771; MCC: 0.0976.

Quantum reservoir pilot AUROC: 0.5469 (2 layers).

Classical reservoir pilot AUROC: 0.6122 (2 layers).

These are single-split pilot metrics and are not directly comparable with five-scaffold classical baseline averages.

## Pending

- Systematic literature review with verified DOI and quartile information.
- Five-scaffold Hybrid VQC evaluation.
- Nested training-size experiments.
- Controlled comparison of re-uploading depths and entanglement structures.
- QM-enhanced Hybrid VQC with explicit missing-descriptor handling.
- Statistical uncertainty and paired comparisons.
- Final manuscript and full reproducibility audit.

## Data coverage limitation

The QM descriptor candidate contains 9,778 of 9,780 planned records. Molecules `herg_karim_003270` and `herg_karim_007362` remain unresolved. Do not report full QM descriptor coverage.
