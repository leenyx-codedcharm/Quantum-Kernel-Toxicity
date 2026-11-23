# Quantum Kernel Toxicity — Research Progress

Last updated: 24 November 2026

## Project

This project investigates quantum machine learning for binary hERG
inhibition prediction using the hERG_Karim molecular dataset.

The research compares classical machine learning, quantum kernels,
quantum reservoir computing and hybrid quantum-classical classifiers.

## Dataset

- Curated molecules: 10,824
- Negative class: 5,482 (50.65%)
- Positive class: 5,342 (49.35%)
- ECFP4 fingerprints: 2,048 dimensions
- RDKit descriptors: 14
- Evaluation design: five frozen scaffold splits

The exploratory reservoir pilot uses scaffold_r0 with 64 training,
128 validation and 128 test molecules.

## Classical baselines

Previously reported five-scaffold results:

| Model | Positive F1 | AUROC | AUPRC |
|---|---:|---:|---:|
| Random Forest combined | 0.7833 | 0.8759 | 0.8766 |
| LightGBM combined | 0.7666 | 0.8616 | 0.8632 |
| Tanimoto-SVM | 0.7631 | 0.8605 | 0.8637 |

## Quantum reservoir results

The exploratory single-split reservoir comparison:

| Model | Positive F1 | AUROC |
|---|---:|---:|
| Quantum, 2 layers | 0.4483 | 0.5469 |
| Quantum, 4 layers | 0.3689 | 0.5043 |
| Quantum, 6 layers | 0.1333 | 0.5332 |
| Classical, 2 layers | 0.5512 | 0.6122 |

## PennyLane migration

PennyLane 0.45.1 and PennyLane Lightning 0.45.0 were installed in
a separate environment.

The existing four-qubit Qiskit reservoir was reproduced in PennyLane.

Numerical parity passed for:
- 2 encoding layers: 30 features per molecule
- 4 encoding layers: 60 features per molecule
- 6 encoding layers: 90 features per molecule

Full pilot parity was verified for 64 training, 128 validation and
128 test molecules. The maximum observed absolute difference was
approximately 1.55e-15.

These are feature-generation equivalence checks, not evidence of
quantum predictive advantage.

## PennyLane Hybrid VQC

An exploratory four-qubit, two-layer data-reuploading Hybrid VQC
was implemented using Adam and a jointly trained classical sigmoid
readout.

Validation-selected configuration:
- Seed: 20261120
- Best epoch: 30
- Validation AUROC: 0.5618
- Validation balanced accuracy: 0.5997

Final exploratory test results:

| Metric | Result |
|---|---:|
| Accuracy | 0.5547 |
| Positive F1 | 0.4771 |
| Macro-F1 | 0.5447 |
| AUROC | 0.5448 |
| AUPRC | 0.4183 |
| MCC | 0.0976 |

QM descriptors were not included in this VQC experiment.

## Molecular QM descriptors

The initial descriptor generation produced 9,736 successful records.
Targeted recovery added 42 records, leaving two unresolved molecules.

The consolidated candidate contains 9,778 of 9,780 planned records.
It must not be described as complete coverage.

## Teacher-requested analysis

- Class distribution: generated.
- Classical and quantum t-SNE maps: generated for the matched pilot.
- Classical and quantum confusion matrices: generated for the matched pilot.
- Systematic verified Q1 literature review: pending.
- PennyLane implementation: core migration and exploratory VQC completed.

## Current limitations

1. Quantum reservoir and Hybrid VQC results are exploratory single-split results.
2. They are not directly comparable to five-split classical averages.
3. No quantum advantage has been demonstrated.
4. The Hybrid VQC has not yet been evaluated across five scaffold splits.
5. QM-enhanced Hybrid VQC training remains future work.
6. Literature quartile verification and systematic review remain pending.

## Next research milestones

1. Investigate shallow and data-reuploading PennyLane VQC architectures.
2. Evaluate training-size effects using nested training samples.
3. Integrate QM descriptors with explicit missing-data handling.
4. Compare hybrid quantum and classical models under matched conditions.
5. Complete five-scaffold benchmarking.
6. Complete the systematic literature review.
7. Prepare the final research manuscript.
