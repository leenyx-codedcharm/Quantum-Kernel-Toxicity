# Reproducibility Guide

## General rules

1. Use the frozen dataset and scaffold split files already present in the project.
2. Preserve molecule IDs and row-order checks.
3. Fit preprocessing on training data only.
4. Select models and thresholds on validation data only.
5. Evaluate test data only after recording a selection lock.
6. Do not overwrite completed experiment directories.
7. Preserve configuration files, random seeds, manifests, checksums, and prediction files.

## PennyLane environment

The exploratory PennyLane work used Python 3.11, PennyLane 0.45.1, and PennyLane Lightning 0.45.0 in a dedicated `.venv-pennylane` environment.

Activate/use the environment on Windows:

```powershell
.\.venv-pennylane\Scripts\python.exe -c "import pennylane as qml; print(qml.__version__)"
```

## Existing Hybrid VQC pilot

The selection phase uses training and validation data:

```powershell
.\.venv-pennylane\Scripts\python.exe .\scripts\pennylane_hybrid_vqc.py --phase select
```

The final pilot evaluation is a separate command:

```powershell
.\.venv-pennylane\Scripts\python.exe .\scripts\pennylane_hybrid_vqc.py --phase evaluate
```

Do not rerun selection in an existing output directory unless the script and experiment configuration explicitly support a new, separate run.

## Output locations

Experiment-specific outputs are stored below:

`results/quantum_benchmarks/herg_karim/pennylane_research/`

Each experiment should have a unique directory and retain its configuration, selection lock, metrics, predictions, and runtime metadata.

## Interpretation

Exact numerical agreement between Qiskit and PennyLane verifies implementation parity for the tested circuits. It does not show that the quantum model predicts better than a classical model.
