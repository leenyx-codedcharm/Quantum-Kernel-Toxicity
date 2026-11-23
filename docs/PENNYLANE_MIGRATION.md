# PennyLane Migration and Hybrid VQC

## Purpose

PennyLane is the primary framework for new quantum model development.
The existing Qiskit experiments are retained as historical references.

## Environment

- Python: 3.11
- PennyLane: 0.45.1
- PennyLane Lightning: 0.45.0

Create the isolated environment on Windows:

```powershell
py -3.11 -m venv .venv-pennylane
.\.venv-pennylane\Scripts\python.exe -m pip install -r requirements-pennylane.txt
```

The environment is intentionally separate from the existing RDKit
and Qiskit environments.

## QRC parity

The four-qubit reservoir was reproduced at 2, 4 and 6 encoding layers.

The full pilot contains:
- 64 training molecules
- 128 validation molecules
- 128 test molecules

The maximum observed absolute numerical difference was approximately
1.55e-15.

This verifies numerical feature parity for the tested simulator
configurations. It does not demonstrate better model accuracy.

## Hybrid VQC

The exploratory Hybrid VQC uses:
- Four qubits
- Two data-reuploading layers
- Trainable quantum rotations
- Linear CNOT entanglement
- Four local Pauli-Z measurements
- Three adjacent ZZ measurements
- A jointly trained sigmoid readout
- Adam optimisation
- Validation-based early stopping

The test set is evaluated only after a validation selection lock.

## Run

```powershell
.\.venv-pennylane\Scripts\python.exe `
    .\scripts\pennylane_hybrid_vqc.py --phase select

.\.venv-pennylane\Scripts\python.exe `
    .\scripts\pennylane_hybrid_vqc.py --phase evaluate
```

The experiment output is stored under:

`results/quantum_benchmarks/herg_karim/pennylane_research/hybrid_vqc_adam/`

## Reproducibility

The selection lock, model checkpoint, training histories,
test predictions and test metrics are retained.

Do not overwrite completed experiment directories.

## Scientific limitations

The current VQC result is a single-split exploratory result.
It is not a five-split benchmark and does not establish quantum advantage.

Future experiments must use frozen scaffold splits, training-only
preprocessing, validation-based selection and untouched final test data.
