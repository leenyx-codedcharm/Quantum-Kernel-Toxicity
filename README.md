# Quantum Kernels for Toxicity Prediction on Unfamiliar Molecular Structures

CSE 491 research project comparing simulated quantum kernels with classical models for hERG blockade and drug-induced liver injury (DILI) on structurally unfamiliar molecules.

## Windows setup

Use Python 3.11 and PowerShell. From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the test suite and the read-only audit:

```powershell
python -m pytest -q
python scripts/04_audit.py
```

The audit requires the curated datasets, curation logs, and frozen scaffold splits to be present. The scripts `01_download.py`, `02_curate.py`, and `03_make_splits.py` overwrite data files; do not run them against this checkout. In particular, `03_make_splits.py` would replace the frozen `splits-v1` files.

## Repository layout

- `configs/`: settings; `src/qktox/`: library code.
- `scripts/`: numbered pipeline steps; `tests/`: pytest tests.
- `data/splits/`: committed frozen splits.
- `results/`: per-molecule predictions, with tables intended to be reproducible from saved predictions.
- `paper/protocol.md`: pre-registered evaluation protocol.

## Reproducibility rules

1. Never use the test set to select hyperparameters, thresholds, features, or calibration.
2. Do not edit `data/processed/<dataset>.csv` or regenerate `data/splits/*_scaffold.json`. Split files refer to row positions in the processed datasets.
3. Save per-molecule predictions under `results/<dataset>/<split>/<model>.csv`.
4. Do not commit raw or processed data, model weights, or large caches.

DILIrank 2.0 is not part of TDC; any future source data belongs in `data/raw/` and must follow the protocol's declared curation and label-mapping rules.
