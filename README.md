# Quantum Kernels for Toxicity Prediction on Unfamiliar Molecular Structures

CSE 491 research project comparing simulated quantum kernels with classical models for hERG blockade and drug-induced liver injury (DILI), with a focus on structurally unfamiliar molecules. The protocol is documented in [paper/protocol.md](paper/protocol.md). A result with no quantum gain is valid; no quantum advantage is claimed.

## Windows setup

Use Python 3.11 and Windows PowerShell from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

The current environment used Python 3.11.9 and PyTDC 0.4.1. The requirements file pins the direct dependencies used by the current download, curation, split, and test code. Experimental packages are not installed as part of setup; ask before adding PennyLane, Qiskit, Chemprop, or PyTorch.

## Data and reproducibility

The active TDC datasets are `herg`, `herg_karim`, and `dili`. Raw source rows use `Drug_ID`, `Drug` (SMILES), and `Y`; curated rows include `smiles`, `inchikey`, `parent_key`, `scaffold`, `label`, and `acyclic`. Labels are binary, where 1 is hERG blocker / DILI concern. A molecule ID in derived data is `<dataset>_<row_index:06d>`, using the row position in the corresponding processed CSV.

The canonical processed datasets and frozen scaffold splits are research inputs:

- Never edit, reorder, or regenerate `data/processed/<dataset>.csv`; split indices refer to its row positions.
- Never overwrite `data/splits/*_scaffold.json`. They are frozen as `splits-v1`. Any future split design must use new files and a new tag.
- Test data is for final evaluation only. Tune on training data with scaffold-grouped cross-validation; use validation data for calibration and threshold selection. Fit all learned preprocessing on training molecules only.
- Novelty bins are fixed at `[0.0, 0.4, 0.6, 1.0]`: familiar `[0.0, 0.4]`, intermediate `(0.4, 0.6]`, novel `(0.6, 1.0]`.

Raw and processed data are git-ignored; `data/raw/MANIFEST.json` records source checksums. Existing checked-out data can be checked without modifying it:

```powershell
python scripts/04_audit.py
```

The source workflow consists of these direct Python entry points:

```powershell
python scripts/01_download.py
python scripts/02_curate.py
python scripts/03_make_splits.py
```

These commands have not been run as part of this setup. By inspection, `01_download.py` refreshes raw files and the manifest, `02_curate.py` overwrites canonical processed files, and `03_make_splits.py` overwrites the frozen split JSON and cluster files. Do not run them in the canonical checkout. If the source workflow must be repeated, first use a disposable copy and review the script writes. A safe full rebuild of the canonical research inputs is not currently supported; non-overwriting output controls are required before one can be recommended.

## Pipeline and outputs

Run Python entry points directly from the repository root, for example:

```powershell
python -m pytest -q
python scripts/04_audit.py
```

The audit verifies raw and processed availability and schemas, binary labels, curation accounting and duplicate-parent uniqueness, chemical/scaffold consistency, manifest checksums, frozen split index and novelty alignment, train/validation/test leakage, and cluster assignment coverage. It is read-only and exits non-zero if checks fail.

When experiment scripts are added, configure their seeds and settings in `configs/`, implement reusable logic in `src/qktox/`, add tests in `tests/`, and save per-molecule predictions under `results/<dataset>/<split>/<model>.csv` with columns `mol_id,y_true,p_pred`. Prediction files are derived outputs and must not be used to change the protocol based on test results.

## Repository layout

- `configs/`: YAML configuration.
- `src/qktox/`: reusable curation, novelty, split, and audit utilities.
- `scripts/`: numbered pipeline entry points.
- `tests/`: pytest suite.
- `data/raw/`: source files and checksum manifest (ignored except the manifest).
- `data/processed/`: canonical curated tables (ignored; row order is frozen).
- `data/splits/`: committed frozen splits and cluster assignments.
- `paper/`: research protocol.
- `results/`: generated predictions and reports; do not commit large caches or model weights.
