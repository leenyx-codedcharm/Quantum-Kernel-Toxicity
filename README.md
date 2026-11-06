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

The source scripts refuse to start if any output file already exists. Supply `--output-dir <new directory>` to redirect that script's outputs; inputs remain at their canonical paths, so the three commands are not a redirected end-to-end rebuild. The frozen split files and canonical processed data are never overwritten by default.

## Pipeline and outputs

Run Python entry points directly from the repository root, for example:

```powershell
python -m pytest -q
python scripts/04_audit.py
```

Run both the tests and read-only audit in sequence with the fail-fast PowerShell helper:

```powershell
.\scripts\run_pipeline_check.ps1
```

The audit verifies raw and processed availability and schemas, binary labels, curation accounting and duplicate-parent uniqueness, chemical/scaffold consistency, manifest checksums, frozen split index and novelty alignment, train/validation/test leakage, and cluster assignment coverage. It is read-only and exits non-zero if checks fail.

### Molecular feature generation

Generate the hERG_Karim representations from the existing raw and curated tables and the five frozen scaffold repeats:

```powershell
.\.venv\Scripts\python.exe scripts\07_generate_features.py
```

The pipeline reads `data\raw\herg_karim.csv`, `data\processed\herg_karim.csv`, its curation log, and `data\splits\herg_karim_scaffold.json` without modifying them. It maps curated representatives to raw rows by `Drug_ID`, checks labels and molecular parent identity, confirms each split exactly partitions the curated rows without canonical-structure overlap, and records raw invalid structures, canonical duplicates, curation merges/conflicts, legacy identity discrepancies, input checksums, and split counts. Fingerprints and descriptors use the frozen curated SMILES; metadata also preserves the original SMILES, its directly canonicalized form/InChIKey, and the standardized raw form. Outputs are written to `data\processed\features\hERG_Karim\`. Split assignments in the metadata and model input files are JSON mappings with one `scaffold_rN` assignment per repeat. This stage does not fit preprocessing transforms, train models, or download pretrained weights.

### Classical baselines

Run a bounded development pilot, then the five-repeat baseline experiment:

```powershell
.\.venv\Scripts\python.exe scripts\08_train_classical_baselines.py --pilot
.\.venv\Scripts\python.exe scripts\08_train_classical_baselines.py
```

The experiment reuses the saved hERG_Karim fingerprints, descriptors, and frozen scaffold membership. LightGBM and Random Forest compare descriptor-only, fingerprint-only, and concatenated inputs; Tanimoto-SVM uses the Morgan fingerprint Tanimoto kernel. Each family receives 12 validation candidates per repeat. Hyperparameters are selected by validation AUROC; the binary decision threshold is selected by validation balanced accuracy. SVM probabilities use training-only internal calibration. Its dense float64 kernel requires about `8 * n_train * (n_train + max(n_validation, n_test))` bytes for stored matrices, in addition to the configured solver cache; blockwise construction bounds temporary products. A configurable memory guard rejects oversized splits. Official test predictions and metrics are generated only after selection and are saved per molecule under `results\herg_karim\<split>\`. Checkpoints, per-split metrics, tuning records, aggregate metrics, figures, and a manifest are under `results\classical_baselines\herg_karim\`; generated result files and model checkpoints are git-ignored.

### Extended metrics and scientific figures

Generate additional classification metrics, confusion matrices, and publication-resolution figures from the saved pilot/full predictions without retraining:

```powershell
.\.venv\Scripts\python.exe scripts\09_generate_scientific_visualizations.py
```

Outputs are written under `results\reports\herg_karim\` and `results\visualizations\herg_karim\`. Full-run results remain distinct from the pilot, whose holdout was sampled within the frozen `scaffold_r0` training partition and is not independent test performance. See `scientific_visualization_report.md` and `figure_manifest.json` for figure descriptions and provenance. The reporting step verifies that original baseline artifacts remain unchanged.

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
