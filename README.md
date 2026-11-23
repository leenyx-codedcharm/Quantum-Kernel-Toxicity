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

<!-- PENNYLANE-RESEARCH-STATUS -->
## Current Research Status â€” November 2026

This repository evaluates classical and quantum machine learning for
hERG inhibition prediction using the hERG_Karim molecular dataset.

### Dataset
- 10,824 curated molecules
- 5,482 negative and 5,342 positive samples
- 2,048-bit ECFP4 fingerprints
- 14 RDKit descriptors
- Five frozen scaffold splits

### Classical baselines
| Model | Positive F1 | AUROC |
|---|---:|---:|
| Random Forest | 0.7833 | 0.8759 |
| LightGBM | 0.7666 | 0.8616 |
| Tanimoto-SVM | 0.7631 | 0.8605 |

These are previously reported five-scaffold results.

### PennyLane development
- PennyLane 0.45.1 isolated environment
- Four-qubit QRC reproduced at 2, 4 and 6 layers
- Full 320-molecule pilot feature parity verified
- Exploratory data-reuploading Hybrid VQC implemented

The Hybrid VQC pilot achieved AUROC 0.5448 and positive F1 0.4771
on one 128-molecule test subset. It has not demonstrated superiority
over the classical baselines.

### Research documentation
- [Research progress](docs/RESEARCH_PROGRESS.md)
- [PennyLane migration](docs/PENNYLANE_MIGRATION.md)

### Reproducibility
The original Qiskit experiments, frozen data splits and previous
results are retained. New PennyLane experiments use a separate
environment and output directories.

The project does not claim quantum advantage. Further five-scaffold
benchmarking, QM-enhanced experiments and literature review remain
in progress.
<!-- END PENNYLANE-RESEARCH-STATUS -->

---

# Research Programme: Quantum Machine Learning for hERG Prediction

## Research Overview

This project investigates classical, quantum and hybrid quantum-classical
machine learning methods for molecular hERG inhibition prediction.

The research examines quantum kernels, quantum reservoir computing,
variational quantum classifiers and molecular electronic descriptors.

No quantum advantage is currently claimed. Existing classical
baselines outperform the exploratory quantum experiments.

## Dataset

The curated hERG_Karim dataset contains 10,824 molecules.

| Class | Label | Samples | Percentage |
|---|---:|---:|---:|
| Negative | 0 | 5,482 | 50.65% |
| Positive | 1 | 5,342 | 49.35% |
| Total | | 10,824 | 100% |

Molecular representations:
- 2,048-bit ECFP4 fingerprints.
- 14 RDKit descriptors.
- Four-dimensional PCA representation for the initial quantum pilot.

## Research Results

Previously reported five-scaffold classical baselines:

| Model | Positive F1 | AUROC |
|---|---:|---:|
| Random Forest | 0.7833 | 0.8759 |
| LightGBM | 0.7666 | 0.8616 |
| Tanimoto-SVM | 0.7631 | 0.8605 |

Exploratory quantum results:

| Model | Positive F1 | AUROC |
|---|---:|---:|
| Quantum Reservoir (2 layers) | 0.4483 | 0.5469 |
| Quantum Reservoir (4 layers) | 0.3689 | 0.5043 |
| Quantum Reservoir (6 layers) | 0.1333 | 0.5332 |
| PennyLane Hybrid VQC | 0.4771 | 0.5448 |

**Evaluation distinction:** The classical results above are five-scaffold results, while the quantum results are from a single exploratory pilot. They are not directly comparable as final benchmark estimates.

## PennyLane Implementation

The project has migrated its new quantum development to PennyLane.

Completed:
- Four-qubit QRC reproduction.
- Numerical parity at 2, 4 and 6 encoding layers.
- Full 320-molecule pilot feature generation.
- PennyLane Hybrid VQC with data re-uploading.
- Validation-based model selection and separate test evaluation.

The maximum observed Qiskit–PennyLane feature difference was approximately 1.55e-15.

## Research Documentation

- [Project overview](docs/PROJECT_OVERVIEW.md)
- [Experimental status](docs/EXPERIMENTAL_STATUS.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Research progress](docs/RESEARCH_PROGRESS.md)
- [PennyLane migration](docs/PENNYLANE_MIGRATION.md)

## Current Research Priorities

1. Complete a verified Q1 literature review.
2. Investigate trainable data-reuploading quantum architectures.
3. Evaluate different training-set sizes.
4. Integrate QM descriptors into hybrid models.
5. Complete matched five-scaffold benchmarking.
6. Assess whether quantum representations provide measurable incremental value.

## Scientific Reporting

All experiments must preserve frozen scaffold partitions, training-only
preprocessing, validation-based model selection and independent final
test evaluation.

Negative and inconclusive findings are retained and reported.
Circuit equivalence does not establish predictive advantage.
