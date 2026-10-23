# Quantum Kernels for Toxicity Prediction on Unfamiliar Molecular Structures
CSE 491 - hERG blockade & DILI on scaffold/cluster-novel test molecules.

## Setup
    conda env create -f environment.yml && conda activate qktox
    make test && make data

## Layout
- `configs/`  all settings (edit here, not in code)    - `src/qktox/`  library code
- `scripts/`  numbered pipeline steps                   - `data/splits/` FROZEN splits (committed)
- `results/`  molecule-level predictions per model (every table regenerates from these)

## Rules (publication-grade reproducibility)
1. Test set is never used for any choice. The primary comparison is written in `paper/protocol.md` BEFORE test runs.
2. Splits are generated once, committed, tagged (`git tag splits-v1`).
3. Every model saves predictions to `results/<dataset>/<split>/<model>.csv`.
4. DILIrank 2.0 is not in TDC: put the curated file at `data/raw/dilirank2.csv` (columns Drug, Y, Drug_ID).
