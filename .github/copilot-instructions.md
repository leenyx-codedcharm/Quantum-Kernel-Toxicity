# Quantum Kernel Toxicity: Copilot Instructions

## Project and goal

This CSE 491 research project compares simulated quantum kernels (projected and fidelity) with classical models for hERG blockade and drug-induced liver injury (DILI), especially on structurally unfamiliar molecules. The goal is a fair, reproducible comparison suitable for publication. A result with no quantum gain is valid. Never claim quantum advantage or invent results, measurements, SMILES, sources, labels, or citations.

## Environment and project architecture

- Development uses Windows, PowerShell, Python 3.11.9, and the repository virtual environment `.venv`. Do not require Conda or Make.
- Put configuration in `configs/` (YAML), reusable library code in `src/qktox/`, numbered pipeline entry points in `scripts/`, tests in `tests/`, and the pre-registered protocol in `paper/protocol.md`.
- `data/raw/` holds source inputs and `MANIFEST.json`; `data/processed/` holds curated datasets and curation logs; `data/splits/` holds committed splits and cluster assignments; `results/` holds derived predictions and reports.
- Use PowerShell commands and Windows paths. In Python source, forward-slash paths are acceptable. Dependencies belong in `requirements.txt`; preserve PyTDC 0.4.1. Ask before adding major experimental packages such as PennyLane, Qiskit, Chemprop, or PyTorch.

## Datasets and row identity

- Current TDC datasets are `herg`, `herg_karim`, and `dili`; `dilirank2` is a future external dataset and is not assumed to be present.
- Raw TDC tables use `Drug_ID`, `Drug` (SMILES), and `Y`. Curated tables use `orig_id`, `smiles`, `inchikey`, `parent_key`, `scaffold`, `label`, and `acyclic`.
- Labels are binary: 1 means hERG blocker or DILI concern; 0 means non-blocker or no concern.
- A molecule ID is `<dataset>_<row_index:06d>`, where `row_index` is the row position in `data/processed/<dataset>.csv`. Preserve that row identity whenever producing a derived table or prediction.
- `parent_key` is the first InChIKey block. Salts and duplicate representations of the same parent must be merged during curation and must never cross split partitions.
- Only report source or reference information when it exists in the source files; otherwise leave it empty.

## Research protocol and frozen data

- Never edit, reorder, sort, or regenerate `data/processed/<dataset>.csv`. Split positions refer to those exact row positions. Sorting is allowed only in a clearly separate, view-only export.
- Never edit or regenerate `data/splits/*_scaffold.json`. These are frozen as `splits-v1`. New split designs must use new filenames and tags.
- Scaffold split files use `scaffold_r0` through `scaffold_r4`, each with `train`, `val`, `test`, and `test_novelty`; novelty values align positionally with `test`.
- Configured novelty boundaries are exactly `[0.0, 0.4, 0.6, 1.0]`: familiar `[0.0, 0.4]`, intermediate `(0.4, 0.6]`, novel `(0.6, 1.0]`. Do not change them.
- The test set is for final evaluation only. Never use it to select features, preprocessing, hyperparameters, calibration, thresholds, or other choices. Tune by scaffold-grouped cross-validation using training data. Use validation data for calibration and decision threshold selection.
- Fit scaling, imputation, PCA, and every learned transformation on training molecules only; apply the fitted transform to validation and test.
- Every model family receives the same tuning budget from configuration. Record fit runtime and package versions.
- The primary dataset is `herg_karim`. `herg` and `dili` are secondary small datasets; pool their five repeats and report intervals.

## Prediction and output format

- Save per-molecule predictions under `results/<dataset>/<split>/<model>.csv` with exactly the shared fields `mol_id,y_true,p_pred`; `p_pred` is the class-1 probability. Final prediction files contain test molecules only.
- Store fit time and model metadata (chosen parameters, tuning trials, package versions) in an adjacent `<model>.json` sidecar.
- Derived outputs must be reproducible from preserved inputs/configuration and saved predictions. Do not commit raw or processed data, model weights, or large caches.
- Excel or other human-facing exports are view-only; never read them back into the pipeline.

## Coding, testing, and new experimental scripts

- Follow existing Python patterns. Use type hints, concise docstrings, configuration-driven paths/seeds, fixed configured random seeds, and explicit failures. Do not hide errors behind broad catches or silent fallbacks.
- Every new module needs focused pytest coverage. Test train-only fitting or leakage protections when relevant. Scripts should print a concise summary and exit non-zero on failure.
- Before adding a script, inspect existing code and reuse its helpers. A new experiment must read frozen data without modifying it, write only to its new designated outputs, document the model and inputs, log runtime and versions, and keep validation/test use within the protocol.
- Do not silently change the research protocol. Record methodological decisions in `paper/protocol.md` before test-set evaluation; ask rather than invent unspecified scientific choices.
- Run `python -m pytest -q` after each stage. Run the relevant script only when its required data and dependencies are available. Work on one stage at a time; report files, commands, results, assumptions, and remaining checks, then stop for approval.
- Use a branch named `stage-<n>-<topic>`. Commit only relevant verified changes, use a message stating what and why, and never push directly to `main`.

## Matched-input track and quantum code

- Before any test-set evaluation, add the matched-input declarations to `paper/protocol.md` as a proposed draft and wait for team approval before committing them. Do not change the frozen datasets, scaffold splits, novelty boundaries, or existing results.
- Generate the full RDKit 2D descriptor set from frozen curated hERG_Karim SMILES into new outputs only. For each frozen split, fit imputation, zero-variance removal, percentile clipping, standardisation, 12-component PCA, and training-range min-max scaling on training rows only. Keep the 4-component input equal to the first four columns of the nested 8-component input.
- Compare class-prior, logistic-regression, and RBF-SVM references on those exact inputs. Use the declared 12-candidate validation budgets, select on validation AUROC, choose thresholds on validation balanced accuracy, and calibrate SVM probabilities using training data only.
- Ask before installing PennyLane. The fixed quantum circuit applies RY then RZ on each qubit in each layer, followed by a nearest-neighbor CNOT chain; projected features include per-qubit X/Y/Z and adjacent ZZ expectations. Cross-check the NumPy implementation against PennyLane.
- Restrict the quantum pilot to the configured scaffold_r0 training subsample and full validation partition; never load test indices or test labels in the pilot. Evaluate the declared go/no-go criteria on validation data only and wait for team approval before any full multi-split quantum experiment.
- Keep all matched-input and pilot products in their new designated, ignored output locations. Add focused leakage and row-identity tests, record versions and runtime, and do not claim a quantum gain without results.
