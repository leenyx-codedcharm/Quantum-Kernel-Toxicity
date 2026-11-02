# Copilot standing rules: quantum-kernel-toxicity

Save this file as `.github/copilot-instructions.md` in the repo. Copilot loads it automatically in every chat.

## Project in one paragraph

CSE 491 research project. We test whether quantum kernels (projected and fidelity, simulated only) predict hERG blockade and drug-induced liver injury (DILI) better than classical models on structurally unfamiliar molecules. Goal: a publishable, fair, reproducible comparison. A result with no quantum gain is acceptable. Never claim quantum advantage.

## Environment (Windows)

- Windows, PowerShell, Python 3.11.9, virtual environment in `.venv`. No conda, no `make`.
- Write commands for PowerShell. Chain with `;` not `&&`. Paths with forward slashes inside Python.
- Dependencies go in `requirements.txt`. Do not upgrade pytdc (0.4.1 is installed on purpose). Record versions with `pip freeze > requirements-lock.txt` after installing anything new.

## Repo layout (do not rename folders)

- `configs/` all settings (YAML). `src/qktox/` library code. `scripts/` numbered pipeline steps. `tests/` pytest.
- `data/raw/` and `data/processed/` are git-ignored (except `data/raw/MANIFEST.json`). `data/splits/` is committed and FROZEN.
- `results/<dataset>/<split>/<model>.csv` holds per-molecule predictions: columns `mol_id,y_true,p_pred`.
- `paper/protocol.md` holds the pre-registered protocol.

## Hard rules (never break these)

1. NEVER reorder, sort, or edit `data/processed/<dataset>.csv`. The split files store row positions of that file. Any sorting happens only in new copies (for example the Excel export).
2. NEVER regenerate or edit `data/splits/*_scaffold.json` (tag `splits-v1`). New splits go in NEW files with a NEW tag (for example `splits-v2` or `cluster-v1`).
3. The test set is never used to choose anything: not hyperparameters, not thresholds, not features, not calibration. Tune with grouped cross-validation on train (group = scaffold). Fit calibration and pick the decision threshold on the validation part only.
4. Scaling, imputation, PCA, and any learned transform are fitted on training molecules only, then applied to val and test.
5. Every model family gets the same tuning budget (set in `configs/tuning.yaml`). Log runtime for every fit.
6. Duplicates and salts of one compound share `parent_key` and must stay on the same side of any split.
7. Novelty bins are fixed in `configs/data.yaml`: [0.0, 0.4, 0.6, 1.0] = familiar, intermediate, novel. Do not change them.
8. Primary dataset is hERG_Karim. hERG and DILI are secondary (small test sets): pool the 5 repeats and report intervals.
9. Never invent numbers, SMILES, citations, or results. If something cannot be run or looked up, say so plainly.
10. Do not commit raw or processed data, model weights, or large caches.

## Coding style

- Config-driven (no hard-coded paths or seeds). Fixed seeds from config. Type hints and short docstrings.
- Each script prints a short summary and exits non-zero on failure. Add a pytest for every new module, including a leakage or fit-on-train-only test where relevant.
- Small commits on a branch named `stage-<n>-<topic>`. Never push to `main` directly. Commit message states what and why.

## How to work

- Do one stage at a time. After each stage: run `python -m pytest -q`, show the commands to run, list files created, state what to check, then STOP and wait for the go-ahead.
- If a requirement conflicts with a hard rule, follow the hard rule and tell me.
