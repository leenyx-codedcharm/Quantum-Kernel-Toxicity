# Pre-registered protocol (commit BEFORE any test-set run)
- Primary dataset: `herg_karim`, evaluated on the five frozen repeated scaffold splits.
- Primary metric: AUROC. The primary comparison is the best quantum model versus the best classical model in the matched-input track, with models selected using training/validation data only.
- Novelty bins are fixed in `configs/data.yaml`: familiar [0.0, 0.4], intermediate (0.4, 0.6], and novel (0.6, 1.0].
- The test set is held out from every modeling choice. Hyperparameters are tuned with scaffold-grouped cross-validation on training data; calibration and the decision threshold are selected on validation data only.
- Scaling, imputation, PCA, and other learned transforms are fit on training molecules only.
- Every model family must receive the same tuning budget. The numeric budget and the null hypothesis/minimum meaningful gain have not yet been specified; declare them before running or examining test results.
- Report the number of examples and positive rate for each evaluation group. For secondary `herg` and `dili` datasets, pool the five repeats and report intervals.

No performance results are asserted here. A result without a quantum gain is valid; do not claim quantum advantage.
