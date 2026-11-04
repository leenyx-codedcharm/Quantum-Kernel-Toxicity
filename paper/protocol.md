# Pre-registered protocol (commit BEFORE any test-set run)
- Primary dataset: `herg_karim`, evaluated on the five frozen repeated scaffold splits.
- Primary metric: AUROC. The primary comparison is the best quantum model versus the best classical model in the matched-input track, with models selected using training/validation data only.
- Novelty bins are fixed in `configs/data.yaml`: familiar [0.0, 0.4], intermediate (0.4, 0.6], and novel (0.6, 1.0].
- The test set is held out from every modeling choice. Hyperparameters are tuned with scaffold-grouped cross-validation on training data; calibration and the decision threshold are selected on validation data only.
- Scaling, imputation, PCA, and other learned transforms are fit on training molecules only.
- Every model family must receive the same tuning budget. The numeric budget and the null hypothesis/minimum meaningful gain have not yet been specified; declare them before running or examining test results.
- Classical hERG_Karim baselines use the already-frozen five `scaffold_r0`–`scaffold_r4` splits and the committed ECFP4/descriptor features. LightGBM and Random Forest each receive four validation-AUROC candidates per representation (descriptors, ECFP4, and concatenated features); Tanimoto-SVM receives 12 candidates (six C values crossed with unweighted/balanced class weights). Thus each model family receives 12 candidate configurations per repeat. Select hyperparameters by validation AUROC; select the classification threshold by validation balanced accuracy. SVM probabilities use a stratified five-fold out-of-fold margin calibration fit on training data only. No test values are used during selection or threshold choice.
- Report the mean, sample standard deviation, and descriptive 95% Student-t interval across the five repeated scaffold splits for each test metric. Because these repeats share molecules and are not independent samples, these intervals are descriptive and do not establish statistical superiority.
- Report the number of examples and positive rate for each evaluation group. For secondary `herg` and `dili` datasets, pool the five repeats and report intervals.

No performance results are asserted here. A result without a quantum gain is valid; do not claim quantum advantage.
