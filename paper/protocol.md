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

## 2026-11-18 fidelity-kernel pilot protocol (locked before pilot test evaluation)
- This is a small, single-repeat pilot on `herg_karim/scaffold_r0`, not a replacement for the five-repeat classical benchmark. It uses 64 training, 128 validation, and 128 test molecules sampled without replacement from their corresponding frozen partitions. The base sampling seed is `20261118`; the train, validation, and test samplers use `seed`, `seed + 1`, and `seed + 2`, respectively. Sampling is independent of test labels.
- The fidelity model input is the concatenation of the frozen ECFP4-2048 bits and all 14 frozen RDKit descriptors. A `StandardScaler` and four-component PCA (`svd_solver="full"`) are fitted only to the 64 sampled training molecules. Validation and test molecules are transformed using those training-fitted parameters. No feature selection is used.
- The quantum feature map is Qiskit's four-qubit `ZZFeatureMap`, with one repetition and linear entanglement. `FidelityStatevectorKernel` runs exact statevector simulation (`shots=None`, PSD enforcement enabled); no hardware or shot-based sampling is used.
- The matched-molecule classical control is an RBF-free precomputed Tanimoto-SVM on ECFP4-2048. Both SVMs receive the same 12 validation candidates: `C` in `[0.01, 0.1, 1, 10, 100, 1000]` crossed with unweighted and balanced class weights. Select each model by validation AUROC; ties follow candidate order. Select each classification threshold on validation by balanced accuracy, breaking ties by proximity to 0.5 and then the lower threshold.
- SVM probability estimates are enabled with a fixed random state and are fitted using training molecules only. The selected parameters and validation thresholds are written to `locked_configuration.json` before any test kernel is constructed or test labels are loaded. The separate evaluation phase refuses to run without this lock.
- Test metrics are descriptive for this 128-molecule pilot holdout only. Report AUROC, AUPRC, positive F1, macro-F1, balanced accuracy, MCC, Brier score, and log loss, together with kernel construction, final training, and test prediction runtimes. Also retain validation selection evidence, per-partition kernel matrices, predictions, PCA/scaler parameters, circuit serialization, model checkpoints, input checksums, and runtime package versions. The single pilot result has no across-repeat uncertainty interval and cannot establish superiority or quantum advantage.
- The fidelity pilot and its Tanimoto control are stored under `results/quantum_fidelity/`; neither may overwrite prior classical or pilot outputs.

## 2026-11-05 post-evaluation reporting addendum
- After the classical baseline test predictions were produced, extended descriptive reporting was requested for positive-class F1, macro-F1, weighted F1, precision, and recall, in addition to the existing metrics. These are supplementary post-evaluation metrics, not newly preregistered primary endpoints, and do not alter model selection or the primary AUROC comparison.
- All thresholded metrics use each split's already-saved threshold selected on validation by balanced accuracy. AUROC, AUPRC, Brier score, and log loss use the saved class-1 probabilities. Figures and tables are derived from existing molecule-level predictions; no model is retrained and test results are not used to select models, parameters, or thresholds.
- The pilot holdout was sampled only from the frozen `scaffold_r0` training partition. Its scores and figures must remain explicitly labeled as pilot holdout results, not independent test performance.
- Report mean, sample standard deviation, and descriptive two-sided 95% Student-t intervals across the five repeated scaffold splits. For a single pilot holdout, the standard deviation and interval are undefined. Repeats share molecules, so intervals are descriptive and not tests of superiority.
- Dataset distribution and sampled ECFP4 similarity plots are descriptive. Frozen test novelty values may be joined to full-run prediction rows for future analysis, but this addendum does not perform SHAP, calibration, conformal, or novelty-versus-performance analyses. Test labels must not be used to fit calibration or conformal procedures.

## 2026-11-18 trainable fidelity-kernel protocol (locked before test evaluation)
- The first trainable pilot reuses the exact 64/128/128 molecule IDs and train-fitted
  scaling/PCA parameters from `fidelity_statevector_pilot_r0_final`.
- The four-qubit feature map combines the existing linear-entangled `ZZFeatureMap`
  with four trainable `RY` rotations. Training parameters are optimized with SLSQP
  using `SVCLoss` on training molecules only.
- Validation molecules are used only to select the C value and unweighted/balanced
  class weight. The validation threshold maximizes balanced accuracy. Test data
  remains untouched until these choices are locked.
