# hERG_Karim scientific visualisation and metric report

This report was generated from the existing molecule-level predictions. No baseline model was retrained, no feature or split was regenerated, and no test result was used for model selection.

The full experiment contains five repeated frozen scaffold splits. Its 95% t intervals are descriptive because the repeats share molecules. The pilot is a single 512/128/128 training/validation/holdout experiment sampled only from the frozen `scaffold_r0` training partition; pilot results are **not independent test performance** and have no across-repeat confidence interval.

## Dataset summary

- Curated molecules: 10,824.
- Class counts (0/1): 5,482 / 5,342.
- Non-empty scaffold groups: 5,784; molecules without a scaffold: 17.
- Similarity sample: 50,000 unique random pairs and 512 nearest-neighbor queries using the frozen ECFP4 settings.

## Figures

### Curated class distribution

- File: `D:/qk-tox/results/visualizations/herg_karim/dataset_analysis/class_distribution.png`
- Description: Counts of curated non-blockers and blockers from the frozen hERG_Karim feature metadata.

### RDKit descriptor distributions by class

- File: `D:/qk-tox/results/visualizations/herg_karim/dataset_analysis/descriptor_distributions.png`
- Description: All documented RDKit descriptor values, shown separately by class; no feature scaling or selection is applied.

### Scaffold distribution

- File: `D:/qk-tox/results/visualizations/herg_karim/dataset_analysis/scaffold_distribution.png`
- Description: Top 20 non-empty scaffold groups, a combined long-tail group, and molecules without a scaffold. Exact scaffold counts are in scaffold_counts.csv.

### Class balance across train, validation, and test

- File: `D:/qk-tox/results/visualizations/herg_karim/dataset_analysis/class_balance_by_partition.png`
- Description: Counts and positive rates for each of the five frozen scaffold repeats. The evaluation partitions are shown for descriptive analysis only.

### Molecular similarity distributions

- File: `D:/qk-tox/results/visualizations/herg_karim/dataset_analysis/molecular_similarity_distributions.png`
- Description: Chirality-aware ECFP4 (radius 2, 2048 bits): a deterministic sample of unique random molecule pairs and nearest-neighbor similarities for a deterministic query sample. Sampling details and raw values are recorded in molecular_similarity_sample.csv.

### Full ROC curves

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/full/roc_curves.png`
- Description: Each curve is the mean interpolated ROC across held-out split predictions; full-run ribbons show one split standard deviation. Pilot curves use only the training-partition holdout.

### Full precision-recall curves

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/full/precision_recall_curves.png`
- Description: Each curve is the mean interpolated precision-recall curve across held-out split predictions; full-run ribbons show one split standard deviation. Pilot curves are not independent test performance.

### Full AUROC comparison

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/full/auroc_comparison.png`
- Description: Mean held-out auroc by model with descriptive t-based 95% confidence intervals across repeats; pilot intervals are undefined for a single holdout.

### Full AUPRC comparison

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/full/auprc_comparison.png`
- Description: Mean held-out auprc by model with descriptive t-based 95% confidence intervals across repeats; pilot intervals are undefined for a single holdout.

### Full F1-score comparison

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/full/f1_comparison.png`
- Description: Positive-class, macro, and weighted F1 computed from validation-selected thresholds; bars show means and descriptive 95% t intervals where repeats permit.

### Full MCC comparison

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/full/mcc_comparison.png`
- Description: Mean held-out mcc by model with descriptive t-based 95% confidence intervals across repeats; pilot intervals are undefined for a single holdout.

### Full confusion matrices

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/full/confusion_matrices.png`
- Description: Counts are summed across held-out predictions using validation-selected thresholds. Full-repeat test molecules overlap across repeats, so pooled counts are descriptive, not independent observations.

### Full split-level performance

- File: `D:/qk-tox/results/visualizations/herg_karim/split_analysis/full/split_level_performance.png`
- Description: Each point is a split-level held-out metric calculated from saved molecule predictions. The pilot panel represents one training-partition holdout only.

### Pilot ROC curves

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/pilot/roc_curves.png`
- Description: Each curve is the mean interpolated ROC across held-out split predictions; full-run ribbons show one split standard deviation. Pilot curves use only the training-partition holdout.

### Pilot precision-recall curves

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/pilot/precision_recall_curves.png`
- Description: Each curve is the mean interpolated precision-recall curve across held-out split predictions; full-run ribbons show one split standard deviation. Pilot curves are not independent test performance.

### Pilot AUROC comparison

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/pilot/auroc_comparison.png`
- Description: Mean held-out auroc by model with descriptive t-based 95% confidence intervals across repeats; pilot intervals are undefined for a single holdout.

### Pilot AUPRC comparison

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/pilot/auprc_comparison.png`
- Description: Mean held-out auprc by model with descriptive t-based 95% confidence intervals across repeats; pilot intervals are undefined for a single holdout.

### Pilot F1-score comparison

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/pilot/f1_comparison.png`
- Description: Positive-class, macro, and weighted F1 computed from validation-selected thresholds; bars show means and descriptive 95% t intervals where repeats permit.

### Pilot MCC comparison

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/pilot/mcc_comparison.png`
- Description: Mean held-out mcc by model with descriptive t-based 95% confidence intervals across repeats; pilot intervals are undefined for a single holdout.

### Pilot confusion matrices

- File: `D:/qk-tox/results/visualizations/herg_karim/model_comparison/pilot/confusion_matrices.png`
- Description: Counts are summed across held-out predictions using validation-selected thresholds. Full-repeat test molecules overlap across repeats, so pooled counts are descriptive, not independent observations.

### Pilot split-level performance

- File: `D:/qk-tox/results/visualizations/herg_karim/split_analysis/pilot/split_level_performance.png`
- Description: Each point is a split-level held-out metric calculated from saved molecule predictions. The pilot panel represents one training-partition holdout only.

## Metric definitions and integrity

- Precision, recall, positive-class F1, macro-F1, weighted F1, balanced accuracy, and MCC use the saved threshold selected on validation by balanced accuracy.
- AUROC, average precision (AUPRC), Brier score, and log loss use saved class-1 probabilities.
- Per-split metrics are recomputed and checked against the baseline's original six metrics; prediction IDs and labels are checked against curated metadata and frozen test membership.
- Hyperparameter selections are checked against the best recorded validation-AUROC tuning trial. The thresholds are taken only from the baseline's recorded validation-selected thresholds.
- Pilot held-out IDs are verified to belong to the frozen `scaffold_r0` training partition. Its predictions are presented separately from official test results.
- See `complete_metric_comparison.csv`, `per_split_metrics.csv`, and `confusion_matrices.csv` for numeric tables.

## Prepared advanced-analysis inputs (not analyses)

The `advanced_analysis_inputs.csv` and `advanced_analysis_manifest.json` provide compatible prediction, feature, checkpoint, and frozen novelty references. No SHAP values, calibration curves, conformal sets, or novelty-performance analyses were produced. The conformal template is intentionally empty; test labels must not be used to estimate conformal quantiles.
