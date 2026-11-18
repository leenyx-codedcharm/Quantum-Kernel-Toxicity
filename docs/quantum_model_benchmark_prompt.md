# Quantum Kernel Toxicity
# Advanced Quantum Model Implementation and Benchmarking

You are working on the Quantum Kernel Toxicity research repository.

The existing project has:
- Classical baselines: LightGBM, Random Forest and Tanimoto-SVM.
- A completed 4-qubit Fidelity Quantum Kernel pilot.
- Five frozen hERG_Karim scaffold splits.
- Molecular fingerprints: ECFP4, 2048 bits.
- RDKit descriptors: 14 features.
- A separate quantum environment: .venv-quantum.

Research objective:

Investigate whether different quantum architectures and trainable
quantum kernels can improve molecular toxicity prediction on
structurally unfamiliar molecules.

==================================================
1. ENVIRONMENT AND DEPENDENCIES
==================================================

Use ONLY:

.venv-quantum

Existing reported versions:
- Python 3.11.9
- Qiskit 2.6.2
- Qiskit Machine Learning 0.9.1
- NumPy 2.4.6
- Pandas 3.1.1
- Scikit-learn 1.9.1

First inspect the installed packages and dependency compatibility.

Check:
- scipy
- qiskit
- qiskit-machine-learning
- qiskit-algorithms
- numpy
- pandas
- scikit-learn

Install only missing packages with versions compatible with the
existing Qiskit environment.

SLSQP and COBYLA can use scipy.optimize where appropriate.
Use Qiskit Algorithms SPSA if compatible.

Do not install PennyLane unless a specific implementation
requirement justifies it.

Do not modify the existing .venv environment.
Do not unnecessarily upgrade existing packages.

Record all installed package versions.

==================================================
2. EXISTING EXPERIMENTS MUST BE PRESERVED
==================================================

Inspect the completed experiment:

fidelity_statevector_pilot_r0_final/

Preserve:
- Existing kernel matrices.
- Existing model checkpoints.
- Existing predictions.
- Existing PCA and scaling parameters.
- Existing configuration.
- Existing experiment manifest.
- Existing results.

Do not overwrite or modify the completed fidelity experiment.

Existing pilot results:

Fidelity SVM:
- AUROC: 0.5704
- AUPRC: 0.4499
- Positive F1: 0.4082
- Macro-F1: 0.5205
- Balanced accuracy: 0.5205
- MCC: 0.0411

Tanimoto SVM:
- AUROC: 0.6590
- AUPRC: 0.5752
- Positive F1: 0.5225
- Macro-F1: 0.5785
- Balanced accuracy: 0.5871
- MCC: 0.1693

These are single-split pilot results, not final publication results.

==================================================
3. COMMON EXPERIMENTAL CONFIGURATION
==================================================

Use:
- Four qubits.
- Four molecular input dimensions.
- ECFP4 plus RDKit descriptors as the primary input.
- Scaling fitted on training data only.
- PCA with four components, fitted on training data only.
- ZZ feature maps where appropriate.
- Exact statevector simulation for deterministic initial experiments.
- Fixed random seeds.

Use the existing frozen scaffold_r0 split for the initial
architecture pilot.

Initial matched experiment:
- 64 training molecules.
- 128 validation molecules.
- 128 test molecules.

Use the exact same molecule IDs as the existing fidelity pilot.

Training molecules must come exclusively from the frozen
training partition.

Validation molecules must come from the frozen validation partition.

Test molecules must come from the frozen test partition.

Do not generate new random train/test splits.

After the pilot passes, evaluate the selected configurations
across all five frozen scaffold splits.

Explore training subset sizes of:
- 64
- 128
- 256
- 512

Increase computational scale only after measuring runtime.

==================================================
4. MODEL A: VARIATIONAL QUANTUM CLASSIFIER
==================================================

Implement a four-qubit VQC.

Architecture:
- Four input qubits.
- Parameterized rotation gates.
- Trainable ansatz.
- Measurement-based binary classification.
- Cross-entropy loss.

Compare three optimizers:

A. SLSQP
B. COBYLA
C. SPSA

Use identical:
- Molecular features.
- Training IDs.
- Validation IDs.
- Test IDs.
- Circuit architecture.
- Evaluation protocol.

Only change the optimizer.

Record:
- Optimization iterations.
- Loss convergence.
- Final parameters.
- Training runtime.
- Validation performance.
- Test performance after configuration lock.

Use exact statevector evaluation where feasible.
If SPSA requires a shot-based or stochastic objective,
document that distinction and keep its evaluation protocol
clear.

==================================================
5. MODEL B: DATA RE-UPLOADING VQC
==================================================

Implement quantum data re-uploading.

Repeatedly encode the same molecular input features
through multiple circuit layers.

Test:
- 2 re-upload layers.
- 4 re-upload layers.
- 6 re-upload layers.

Start with four qubits.

Compare:
- No entanglement.
- Linear entanglement.
- Full entanglement.

Use SLSQP initially.

Measure whether repeated encoding improves:
- AUROC.
- AUPRC.
- Positive F1.
- Macro-F1.
- MCC.

Avoid excessive circuit depth and unnecessary parameters.

==================================================
6. MODEL C: PROJECTED QUANTUM KERNEL
==================================================

Investigate projected quantum kernels.

Inspect the installed Qiskit Machine Learning API first.

If a documented projected-kernel implementation is available,
use it.

Otherwise implement a clearly documented projected quantum
feature representation using measured quantum observables
and a classical similarity kernel.

Do not incorrectly label an ordinary quantum expectation-value
feature vector as a standard projected quantum kernel without
documenting its mathematical definition.

Compare:
- Linear projected-feature kernel.
- RBF projected-feature kernel.

Use an SVM with precomputed kernels.

Save the projected feature vectors and kernel matrices.

==================================================
7. MODEL D: TRAINABLE FIDELITY QUANTUM KERNEL
==================================================

Investigate the available:

- TrainableFidelityQuantumKernel
- QuantumKernelTrainer
- SVCLoss

Use official supported APIs where available.

Investigate optimizing trainable feature-map parameters
using SLSQP.

The optimization objective must use training data only.

Validation data may be used for model selection.

Never use test labels or test performance to optimize
quantum circuit parameters.

If the installed API does not support the intended
implementation, document the limitation rather than
silently substituting another method.

==================================================
8. ENTANGLEMENT ABLATION
==================================================

For selected four-qubit architectures compare:

- No entanglement.
- Linear entanglement.
- Full entanglement.

Keep other circuit parameters fixed during each ablation.

Record:
- Qubit count.
- Circuit depth.
- Gate count.
- Two-qubit gate count.
- Trainable parameter count.
- Kernel computation time.
- Predictive performance.

Do not assume more entanglement produces higher accuracy.

==================================================
9. BENCHMARKING AND F1 SCORING
==================================================

Every model must report:

1. Accuracy.
2. Balanced accuracy.
3. Precision.
4. Recall.
5. Positive-class F1.
6. Macro-F1.
7. Weighted F1.
8. AUROC.
9. AUPRC.
10. MCC.
11. Brier score.
12. Log loss.

Select classification thresholds using validation data only.

Use the same threshold-selection rule across models:
maximize validation balanced accuracy.

Record the selected threshold for every experiment.

Never tune using test labels.

For the initial pilot, report the 128-molecule test results
separately from the full scaffold experiments.

For the final benchmark, report:
- Mean across five scaffold splits.
- Standard deviation.
- Descriptive confidence intervals.
- Per-split results.

Clearly explain that repeated scaffold splits reuse molecules
across different repetitions; do not treat all predictions
as independent observations.

==================================================
10. FAIR CLASSICAL COMPARISON
==================================================

Compare every quantum architecture against:

- Fixed Fidelity Quantum Kernel + SVM.
- Tanimoto-SVM.

Use exactly the same selected training, validation and test
molecules for controlled comparisons.

Where practical, also retrain:
- Random Forest with combined features.
- LightGBM with combined features.

on the same selected subsets.

Do not directly compare the small quantum pilot against
the full-data five-split classical results as if they were
matched experiments.

==================================================
11. VISUALIZATION AND REPORTING
==================================================

Generate publication-quality graphs:

- AUROC comparison.
- AUPRC comparison.
- Positive F1 comparison.
- Macro-F1 comparison.
- MCC comparison.
- Balanced accuracy comparison.
- ROC curves.
- Precision-recall curves.
- Confusion matrices.
- Optimizer convergence plots.
- Circuit depth and gate-count comparison.
- Runtime comparison.
- Entanglement ablation plots.
- Training-size versus performance plots.

Use consistent formatting across all models.

Save numeric results and molecule-level predictions.

==================================================
12. OUTPUT STRUCTURE
==================================================

Create a separate results directory:

results/quantum_benchmarks/herg_karim/

Organize results by:

- vqc_slsqp/
- vqc_cobyla/
- vqc_spsa/
- data_reuploading/
- projected_quantum_kernel/
- trainable_fidelity_kernel/
- entanglement_ablation/
- comparisons/
- visualizations/
- reports/

Save:
- Model checkpoints.
- Kernel matrices.
- Predictions.
- Circuit configurations.
- Optimization histories.
- Metrics CSV files.
- Experiment manifests.
- Runtime logs.
- Reproducibility information.

Do not overwrite existing fidelity pilot outputs.

==================================================
13. TESTING AND VALIDATION
==================================================

Add automated tests for:

- Circuit construction.
- Four-qubit configuration.
- Kernel symmetry.
- Kernel diagonal values.
- Kernel positive-semidefinite diagnostics.
- Finite values.
- Feature alignment.
- Molecule ID alignment.
- Optimizer convergence.
- Prediction dimensions.
- F1 calculations.
- Threshold selection.
- Train/validation/test separation.

Run tests using the quantum environment.

Keep quantum-specific tests separate from the existing
RDKit environment tests.

==================================================
14. STRICT SCIENTIFIC RESTRICTIONS
==================================================

- Do not modify frozen scaffold splits.
- Do not modify original molecular datasets.
- Do not use test data for optimization.
- Do not claim quantum advantage from simulator experiments.
- Do not select the winning model using test performance.
- Do not invent results.
- Do not overwrite completed experiments.
- Do not combine different experimental subsets without
  documenting their differences.
- Do not call an optimizer a separate quantum model.
- Distinguish architecture comparisons from optimizer ablations.
- Preserve all molecule IDs and provenance.

==================================================
15. IMPLEMENTATION ORDER
==================================================

Work in this order:

1. Inspect environment and install missing dependencies.
2. Implement VQC with SLSQP.
3. Implement the same VQC with COBYLA and SPSA.
4. Implement data re-uploading VQC.
5. Implement projected quantum kernel.
6. Implement trainable fidelity kernel.
7. Run the matched pilot experiments.
8. Generate F1 and other metric comparisons.
9. Evaluate selected models across five scaffold splits.
10. Produce a final scientific benchmark report.

Start with environment inspection and a small VQC pilot.

Do not launch all large experiments simultaneously.

At completion, report:
- Packages installed.
- Models implemented.
- Actual pilot results.
- F1 scores.
- AUROC and AUPRC.
- Runtime.
- Circuit configurations.
- Tests passed.
- Unresolved limitations.
- Exact commands to reproduce every experiment.

The objective is a rigorous, reproducible comparison suitable
for a peer-reviewed molecular machine-learning research paper.
