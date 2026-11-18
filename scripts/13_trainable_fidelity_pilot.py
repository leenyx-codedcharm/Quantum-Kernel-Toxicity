"""Run the matched, two-phase trainable fidelity-kernel pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from qiskit import qpy
from qiskit_algorithms.optimizers import SLSQP
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.utils.loss_functions.kernel_loss_functions import SVCLoss
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qktox.quantum_pilot import (
    apply_train_preprocessor,
    calculate_pilot_metrics,
    select_validation_threshold,
    validate_square_kernel,
)
from qktox.trainable_kernel import build_trainable_fidelity_kernel

DEFAULT_OUTPUT = Path(
    "results/quantum_benchmarks/herg_karim/fidelity_trainable"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def read_labels(path: Path, indices: np.ndarray) -> np.ndarray:
    selected = sorted({int(index) for index in indices})
    if len(selected) != len(indices):
        raise ValueError("Requested label row positions must be unique")
    selected_set = set(selected)
    labels = pd.read_csv(
        path,
        usecols=["label"],
        skiprows=lambda line_number: line_number != 0
        and line_number - 1 not in selected_set,
    )["label"]
    numeric = pd.to_numeric(labels, errors="coerce")
    if len(labels) != len(selected) or numeric.isna().any():
        raise ValueError("Could not align all requested label rows")
    if not set(numeric.unique()).issubset({0, 1}):
        raise ValueError("Requested labels must be binary 0/1")
    return numeric.to_numpy(dtype=np.int8)


def load_inputs(root: Path) -> dict[str, Any]:
    lock_path = (
        root
        / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/locked_configuration.json"
    )
    preprocessing_path = (
        root
        / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/preprocessing/scaler_pca_parameters.npz"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    with np.load(preprocessing_path, allow_pickle=False) as archive:
        preprocessing = {key: archive[key] for key in archive.files}

    feature_dir = root / "data/processed/features/hERG_Karim"
    with np.load(feature_dir / "ecfp4_2048.npz", allow_pickle=False) as archive:
        fingerprints = archive["fingerprints"]
        fingerprint_ids = archive["mol_ids"].astype(str)
    descriptors = pd.read_csv(feature_dir / "rdkit_descriptors.csv")
    descriptor_names = json.loads(
        (feature_dir / "feature_manifest.json").read_text(encoding="utf-8")
    )["descriptor_names"]
    descriptor_ids = descriptors["mol_id"].astype(str).to_numpy()
    combined = np.concatenate(
        (
            fingerprints.astype(np.float64),
            descriptors.loc[:, descriptor_names].to_numpy(dtype=np.float64),
        ),
        axis=1,
    )
    expected_ids = np.asarray(
        [f"herg_karim_{index:06d}" for index in range(len(fingerprint_ids))]
    )
    if not np.array_equal(fingerprint_ids, expected_ids):
        raise ValueError("Fingerprint molecule IDs do not align with processed rows")
    if not np.array_equal(descriptor_ids, fingerprint_ids):
        raise ValueError("Descriptor and fingerprint molecule IDs differ")

    indices: dict[str, np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    for partition in ("train", "val", "test"):
        indices[partition] = np.asarray(
            [
                int(molecule_id.rsplit("_", 1)[1])
                for molecule_id in lock["molecule_ids"][partition]
            ],
            dtype=np.int64,
        )
        molecule_ids = [
            f"herg_karim_{int(index):06d}" for index in indices[partition]
        ]
        if molecule_ids != lock["molecule_ids"][partition]:
            raise ValueError(f"{partition}: pilot molecule IDs do not map to frozen rows")
        features[partition] = apply_train_preprocessor(
            combined[indices[partition]], preprocessing
        )
    return {
        "lock": lock,
        "indices": indices,
        "features": features,
        "preprocessing": preprocessing,
    }


def input_checksums(root: Path) -> dict[str, str]:
    paths = {
        "frozen_splits": root / "data/splits/herg_karim_scaffold.json",
        "processed_dataset": root / "data/processed/herg_karim.csv",
        "fingerprints": root / "data/processed/features/hERG_Karim/ecfp4_2048.npz",
        "descriptors": root / "data/processed/features/hERG_Karim/rdkit_descriptors.csv",
        "source_fidelity_lock": root
        / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/locked_configuration.json",
        "source_fidelity_preprocessing": root
        / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/preprocessing/scaler_pca_parameters.npz",
        "config": root / "configs/trainable_fidelity_pilot.yaml",
        "script": root / "scripts/13_trainable_fidelity_pilot.py",
        "module": root / "src/qktox/trainable_kernel.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing trainable-kernel inputs: " + ", ".join(missing))
    return {name: sha256(path) for name, path in paths.items()}


def runtime_info() -> dict[str, Any]:
    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "qiskit": "qiskit",
        "qiskit_machine_learning": "qiskit-machine-learning",
        "qiskit_algorithms": "qiskit-algorithms",
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "sklearn_version": sklearn.__version__,
        "versions": {
            key: importlib.metadata.version(distribution)
            for key, distribution in distributions.items()
        },
    }


def save_predictions(
    path: Path,
    molecule_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "mol_id": molecule_ids,
            "y_true": labels.astype(np.int8),
            "p_pred": probabilities,
        }
    ).to_csv(path, index=False)


def tune_svc(
    config: dict[str, Any],
    train_kernel: np.ndarray,
    validation_kernel: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    tuning_started = time.perf_counter()
    for C in config["tuning"]["C_values"]:
        for class_weight in config["tuning"]["class_weights"]:
            model = SVC(
                C=float(C),
                kernel="precomputed",
                class_weight=class_weight,
                probability=True,
                random_state=int(config["seed"]),
            )
            fit_started = time.perf_counter()
            model.fit(train_kernel, train_labels)
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            probabilities = model.predict_proba(validation_kernel)[:, 1]
            prediction_seconds = time.perf_counter() - predict_started
            auroc = float(roc_auc_score(validation_labels, probabilities))
            trial = {
                "C": float(C),
                "class_weight": class_weight,
                "validation_auroc": auroc,
                "fit_seconds": fit_seconds,
                "validation_prediction_seconds": prediction_seconds,
            }
            trials.append(trial)
            if selected is None or auroc > selected["validation_auroc"]:
                selected = {**trial, "probabilities": probabilities}
    if selected is None:
        raise RuntimeError("No trainable-kernel SVC candidate was produced")
    threshold, balanced_accuracy = select_validation_threshold(
        validation_labels, selected["probabilities"]
    )
    metrics = calculate_pilot_metrics(
        validation_labels, selected["probabilities"], threshold
    )
    return {
        "parameters": {
            "C": selected["C"],
            "class_weight": selected["class_weight"],
            "probability": True,
            "random_state": int(config["seed"]),
        },
        "threshold": threshold,
        "validation_balanced_accuracy_at_threshold": balanced_accuracy,
        "validation_metrics": metrics,
        "trials": trials,
        "tuning_seconds": time.perf_counter() - tuning_started,
        "selected_validation_probabilities": selected["probabilities"],
    }


def select_phase(root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    config = read_yaml(root / "configs/trainable_fidelity_pilot.yaml")
    checksums = input_checksums(root)
    inputs = load_inputs(root)
    train_features = inputs["features"]["train"]
    validation_features = inputs["features"]["val"]
    train_labels = read_labels(
        root / "data/processed/herg_karim.csv", inputs["indices"]["train"]
    )
    validation_labels = read_labels(
        root / "data/processed/herg_karim.csv", inputs["indices"]["val"]
    )
    if set(np.unique(train_labels)) != {0, 1} or set(np.unique(validation_labels)) != {0, 1}:
        raise ValueError("Train and validation samples must contain both classes")

    circuits = build_trainable_fidelity_kernel(
        qubits=int(config["qubits"]),
        reps=int(config["feature_map"]["repetitions"]),
        entanglement=str(config["feature_map"]["entanglement"]),
    )
    optimizer = SLSQP(
        maxiter=int(config["optimizer"]["maxiter"]),
        ftol=float(config["optimizer"]["ftol"]),
        disp=False,
    )
    rng = np.random.default_rng(int(config["seed"]))
    initial_point = rng.uniform(-0.1, 0.1, size=circuits["training_parameter_count"])
    trainer = QuantumKernelTrainer(
        quantum_kernel=circuits["kernel"],
        loss=SVCLoss(C=1.0),
        optimizer=optimizer,
        initial_point=initial_point,
    )
    started = time.perf_counter()
    result = trainer.fit(train_features, train_labels)
    training_seconds = time.perf_counter() - started

    kernel_started = time.perf_counter()
    train_kernel = np.asarray(circuits["kernel"].evaluate(train_features), dtype=np.float64)
    validation_kernel = np.asarray(
        circuits["kernel"].evaluate(validation_features, train_features),
        dtype=np.float64,
    )
    validation_self = np.asarray(
        circuits["kernel"].evaluate(validation_features), dtype=np.float64
    )
    kernel_seconds = time.perf_counter() - kernel_started
    validate_square_kernel(train_kernel)
    validate_square_kernel(validation_self)

    tuned = tune_svc(
        config,
        train_kernel,
        validation_kernel,
        train_labels,
        validation_labels,
    )
    validation_probabilities = tuned.pop("selected_validation_probabilities")

    output_dir.mkdir(parents=True)
    (output_dir / "validation_predictions").mkdir(parents=True)
    (output_dir / "kernels").mkdir(parents=True)
    (output_dir / "models").mkdir(parents=True)
    save_predictions(
        output_dir / "validation_predictions/trainable_fidelity_svm.csv",
        inputs["lock"]["molecule_ids"]["val"],
        validation_labels,
        validation_probabilities,
    )
    np.savez_compressed(
        output_dir / "kernels/trainable_fidelity_train.npz",
        matrix=train_kernel,
        row_ids=np.asarray(inputs["lock"]["molecule_ids"]["train"], dtype=str),
        column_ids=np.asarray(inputs["lock"]["molecule_ids"]["train"], dtype=str),
    )
    np.savez_compressed(
        output_dir / "kernels/trainable_fidelity_validation.npz",
        matrix=validation_kernel,
        row_ids=np.asarray(inputs["lock"]["molecule_ids"]["val"], dtype=str),
        column_ids=np.asarray(inputs["lock"]["molecule_ids"]["train"], dtype=str),
        self_matrix=validation_self,
        self_ids=np.asarray(inputs["lock"]["molecule_ids"]["val"], dtype=str),
    )
    (output_dir / "circuit").mkdir(parents=True)
    with (output_dir / "circuit/trainable_fidelity.qpy").open("wb") as file:
        qpy.dump(circuits["circuit"], file)
    np.savez_compressed(
        output_dir / "models/trainable_fidelity_checkpoint.npz",
        initial_point=initial_point,
        optimal_point=np.asarray(result.optimal_point, dtype=np.float64),
        optimal_parameters=np.asarray(
            list(result.optimal_parameters.values()), dtype=np.float64
        ),
    )

    lock = {
        "status": "locked_for_test_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "trainable_fidelity_quantum_kernel",
        "optimizer": "SLSQP",
        "loss": "SVCLoss",
        "seed": int(config["seed"]),
        "molecule_ids": inputs["lock"]["molecule_ids"],
        "input_sample_sizes": {
            partition: len(inputs["lock"]["molecule_ids"][partition])
            for partition in ("train", "val", "test")
        },
        "feature_map": config["feature_map"],
        "circuit_properties": {
            key: circuits[key]
            for key in (
                "qubit_count",
                "depth",
                "gate_count",
                "two_qubit_gate_count",
                "feature_parameter_count",
                "training_parameter_count",
                "total_parameter_count",
            )
        },
        "threshold": float(tuned["threshold"]),
        "validation_metrics": tuned["validation_metrics"],
        "tuned_parameters": tuned["parameters"],
        "training": {
            "training_seconds": training_seconds,
            "objective_evaluations": int(result.optimizer_evals),
            "optimal_value": float(result.optimal_value),
            "initial_point": initial_point.tolist(),
            "optimal_point": np.asarray(result.optimal_point, dtype=np.float64).tolist(),
        },
        "kernel_construction_seconds": kernel_seconds,
        "tuning": tuned,
        "input_checksums": checksums,
    }
    lock["tuning"].pop("selected_validation_probabilities", None)
    write_json(output_dir / "locked_configuration.json", lock)

    manifest = {
        "status": "configuration_locked",
        "dataset": "herg_karim",
        "split": "scaffold_r0",
        "architecture": "trainable_fidelity_kernel",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_configuration_sha256": sha256(
            output_dir / "locked_configuration.json"
        ),
        "runtime": runtime_info(),
        "input_checksums": checksums,
        "training": lock["training"],
        "validation_metrics": tuned["validation_metrics"],
        "circuit_properties": lock["circuit_properties"],
    }
    write_json(output_dir / "experiment_manifest.json", manifest)
    print(
        f"Configuration locked; test labels/features are unused. "
        f"Validation AUROC={tuned['validation_metrics']['auroc']:.4f}, "
        f"training_s={training_seconds:.3f}"
    )


def evaluate_phase(root: Path, output_dir: Path) -> None:
    lock_path = output_dir / "locked_configuration.json"
    manifest_path = output_dir / "experiment_manifest.json"
    if not lock_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run --phase select before --phase evaluate")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "configuration_locked":
        raise ValueError("Experiment is not in the configuration-locked state")
    if manifest["locked_configuration_sha256"] != sha256(lock_path):
        raise ValueError("Locked configuration checksum mismatch")
    if input_checksums(root) != lock["input_checksums"]:
        raise ValueError("Input checksums changed since selection")

    inputs = load_inputs(root)
    test_labels = read_labels(
        root / "data/processed/herg_karim.csv", inputs["indices"]["test"]
    )
    if set(np.unique(test_labels)) != {0, 1}:
        raise ValueError("Test subset must contain both classes")

    checkpoint = np.load(
        output_dir / "models/trainable_fidelity_checkpoint.npz",
        allow_pickle=False,
    )
    optimal_point = checkpoint["optimal_point"]
    circuits = build_trainable_fidelity_kernel(
        qubits=int(lock["circuit_properties"]["qubit_count"]),
        reps=int(lock["feature_map"]["repetitions"]),
        entanglement=str(lock["feature_map"]["entanglement"]),
    )
    circuits["kernel"].assign_training_parameters(
        dict(zip(circuits["training_parameters"], optimal_point))
    )
    train_features = inputs["features"]["train"]
    test_features = inputs["features"]["test"]
    train_kernel = np.asarray(
        circuits["kernel"].evaluate(train_features), dtype=np.float64
    )
    test_kernel = np.asarray(
        circuits["kernel"].evaluate(test_features, train_features), dtype=np.float64
    )
    parameters = lock["tuned_parameters"]
    model = SVC(
        C=float(parameters["C"]),
        kernel="precomputed",
        class_weight=parameters["class_weight"],
        probability=True,
        random_state=int(lock["seed"]),
    )
    train_labels = read_labels(
        root / "data/processed/herg_karim.csv", inputs["indices"]["train"]
    )
    started = time.perf_counter()
    model.fit(train_kernel, train_labels)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    test_probabilities = model.predict_proba(test_kernel)[:, 1]
    test_prediction_seconds = time.perf_counter() - started
    test_metrics = calculate_pilot_metrics(
        test_labels, test_probabilities, float(lock["threshold"])
    )
    (output_dir / "test_predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    save_predictions(
        output_dir / "test_predictions/trainable_fidelity_svm.csv",
        lock["molecule_ids"]["test"],
        test_labels,
        test_probabilities,
    )
    np.savez_compressed(
        output_dir / "kernels/trainable_fidelity_test.npz",
        matrix=test_kernel,
        row_ids=np.asarray(lock["molecule_ids"]["test"], dtype=str),
        column_ids=np.asarray(lock["molecule_ids"]["train"], dtype=str),
    )
    joblib.dump(model, output_dir / "models/trainable_fidelity_svm.joblib")
    write_json(
        output_dir / "metrics/test_metrics.json",
        {
            "n_test": int(len(test_labels)),
            "positive_rate": float(test_labels.mean()),
            "threshold": float(lock["threshold"]),
            "test_metrics": test_metrics,
            "training_seconds": training_seconds,
            "test_prediction_seconds": test_prediction_seconds,
            "kernel_training_seconds": lock["training"]["training_seconds"],
            "kernel_construction_seconds": lock["kernel_construction_seconds"],
            "objective_evaluations": lock["training"]["objective_evaluations"],
        },
    )
    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["test_metrics"] = test_metrics
    manifest["test_prediction_seconds"] = test_prediction_seconds
    manifest["artifact_checksums"] = {
        path.relative_to(output_dir).as_posix(): sha256(path)
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    write_json(manifest_path, manifest)
    print(json.dumps(test_metrics, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.phase == "select":
        select_phase(ROOT, output_dir)
    else:
        evaluate_phase(ROOT, output_dir)


if __name__ == "__main__":
    main()
