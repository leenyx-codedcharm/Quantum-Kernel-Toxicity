"""Run the matched four-qubit VQC SLSQP pilot in the quantum environment."""

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
from qiskit_algorithms.optimizers import SLSQP
from qiskit_machine_learning.algorithms.classifiers import VQC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qktox.quantum_pilot import (
    apply_train_preprocessor,
    calculate_pilot_metrics,
    select_validation_threshold,
)
from qktox.vqc import build_vqc_circuits

DEFAULT_OUTPUT = Path("results/quantum_benchmarks/herg_karim/vqc_slsqp")


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


def read_indices(path: Path, indices: np.ndarray) -> np.ndarray:
    selected = sorted({int(index) for index in indices})
    if len(selected) != len(indices):
        raise ValueError("Requested row positions must be unique")
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


def load_preprocessed_inputs(root: Path) -> dict[str, Any]:
    lock_path = root / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/locked_configuration.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    preprocessing_path = (
        root
        / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/preprocessing/scaler_pca_parameters.npz"
    )
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
    descriptor_values = descriptors.loc[:, descriptor_names].to_numpy(dtype=np.float64)
    combined = np.concatenate((fingerprints.astype(np.float64), descriptor_values), axis=1)

    def indices_for(partition: str) -> np.ndarray:
        return np.asarray(
            [int(molecule_id.rsplit("_", 1)[1]) for molecule_id in lock["molecule_ids"][partition]],
            dtype=np.int64,
        )

    indices = {partition: indices_for(partition) for partition in ("train", "val", "test")}
    for partition, values in indices.items():
        molecule_ids = [f"herg_karim_{int(index):06d}" for index in values]
        if molecule_ids != lock["molecule_ids"][partition]:
            raise ValueError(f"{partition}: molecule IDs do not map back to frozen pilot positions")
    if not np.array_equal(fingerprint_ids, np.asarray([
        f"herg_karim_{index:06d}" for index in range(len(fingerprint_ids))
    ])):
        raise ValueError("Fingerprint rows do not align with processed row positions")
    if not np.array_equal(descriptor_ids, fingerprint_ids):
        raise ValueError("Descriptor and fingerprint molecule IDs differ")

    transformed = {
        partition: apply_train_preprocessor(combined[values], preprocessing)
        for partition, values in indices.items()
    }
    return {
        "lock": lock,
        "indices": indices,
        "features": transformed,
        "combined": combined,
    }


def build_optimizer(config: dict[str, Any]) -> SLSQP:
    settings = config["optimizer"]
    return SLSQP(
        maxiter=int(settings["maxiter"]),
        ftol=float(settings["ftol"]),
        disp=False,
    )


def probabilities(vqc: VQC, features: np.ndarray) -> np.ndarray:
    values = np.asarray(vqc.predict_proba(features), dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(features) or values.shape[1] != 2:
        raise ValueError(f"VQC probabilities have unexpected shape {values.shape}")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("VQC probabilities must be finite values in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("VQC probability rows do not sum to one")
    return values[:, 1]


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


def input_checksums(root: Path) -> dict[str, str]:
    paths = {
        "frozen_splits": root / "data/splits/herg_karim_scaffold.json",
        "processed_dataset": root / "data/processed/herg_karim.csv",
        "fingerprints": root / "data/processed/features/hERG_Karim/ecfp4_2048.npz",
        "descriptors": root / "data/processed/features/hERG_Karim/rdkit_descriptors.csv",
        "source_fidelity_lock": root / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/locked_configuration.json",
        "source_fidelity_preprocessing": root / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/preprocessing/scaler_pca_parameters.npz",
        "config": root / "configs/vqc_pilot.yaml",
        "script": root / "scripts/12_vqc_pilot.py",
        "module": root / "src/qktox/vqc.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing VQC inputs: " + ", ".join(missing))
    return {name: sha256(path) for name, path in paths.items()}


def fit_vqc(
    config: dict[str, Any],
    train_features: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[VQC, dict[str, Any]]:
    circuits = build_vqc_circuits(
        qubits=int(config["qubits"]),
        feature_repetitions=int(config["feature_map"]["repetitions"]),
        feature_entanglement=config["feature_map"]["entanglement"],
        ansatz_repetitions=int(config["ansatz"]["repetitions"]),
        ansatz_entanglement=config["ansatz"]["entanglement"],
    )
    rng = np.random.default_rng(int(config["seed"]))
    initial_point = rng.uniform(
        -0.1,
        0.1,
        size=int(circuits["ansatz_parameter_count"]),
    )
    history: list[dict[str, Any]] = []

    def callback(weights: np.ndarray, objective_value: float) -> None:
        history.append({
            "objective": float(objective_value),
            "parameter_norm": float(np.linalg.norm(weights)),
        })

    optimizer = build_optimizer(config)
    started = time.perf_counter()
    vqc = VQC(
        num_qubits=int(config["qubits"]),
        feature_map=circuits["feature_map"],
        ansatz=circuits["ansatz"],
        loss=config["loss"],
        optimizer=optimizer,
        initial_point=initial_point,
        callback=callback,
    )
    vqc.fit(train_features, train_labels)
    training_seconds = time.perf_counter() - started
    record = {
        "training_seconds": training_seconds,
        "objective_evaluations": len(history),
        "optimization_history": history,
        "initial_point": initial_point.tolist(),
        "final_parameters": np.asarray(vqc.weights, dtype=np.float64).tolist(),
    }
    return vqc, record


def save_predictions(
    path: Path,
    molecule_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    frame = pd.DataFrame({
        "mol_id": molecule_ids,
        "y_true": labels.astype(np.int8),
        "p_pred": probabilities,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def select_phase(root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    config = read_yaml(root / "configs/vqc_pilot.yaml")
    inputs = load_preprocessed_inputs(root)
    lock = inputs["lock"]
    train_indices = inputs["indices"]["train"]
    validation_indices = inputs["indices"]["val"]
    processed = root / "data/processed/herg_karim.csv"
    train_labels = read_indices(processed, train_indices)
    validation_labels = read_indices(processed, validation_indices)
    if set(np.unique(train_labels)) != {0, 1} or set(np.unique(validation_labels)) != {0, 1}:
        raise ValueError("VQC train and validation samples must contain both classes")

    vqc, training = fit_vqc(
        config,
        inputs["features"]["train"],
        train_labels,
    )
    predict_started = time.perf_counter()
    validation_probabilities = probabilities(vqc, inputs["features"]["val"])
    validation_prediction_seconds = time.perf_counter() - predict_started
    threshold, validation_balanced_accuracy = select_validation_threshold(
        validation_labels,
        validation_probabilities,
    )
    validation_metrics = calculate_pilot_metrics(
        validation_labels,
        validation_probabilities,
        threshold,
    )
    output_dir.mkdir(parents=True)

    train_ids = lock["molecule_ids"]["train"]
    validation_ids = lock["molecule_ids"]["val"]
    test_ids = lock["molecule_ids"]["test"]
    save_predictions(
        output_dir / "validation_predictions/vqc_slsqp.csv",
        validation_ids,
        validation_labels,
        validation_probabilities,
    )
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "models/vqc_slsqp_checkpoint.npz",
        weights=np.asarray(vqc.weights, dtype=np.float64),
        initial_point=np.asarray(training["initial_point"], dtype=np.float64),
    )
    circuits = build_vqc_circuits(
        qubits=int(config["qubits"]),
        feature_repetitions=int(config["feature_map"]["repetitions"]),
        feature_entanglement=config["feature_map"]["entanglement"],
        ansatz_repetitions=int(config["ansatz"]["repetitions"]),
        ansatz_entanglement=config["ansatz"]["entanglement"],
    )
    circuit_properties = {
        "qubit_count": circuits["qubits"],
        "depth": circuits["depth"],
        "gate_count": circuits["gate_count"],
        "two_qubit_gate_count": circuits["two_qubit_gate_count"],
        "feature_map_parameter_count": circuits["feature_map_parameter_count"],
        "ansatz_parameter_count": circuits["ansatz_parameter_count"],
        "total_parameter_count": circuits["total_parameter_count"],
    }
    from qiskit import qpy

    (output_dir / "circuit").mkdir(parents=True, exist_ok=True)
    with (output_dir / "circuit/vqc_slsqp.qpy").open("wb") as file:
        qpy.dump(circuits["circuit"], file)

    locked = {
        "status": "locked_for_test_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "variational_quantum_classifier",
        "optimizer": "SLSQP",
        "seed": int(config["seed"]),
        "molecule_ids": {"train": train_ids, "val": validation_ids, "test": test_ids},
        "input_sample_sizes": {
            "train": len(train_ids),
            "val": len(validation_ids),
            "test": len(test_ids),
        },
        "feature_map": config["feature_map"],
        "ansatz": config["ansatz"],
        "circuit_properties": circuit_properties,
        "loss": config["loss"],
        "threshold": float(threshold),
        "validation_balanced_accuracy_at_threshold": float(validation_balanced_accuracy),
        "validation_metrics": validation_metrics,
        "training": training,
        "validation_prediction_seconds": validation_prediction_seconds,
        "input_checksums": input_checksums(root),
        "source_fidelity_pilot": str(
            root / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final"
        ),
    }
    write_json(output_dir / "locked_configuration.json", locked)

    manifest = {
        "status": "configuration_locked",
        "dataset": "herg_karim",
        "split": "scaffold_r0",
        "architecture": "VQC",
        "optimizer": "SLSQP",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_configuration_sha256": sha256(output_dir / "locked_configuration.json"),
        "runtime": runtime_info(),
        "input_checksums": locked["input_checksums"],
        "selected_threshold": float(threshold),
        "validation_metrics": validation_metrics,
        "training": training,
        "circuit_properties": circuit_properties,
    }
    write_json(output_dir / "experiment_manifest.json", manifest)
    print(
        f"Configuration locked; test labels/features are unused. "
        f"Validation AUROC={validation_metrics['auroc']:.4f}, "
        f"threshold={threshold:.6g}, "
        f"training_s={training['training_seconds']:.3f}"
    )


def evaluate_phase(root: Path, output_dir: Path) -> None:
    lock_path = output_dir / "locked_configuration.json"
    manifest_path = output_dir / "experiment_manifest.json"
    if not lock_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run --phase select before --phase evaluate")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "configuration_locked"
        or lock.get("status") != "locked_for_test_evaluation"
    ):
        raise ValueError("This VQC experiment is not in the lockable selection state")
    if manifest["locked_configuration_sha256"] != sha256(lock_path):
        raise ValueError("Locked configuration checksum mismatch")
    if input_checksums(root) != lock["input_checksums"]:
        raise ValueError("Input checksums changed since VQC selection")

    config = read_yaml(root / "configs/vqc_pilot.yaml")
    inputs = load_preprocessed_inputs(root)
    test_indices = inputs["indices"]["test"]
    test_labels = read_indices(root / "data/processed/herg_karim.csv", test_indices)
    if set(np.unique(test_labels)) != {0, 1}:
        raise ValueError("Test subset must contain both classes")
    checkpoint = np.load(
        output_dir / "models/vqc_slsqp_checkpoint.npz",
        allow_pickle=False,
    )
    config = read_yaml(root / "configs/vqc_pilot.yaml")
    circuits = build_vqc_circuits(
        qubits=int(config["qubits"]),
        feature_repetitions=int(config["feature_map"]["repetitions"]),
        feature_entanglement=config["feature_map"]["entanglement"],
        ansatz_repetitions=int(config["ansatz"]["repetitions"]),
        ansatz_entanglement=config["ansatz"]["entanglement"],
    )
    from qiskit_algorithms.optimizers import SLSQP

    vqc = VQC(
        num_qubits=int(config["qubits"]),
        feature_map=circuits["feature_map"],
        ansatz=circuits["ansatz"],
        loss=config["loss"],
        optimizer=build_optimizer(config),
        initial_point=np.asarray(checkpoint["initial_point"], dtype=np.float64),
    )
    vqc._fit_result = None
    vqc.weights = np.asarray(checkpoint["weights"], dtype=np.float64)
    predict_started = time.perf_counter()
    test_probabilities = probabilities(vqc, inputs["features"]["test"])
    test_prediction_seconds = time.perf_counter() - predict_started
    threshold = float(lock["threshold"])
    test_metrics = calculate_pilot_metrics(test_labels, test_probabilities, threshold)
    test_ids = lock["molecule_ids"]["test"]
    save_predictions(
        output_dir / "test_predictions/vqc_slsqp.csv",
        test_ids,
        test_labels,
        test_probabilities,
    )
    write_json(
        output_dir / "metrics/test_metrics.json",
        {
            "n_test": int(len(test_labels)),
            "positive_rate": float(test_labels.mean()),
            "threshold": threshold,
            "test_metrics": test_metrics,
            "training_seconds": lock["training"]["training_seconds"],
            "validation_prediction_seconds": lock["validation_prediction_seconds"],
            "test_prediction_seconds": test_prediction_seconds,
            "objective_evaluations": lock["training"]["objective_evaluations"],
        },
    )
    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["test_metrics"] = test_metrics
    manifest["runtime"] = runtime_info()
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
