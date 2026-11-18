"""Run matched VQC pilot comparisons for SLSQP, COBYLA, and SPSA."""

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

import numpy as np
import pandas as pd
import sklearn
import yaml
from qiskit import qpy
from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit_machine_learning.algorithms.classifiers import VQC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qktox.quantum_pilot import (
    apply_train_preprocessor,
    calculate_pilot_metrics,
    select_validation_threshold,
)
from qktox.vqc import build_vqc_circuits


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_labels(path: Path, indices: np.ndarray) -> np.ndarray:
    selected = sorted({int(index) for index in indices})
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


def load_inputs(root: Path, include_test: bool = False) -> dict[str, Any]:
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
    descriptors = pd.read_csv(feature_dir / "rdkit_descriptors.csv")
    names = json.loads((feature_dir / "feature_manifest.json").read_text(encoding="utf-8"))["descriptor_names"]
    combined = np.concatenate(
        (fingerprints.astype(np.float64), descriptors.loc[:, names].to_numpy(dtype=np.float64)),
        axis=1,
    )
    indices = {
        partition: np.asarray(
            [int(mol_id.rsplit("_", 1)[1]) for mol_id in lock["molecule_ids"][partition]],
            dtype=np.int64,
        )
        for partition in ("train", "val", "test")
    }
    feature_partitions = ("train", "val", "test") if include_test else ("train", "val")
    features = {
        partition: apply_train_preprocessor(combined[indices[partition]], preprocessing)
        for partition in feature_partitions
    }
    return {"lock": lock, "indices": indices, "features": features}


def build_optimizer(name: str, settings: dict[str, Any]) -> Any:
    if name == "SLSQP":
        return SLSQP(
            maxiter=int(settings["maxiter"]),
            ftol=float(settings["ftol"]),
            disp=False,
        )
    if name == "COBYLA":
        return COBYLA(
            maxiter=int(settings["maxiter"]),
            rhobeg=float(settings["rhobeg"]),
            tol=float(settings["tol"]),
            disp=False,
        )
    if name == "SPSA":
        return SPSA(
            maxiter=int(settings["maxiter"]),
            learning_rate=float(settings["learning_rate"]),
            perturbation=float(settings["perturbation"]),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def runtime_info() -> dict[str, Any]:
    names = {
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
            for key, distribution in names.items()
        },
    }


def probabilities(vqc: VQC, features: np.ndarray) -> np.ndarray:
    values = np.asarray(vqc.predict_proba(features), dtype=np.float64)
    if values.shape != (len(features), 2):
        raise ValueError(f"VQC probabilities have unexpected shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("VQC probabilities must be finite")
    return values[:, 1]


def save_predictions(path: Path, ids: list[str], labels: np.ndarray, probs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"mol_id": ids, "y_true": labels.astype(np.int8), "p_pred": probs}
    ).to_csv(path, index=False)


def run_optimizer(root: Path, optimizer_name: str) -> None:
    output_dir = root / "results/quantum_benchmarks/herg_karim" / f"vqc_{optimizer_name.lower()}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    config = read_yaml(root / "configs/vqc_optimizer_pilot.yaml")
    inputs = load_inputs(root)
    train_labels = read_labels(
        root / "data/processed/herg_karim.csv", inputs["indices"]["train"]
    )
    validation_labels = read_labels(
        root / "data/processed/herg_karim.csv", inputs["indices"]["val"]
    )
    if set(np.unique(train_labels)) != {0, 1} or set(np.unique(validation_labels)) != {0, 1}:
        raise ValueError("Training and validation samples must contain both classes")
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
        size=circuits["ansatz_parameter_count"],
    )
    history: list[dict[str, float | int]] = []

    def callback(*args: Any) -> None:
        weights = np.asarray(args[0], dtype=np.float64).reshape(-1)
        objective_value = args[1] if len(args) > 1 else np.nan
        objective_array = np.asarray(objective_value, dtype=np.float64).reshape(-1)
        objective = (
            float(objective_array.item())
            if objective_array.size == 1
            else float(objective_array.mean())
        )
        history.append({
            "objective": objective,
            "parameter_norm": float(np.linalg.norm(weights)),
            "evaluation": len(history) + 1,
        })

    optimizer = build_optimizer(optimizer_name, config["optimizers"][optimizer_name])
    from qiskit_algorithms.utils import algorithm_globals

    algorithm_globals.random_seed = int(config["seed"])
    vqc = VQC(
        num_qubits=int(config["qubits"]),
        feature_map=circuits["feature_map"],
        ansatz=circuits["ansatz"],
        loss=config["loss"],
        optimizer=optimizer,
        initial_point=initial_point,
        callback=callback,
    )
    started = time.perf_counter()
    vqc.fit(inputs["features"]["train"], train_labels)
    training_seconds = time.perf_counter() - started
    validation_probabilities = probabilities(
        vqc, inputs["features"]["val"]
    )
    threshold, threshold_score = select_validation_threshold(
        validation_labels, validation_probabilities
    )
    validation_metrics = calculate_pilot_metrics(
        validation_labels, validation_probabilities, threshold
    )

    output_dir.mkdir(parents=True)
    save_predictions(
        output_dir / f"validation_predictions/vqc_{optimizer_name.lower()}.csv",
        inputs["lock"]["molecule_ids"]["val"],
        validation_labels,
        validation_probabilities,
    )
    locked = {
        "status": "locked_for_test_evaluation",
        "architecture": "VQC",
        "optimizer": optimizer_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(config["seed"]),
        "molecule_ids": inputs["lock"]["molecule_ids"],
        "feature_map": config["feature_map"],
        "ansatz": config["ansatz"],
        "threshold": threshold,
        "validation_balanced_accuracy_at_threshold": threshold_score,
        "validation_metrics": validation_metrics,
        "circuit_properties": {
            key: circuits[key]
            for key in (
                "qubits",
                "depth",
                "gate_count",
                "two_qubit_gate_count",
                "feature_map_parameter_count",
                "ansatz_parameter_count",
                "total_parameter_count",
            )
        },
        "training_seconds": training_seconds,
        "optimization_history": history,
        "final_parameters": np.asarray(vqc.weights, dtype=np.float64).tolist(),
    }
    write_json(output_dir / "locked_configuration.json", locked)
    lock_checksum = sha256(output_dir / "locked_configuration.json")

    inputs = load_inputs(root, include_test=True)
    test_labels = read_labels(
        root / "data/processed/herg_karim.csv", inputs["indices"]["test"]
    )
    started = time.perf_counter()
    test_probabilities = probabilities(vqc, inputs["features"]["test"])
    test_prediction_seconds = time.perf_counter() - started
    test_metrics = calculate_pilot_metrics(
        test_labels, test_probabilities, threshold
    )
    save_predictions(
        output_dir / f"test_predictions/vqc_{optimizer_name.lower()}.csv",
        inputs["lock"]["molecule_ids"]["test"],
        test_labels,
        test_probabilities,
    )
    (output_dir / "models").mkdir(parents=True)
    np.savez_compressed(
        output_dir / f"models/vqc_{optimizer_name.lower()}_checkpoint.npz",
        weights=np.asarray(vqc.weights, dtype=np.float64),
        initial_point=initial_point,
    )
    (output_dir / "circuit").mkdir(parents=True)
    with (output_dir / f"circuit/vqc_{optimizer_name.lower()}.qpy").open("wb") as file:
        qpy.dump(circuits["circuit"], file)
    (output_dir / "metrics").mkdir(parents=True)
    write_json(
        output_dir / "metrics/test_metrics.json",
        {
            "n_test": int(len(test_labels)),
            "positive_rate": float(test_labels.mean()),
            "threshold": threshold,
            "test_metrics": test_metrics,
            "training_seconds": training_seconds,
            "test_prediction_seconds": test_prediction_seconds,
            "objective_evaluations": len(history),
            "optimization_history": history,
        },
    )
    write_json(
        output_dir / "experiment_manifest.json",
        {
            "status": "completed",
            "dataset": "herg_karim",
            "split": "scaffold_r0",
            "architecture": "VQC",
            "optimizer": optimizer_name,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "locked_configuration_sha256": lock_checksum,
            "molecule_ids": inputs["lock"]["molecule_ids"],
            "circuit_properties": {
                key: circuits[key]
                for key in (
                    "qubits",
                    "depth",
                    "gate_count",
                    "two_qubit_gate_count",
                    "feature_map_parameter_count",
                    "ansatz_parameter_count",
                    "total_parameter_count",
                )
            },
            "threshold": threshold,
            "validation_balanced_accuracy_at_threshold": threshold_score,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "training_seconds": training_seconds,
            "test_prediction_seconds": test_prediction_seconds,
            "objective_evaluations": len(history),
            "optimization_history": history,
            "runtime": runtime_info(),
        },
    )
    print(
        f"{optimizer_name}: "
        f"validation AUROC={validation_metrics['auroc']:.4f}, "
        f"test AUROC={test_metrics['auroc']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimizer",
        choices=("SLSQP", "COBYLA", "SPSA"),
        action="append",
        required=True,
    )
    args = parser.parse_args()
    for optimizer_name in args.optimizer:
        run_optimizer(ROOT, optimizer_name)


if __name__ == "__main__":
    main()
