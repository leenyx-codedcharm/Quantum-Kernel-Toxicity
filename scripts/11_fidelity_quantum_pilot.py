"""Run the locked, two-phase hERG_Karim fidelity-kernel pilot."""

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
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qktox.kernels import tanimoto_kernel_matrix
from qktox.quantum_pilot import (
    apply_train_preprocessor,
    calculate_pilot_metrics,
    fit_train_preprocessor,
    sample_partition_indices,
    select_validation_threshold,
    validate_cross_kernel,
    validate_square_kernel,
)

DEFAULT_OUTPUT = Path("results/quantum_fidelity/fidelity_statevector_pilot_r0_final")
PARTITIONS = ("train", "val", "test")
MODEL_NAMES = ("fidelity_svc", "tanimoto_svm")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _read_config(root: Path) -> tuple[dict[str, Any], Path]:
    config_path = root / "configs" / "quantum_fidelity_pilot.yaml"
    config = _read_yaml(config_path)
    if config.get("dataset_key") != "herg_karim":
        raise ValueError("The fidelity pilot is configured for herg_karim only")
    return config, config_path


def _input_paths(root: Path, config_path: Path, script_path: Path) -> dict[str, Path]:
    return {
        "processed_dataset": root / "data" / "processed" / "herg_karim.csv",
        "frozen_splits": root / "data" / "splits" / "herg_karim_scaffold.json",
        "fingerprints": root / "data" / "processed" / "features" / "hERG_Karim" / "ecfp4_2048.npz",
        "descriptors": root / "data" / "processed" / "features" / "hERG_Karim" / "rdkit_descriptors.csv",
        "feature_manifest": root / "data" / "processed" / "features" / "hERG_Karim" / "feature_manifest.json",
        "molecule_metadata": root / "data" / "processed" / "features" / "hERG_Karim" / "molecule_metadata.csv",
        "experiment_config": config_path,
        "quantum_requirements": root / "requirements-quantum.txt",
        "pilot_script": script_path,
        "pilot_module": root / "src" / "qktox" / "quantum_pilot.py",
        "kernel_helpers": root / "src" / "qktox" / "kernels.py",
    }


def _input_checksums(root: Path, config_path: Path, script_path: Path) -> dict[str, str]:
    paths = _input_paths(root, config_path, script_path)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required pilot inputs are missing: " + ", ".join(missing))
    return {name: _sha256(path) for name, path in paths.items()}


def _load_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / config["feature_representation"]["descriptor_manifest"]
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor_names = feature_manifest["descriptor_names"]
    fingerprint_path = root / config["feature_representation"]["fingerprint_file"]
    descriptor_path = root / config["feature_representation"]["descriptor_file"]
    metadata_path = root / "data" / "processed" / "features" / "hERG_Karim" / "molecule_metadata.csv"
    split_path = root / "data" / "splits" / "herg_karim_scaffold.json"
    processed_path = root / "data" / "processed" / "herg_karim.csv"

    with np.load(fingerprint_path, allow_pickle=False) as archive:
        fingerprints = archive["fingerprints"]
        fingerprint_ids = archive["mol_ids"].astype(str)
    descriptors_frame = pd.read_csv(
        descriptor_path,
        usecols=["mol_id", *descriptor_names],
        dtype={"mol_id": str},
    )
    metadata = pd.read_csv(
        metadata_path,
        usecols=["mol_id", "split_assignment"],
        dtype={"mol_id": str},
    )
    metadata_ids = metadata["mol_id"].to_numpy()
    descriptor_ids = descriptors_frame["mol_id"].astype(str).to_numpy()
    expected_ids = np.asarray([f"herg_karim_{index:06d}" for index in range(len(metadata_ids))])
    if not (
        np.array_equal(metadata_ids, expected_ids)
        and np.array_equal(fingerprint_ids, expected_ids)
        and np.array_equal(descriptor_ids, expected_ids)
    ):
        raise ValueError("Molecule IDs do not align with frozen processed-row positions")
    descriptors = descriptors_frame.loc[:, descriptor_names].to_numpy(dtype=np.float64)
    if fingerprints.ndim != 2 or fingerprints.shape[0] != len(expected_ids):
        raise ValueError("Fingerprint rows do not match the processed dataset")
    if fingerprints.shape[1] != int(config["feature_representation"]["fingerprint_bits"]):
        raise ValueError("Fingerprint dimension differs from the pilot configuration")
    fingerprint_config = feature_manifest["fingerprint"]
    if (
        fingerprint_config["n_bits"] != fingerprints.shape[1]
        or fingerprint_config["radius"]
        != int(config["feature_representation"]["fingerprint_radius"])
        or fingerprint_config["use_chirality"]
        != bool(config["feature_representation"]["use_chirality"])
    ):
        raise ValueError("Fingerprint input differs from its recorded feature manifest")
    if not np.isin(fingerprints, (0, 1)).all():
        raise ValueError("ECFP4 fingerprints must contain only binary values")
    if not np.isfinite(descriptors).all():
        raise ValueError("RDKit descriptors contain non-finite values")

    splits = json.loads(split_path.read_text(encoding="utf-8-sig"))
    if set(splits) != {f"scaffold_r{repeat}" for repeat in range(5)}:
        raise ValueError("Frozen split file does not contain exactly scaffold_r0-r4")
    split = splits[config["split_name"]]
    if len(split.get("test_novelty", [])) != len(split["test"]):
        raise ValueError("Frozen test novelty values are not aligned with test row positions")
    partition_sets = {part: set(split[part]) for part in PARTITIONS}
    if any(len(partition_sets[part]) != len(split[part]) for part in PARTITIONS):
        raise ValueError("Frozen split contains duplicate row positions")
    if any(
        partition_sets[left] & partition_sets[right]
        for i, left in enumerate(PARTITIONS)
        for right in PARTITIONS[i + 1 :]
    ):
        raise ValueError("Frozen train, validation, and test partitions overlap")
    if set.union(*partition_sets.values()) != set(range(len(expected_ids))):
        raise ValueError("Frozen split does not partition the processed rows exactly")
    for partition in PARTITIONS:
        for index in split[partition]:
            assignment = json.loads(metadata.iloc[index]["split_assignment"])
            if assignment.get(config["split_name"]) != partition:
                raise ValueError(
                    f"{expected_ids[index]}: feature metadata split assignment differs from frozen split"
                )
    return {
        "fingerprints": fingerprints,
        "descriptors": descriptors,
        "descriptor_names": descriptor_names,
        "feature_manifest": feature_manifest,
        "molecule_ids": expected_ids,
        "split": split,
        "processed_path": processed_path,
    }


def _sample_rows(config: dict[str, Any], split: dict[str, Any]) -> dict[str, np.ndarray]:
    seed = int(config["seed"])
    sizes = config["sample_sizes"]
    return {
        part: sample_partition_indices(
            split[part],
            int(sizes[part]),
            seed + offset,
        )
        for offset, part in enumerate(PARTITIONS)
    }


def _read_labels(path: Path, indices: np.ndarray) -> np.ndarray:
    """Read only the requested CSV rows, so selection cannot inspect test labels."""
    selected = sorted({int(index) for index in indices})
    if len(selected) != len(indices):
        raise ValueError("Requested label row positions must be unique")
    selected_set = set(selected)
    labels = pd.read_csv(
        path,
        usecols=["label"],
        skiprows=lambda line_number: line_number != 0 and line_number - 1 not in selected_set,
    )["label"]
    if len(labels) != len(selected):
        raise ValueError("Could not read all requested label rows")
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.isna().any() or not set(numeric.unique()).issubset({0, 1}):
        raise ValueError("Requested labels must be binary 0/1 values")
    return numeric.to_numpy(dtype=np.int8)


def _ids(indices: np.ndarray) -> list[str]:
    return [f"herg_karim_{int(index):06d}" for index in indices]


def _fit_parameters(config: dict[str, Any], C: float, class_weight: str | None) -> SVC:
    return SVC(
        C=float(C),
        kernel="precomputed",
        class_weight=class_weight,
        probability=bool(config["tuning"]["probability"]),
        random_state=int(config["seed"]),
        cache_size=512,
    )


def _save_kernel(
    path: Path,
    matrix: np.ndarray,
    row_ids: list[str],
    column_ids: list[str],
    self_matrix: np.ndarray | None = None,
) -> None:
    payload: dict[str, Any] = {
        "matrix": np.asarray(matrix, dtype=np.float64),
        "row_ids": np.asarray(row_ids, dtype=str),
        "column_ids": np.asarray(column_ids, dtype=str),
    }
    if self_matrix is not None:
        payload["self_matrix"] = np.asarray(self_matrix, dtype=np.float64)
        payload["self_ids"] = np.asarray(row_ids, dtype=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _load_kernel(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return archive["matrix"], archive["row_ids"].astype(str), archive["column_ids"].astype(str)


def _save_predictions(
    path: Path,
    molecule_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        {"mol_id": molecule_ids, "y_true": labels, "p_pred": probabilities}
    )
    if frame.columns.tolist() != ["mol_id", "y_true", "p_pred"]:
        raise ValueError("Prediction output columns differ from the shared schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _kernel_diagnostics(
    train: np.ndarray,
    validation: np.ndarray,
    validation_self: np.ndarray,
) -> dict[str, Any]:
    validate_cross_kernel(validation)
    return {
        "train": validate_square_kernel(train),
        "validation_self": validate_square_kernel(validation_self),
        "validation_cross_finite": bool(np.isfinite(validation).all()),
    }


def _tune_model(
    config: dict[str, Any],
    train_kernel: np.ndarray,
    validation_kernel: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
) -> dict[str, Any]:
    trials = []
    selected: dict[str, Any] | None = None
    tuning_started = time.perf_counter()
    for C in config["tuning"]["C_values"]:
        for class_weight in config["tuning"]["class_weights"]:
            model = _fit_parameters(config, C, class_weight)
            fit_started = time.perf_counter()
            model.fit(train_kernel, train_labels)
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            probabilities = model.predict_proba(validation_kernel)[:, 1]
            predict_seconds = time.perf_counter() - predict_started
            auroc = float(roc_auc_score(validation_labels, probabilities))
            trial = {
                "C": float(C),
                "class_weight": class_weight,
                "validation_auroc": auroc,
                "fit_seconds": fit_seconds,
                "validation_prediction_seconds": predict_seconds,
            }
            trials.append(trial)
            if selected is None or auroc > selected["validation_auroc"]:
                selected = {
                    **trial,
                    "probabilities": probabilities,
                }

    if selected is None:
        raise RuntimeError("The configured SVM tuning grid produced no candidates")
    threshold, validation_threshold_score = select_validation_threshold(
        validation_labels,
        selected["probabilities"],
    )
    selected_metrics = calculate_pilot_metrics(
        validation_labels,
        selected["probabilities"],
        threshold,
    )
    return {
        "parameters": {
            "C": selected["C"],
            "class_weight": selected["class_weight"],
            "probability": bool(config["tuning"]["probability"]),
            "random_state": int(config["seed"]),
        },
        "threshold": threshold,
        "validation_balanced_accuracy_at_threshold": validation_threshold_score,
        "validation_metrics": selected_metrics,
        "trials": trials,
        "tuning_seconds": time.perf_counter() - tuning_started,
        "selected_validation_probabilities": selected["probabilities"],
    }


def _quantum_kernel(config: dict[str, Any]) -> tuple[Any, Any]:
    try:
        from qiskit.circuit.library import zz_feature_map
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    except ImportError as exc:
        raise RuntimeError(
            "Quantum dependencies are unavailable; install requirements-quantum.txt "
            "into .venv-quantum"
        ) from exc

    quantum = config["quantum_kernel"]
    feature_map = zz_feature_map(
        feature_dimension=int(quantum["qubits"]),
        reps=int(quantum["repetitions"]),
        entanglement=str(quantum["entanglement"]),
    )
    if feature_map.num_qubits != int(quantum["qubits"]):
        raise ValueError("Generated ZZ feature map has an unexpected qubit count")
    kernel = FidelityStatevectorKernel(
        feature_map=feature_map,
        shots=quantum["shots"],
        enforce_psd=bool(quantum["enforce_psd"]),
    )
    return kernel, feature_map


def _write_circuit(feature_map: Any, path: Path) -> None:
    from qiskit import qpy

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        qpy.dump(feature_map, file)


def _save_preprocessing(path: Path, parameters: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **parameters)


def _load_preprocessing(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _runtime_info() -> dict[str, Any]:
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "scipy": "scipy",
        "qiskit": "qiskit",
        "qiskit_machine_learning": "qiskit-machine-learning",
    }
    versions = {
        key: importlib.metadata.version(distribution)
        for key, distribution in packages.items()
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": versions,
        "sklearn_runtime_version": sklearn.__version__,
    }


def _selection_phase(root: Path, output_dir: Path, script_path: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing experiment directory: {output_dir}"
        )
    config, config_path = _read_config(root)
    checksums = _input_checksums(root, config_path, script_path)
    inputs = _load_inputs(root, config)
    samples = _sample_rows(config, inputs["split"])
    train_indices = samples["train"]
    validation_indices = samples["val"]
    train_labels = _read_labels(inputs["processed_path"], train_indices)
    validation_labels = _read_labels(inputs["processed_path"], validation_indices)
    if set(np.unique(train_labels)) != {0, 1} or set(np.unique(validation_labels)) != {0, 1}:
        raise ValueError("Sampled training and validation rows must each contain both classes")

    fingerprints = inputs["fingerprints"]
    descriptors = inputs["descriptors"]
    combined = np.concatenate((fingerprints.astype(np.float64), descriptors), axis=1)
    preprocessing = fit_train_preprocessor(
        combined[train_indices],
        n_components=int(config["feature_representation"]["pca_components"]),
    )
    quantum_train = apply_train_preprocessor(combined[train_indices], preprocessing)
    quantum_validation = apply_train_preprocessor(combined[validation_indices], preprocessing)
    train_ids = _ids(train_indices)
    validation_ids = _ids(validation_indices)
    # Test IDs are fixed from frozen membership before locking, without reading test labels/features.
    test_ids = _ids(samples["test"])

    kernel, feature_map = _quantum_kernel(config)
    quantum_started = time.perf_counter()
    quantum_train_kernel = kernel.evaluate(quantum_train)
    quantum_validation_kernel = kernel.evaluate(quantum_validation, quantum_train)
    quantum_validation_self = kernel.evaluate(quantum_validation)
    quantum_kernel_seconds = time.perf_counter() - quantum_started
    quantum_diagnostics = _kernel_diagnostics(
        quantum_train_kernel,
        quantum_validation_kernel,
        quantum_validation_self,
    )

    tanimoto_started = time.perf_counter()
    tanimoto_train_kernel = tanimoto_kernel_matrix(
        fingerprints[train_indices],
        fingerprints[train_indices],
    )
    tanimoto_validation_kernel = tanimoto_kernel_matrix(
        fingerprints[validation_indices],
        fingerprints[train_indices],
    )
    tanimoto_validation_self = tanimoto_kernel_matrix(
        fingerprints[validation_indices],
        fingerprints[validation_indices],
    )
    tanimoto_kernel_seconds = time.perf_counter() - tanimoto_started
    tanimoto_diagnostics = _kernel_diagnostics(
        tanimoto_train_kernel,
        tanimoto_validation_kernel,
        tanimoto_validation_self,
    )

    output_dir.mkdir(parents=True)
    kernel_dir = output_dir / "kernels"
    _save_kernel(kernel_dir / "fidelity_train.npz", quantum_train_kernel, train_ids, train_ids)
    _save_kernel(
        kernel_dir / "fidelity_validation.npz",
        quantum_validation_kernel,
        validation_ids,
        train_ids,
        quantum_validation_self,
    )
    _save_kernel(kernel_dir / "tanimoto_train.npz", tanimoto_train_kernel, train_ids, train_ids)
    _save_kernel(
        kernel_dir / "tanimoto_validation.npz",
        tanimoto_validation_kernel,
        validation_ids,
        train_ids,
        tanimoto_validation_self,
    )
    preprocessing_path = output_dir / "preprocessing" / "scaler_pca_parameters.npz"
    _save_preprocessing(preprocessing_path, preprocessing)
    circuit_path = output_dir / "circuit" / "zz_feature_map.qpy"
    _write_circuit(feature_map, circuit_path)

    models = {}
    for model_name, train_matrix, validation_matrix, kernel_seconds in (
        (
            "fidelity_svc",
            quantum_train_kernel,
            quantum_validation_kernel,
            quantum_kernel_seconds,
        ),
        (
            "tanimoto_svm",
            tanimoto_train_kernel,
            tanimoto_validation_kernel,
            tanimoto_kernel_seconds,
        ),
    ):
        tuned = _tune_model(
            config,
            train_matrix,
            validation_matrix,
            train_labels,
            validation_labels,
        )
        validation_probabilities = tuned.pop("selected_validation_probabilities")
        _save_predictions(
            output_dir / "validation_predictions" / f"{model_name}.csv",
            validation_ids,
            validation_labels,
            validation_probabilities,
        )
        models[model_name] = {
            **tuned,
            "kernel_construction_seconds_train_validation": kernel_seconds,
        }

    selection_artifact_paths = [
        output_dir / "circuit" / "zz_feature_map.qpy",
        output_dir / "preprocessing" / "scaler_pca_parameters.npz",
        output_dir / "kernels" / "fidelity_train.npz",
        output_dir / "kernels" / "fidelity_validation.npz",
        output_dir / "kernels" / "tanimoto_train.npz",
        output_dir / "kernels" / "tanimoto_validation.npz",
        *[
            output_dir / "validation_predictions" / f"{name}.csv"
            for name in MODEL_NAMES
        ],
    ]
    feature_map_config = {
        "api": "qiskit.circuit.library.zz_feature_map",
        "name": "ZZFeatureMap",
        "qubits": int(config["quantum_kernel"]["qubits"]),
        "repetitions": int(config["quantum_kernel"]["repetitions"]),
        "entanglement": config["quantum_kernel"]["entanglement"],
        "parameter_count": int(len(feature_map.parameters)),
        "depth": int(feature_map.depth()),
        "simulation": "exact statevector",
        "shots": None,
    }
    preprocessing_record = {
        "input_representation": config["feature_representation"]["quantum"],
        "fingerprint_bits": int(fingerprints.shape[1]),
        "fingerprint_radius": int(config["feature_representation"]["fingerprint_radius"]),
        "descriptor_names": inputs["descriptor_names"],
        "input_dimensions": int(combined.shape[1]),
        "output_dimensions": int(config["feature_representation"]["pca_components"]),
        "fit_molecule_ids": train_ids,
        "fit_partition": "sampled scaffold_r0 train only",
        "scaler": "StandardScaler",
        "pca": {
            "components": int(config["feature_representation"]["pca_components"]),
            "solver": config["feature_representation"]["pca_solver"],
            "explained_variance": preprocessing["pca_explained_variance"].tolist(),
            "explained_variance_ratio": preprocessing["pca_explained_variance_ratio"].tolist(),
            "parameter_file": "preprocessing/scaler_pca_parameters.npz",
        },
    }
    lock = {
        "status": "locked_for_test_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(config["seed"]),
        "sample_sizes": {part: len(samples[part]) for part in PARTITIONS},
        "sample_seeds": {part: int(config["seed"]) + offset for offset, part in enumerate(PARTITIONS)},
        "molecule_ids": {
            "train": train_ids,
            "val": validation_ids,
            "test": test_ids,
        },
        "preprocessing": preprocessing_record,
        "quantum_feature_map": feature_map_config,
        "tuning": {
            "selection_metric": config["tuning"]["selection_metric"],
            "threshold_metric": config["tuning"]["threshold_metric"],
            "candidates_per_model": len(config["tuning"]["C_values"])
            * len(config["tuning"]["class_weights"]),
            "models": models,
        },
        "input_checksums": checksums,
        "selection_kernel_diagnostics": {
            "fidelity": quantum_diagnostics,
            "tanimoto": tanimoto_diagnostics,
        },
        "selection_artifact_checksums": {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in selection_artifact_paths
        },
    }
    lock_path = output_dir / "locked_configuration.json"
    _write_json(lock_path, lock)
    lock_hash = _sha256(lock_path)
    manifest = {
        "status": "configuration_locked",
        "dataset": config["dataset_key"],
        "split": config["split_name"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_configuration_sha256": lock_hash,
        "input_checksums": checksums,
        "runtime": _runtime_info(),
        "selected_models": models,
        "selection_kernel_diagnostics": lock["selection_kernel_diagnostics"],
        "artifacts": {
            "locked_configuration": "locked_configuration.json",
            "feature_map_circuit": "circuit/zz_feature_map.qpy",
            "preprocessing_parameters": "preprocessing/scaler_pca_parameters.npz",
            "validation_predictions": [
                f"validation_predictions/{name}.csv" for name in MODEL_NAMES
            ],
            "training_and_validation_kernels": [
                "kernels/fidelity_train.npz",
                "kernels/fidelity_validation.npz",
                "kernels/tanimoto_train.npz",
                "kernels/tanimoto_validation.npz",
            ],
        },
    }
    _write_json(output_dir / "experiment_manifest.json", manifest)
    print(f"Configuration locked; no test labels were read. Output: {output_dir}")
    for name in MODEL_NAMES:
        model = models[name]
        print(
            f"{name}: C={model['parameters']['C']} "
            f"class_weight={model['parameters']['class_weight']} "
            f"validation AUROC={model['validation_metrics']['auroc']:.4f} "
            f"threshold={model['threshold']:.6g}"
        )


def _evaluation_phase(root: Path, output_dir: Path, script_path: Path) -> None:
    lock_path = output_dir / "locked_configuration.json"
    manifest_path = output_dir / "experiment_manifest.json"
    if not lock_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run --phase select first; the locked configuration is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "configuration_locked":
        raise ValueError("Evaluation requires a locked, not-yet-evaluated pilot directory")
    if manifest.get("locked_configuration_sha256") != _sha256(lock_path):
        raise ValueError("Locked configuration checksum mismatch; refusing test evaluation")

    config, config_path = _read_config(root)
    checksums = _input_checksums(root, config_path, script_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if checksums != lock["input_checksums"]:
        raise ValueError("Pilot input checksums changed since selection; refusing test evaluation")
    for relative_path, expected_hash in lock["selection_artifact_checksums"].items():
        artifact_path = output_dir / Path(relative_path)
        if not artifact_path.is_file() or _sha256(artifact_path) != expected_hash:
            raise ValueError(
                f"Locked selection artifact changed: {relative_path}; refusing test evaluation"
            )
    inputs = _load_inputs(root, config)
    samples = _sample_rows(config, inputs["split"])
    molecule_ids = {part: _ids(samples[part]) for part in PARTITIONS}
    if molecule_ids != lock["molecule_ids"]:
        raise ValueError("Sampled molecule IDs differ from the locked configuration")
    train_ids = molecule_ids["train"]
    validation_ids = molecule_ids["val"]
    test_ids = molecule_ids["test"]

    train_labels = _read_labels(inputs["processed_path"], samples["train"])
    validation_labels = _read_labels(inputs["processed_path"], samples["val"])
    preprocessing = _load_preprocessing(
        output_dir / "preprocessing" / "scaler_pca_parameters.npz"
    )
    train_matrices = {}
    validation_matrices = {}
    for model_name, prefix in (("fidelity_svc", "fidelity"), ("tanimoto_svm", "tanimoto")):
        train_matrix, train_rows, train_columns = _load_kernel(
            output_dir / "kernels" / f"{prefix}_train.npz"
        )
        val_matrix, val_rows, val_columns = _load_kernel(
            output_dir / "kernels" / f"{prefix}_validation.npz"
        )
        if not (
            train_rows.tolist() == train_ids
            and train_columns.tolist() == train_ids
            and val_rows.tolist() == validation_ids
            and val_columns.tolist() == train_ids
        ):
            raise ValueError(f"{model_name}: saved kernel molecule IDs do not match the lock")
        train_matrices[model_name] = train_matrix
        validation_matrices[model_name] = val_matrix

    models: dict[str, SVC] = {}
    training_seconds: dict[str, float] = {}
    prediction_seconds: dict[str, float] = {}
    for model_name in MODEL_NAMES:
        parameters = lock["tuning"]["models"][model_name]["parameters"]
        model = _fit_parameters(config, parameters["C"], parameters["class_weight"])
        started = time.perf_counter()
        model.fit(train_matrices[model_name], train_labels)
        training_seconds[model_name] = time.perf_counter() - started
        if not np.array_equal(model.classes_, np.array([0, 1])):
            raise ValueError(f"{model_name}: fitted class order is not [0, 1]")
        started = time.perf_counter()
        probabilities = model.predict_proba(validation_matrices[model_name])[:, 1]
        prediction_seconds[model_name] = time.perf_counter() - started
        frozen_validation = pd.read_csv(
            output_dir / "validation_predictions" / f"{model_name}.csv"
        )
        if frozen_validation["mol_id"].tolist() != validation_ids:
            raise ValueError(f"{model_name}: saved validation predictions are misaligned")
        if not np.array_equal(
            frozen_validation["y_true"].to_numpy(dtype=np.int8),
            validation_labels,
        ):
            raise ValueError(f"{model_name}: saved validation labels changed after selection")
        if not np.allclose(
            frozen_validation["p_pred"].to_numpy(),
            probabilities,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(
                f"{model_name}: deterministic refit does not reproduce locked validation probabilities"
            )
        models[model_name] = model
    # Test features and labels are accessed only after lock verification and parameter refit.
    fingerprints = inputs["fingerprints"]
    descriptors = inputs["descriptors"]
    combined = np.concatenate((fingerprints.astype(np.float64), descriptors), axis=1)
    quantum_test_features = apply_train_preprocessor(
        combined[samples["test"]],
        preprocessing,
    )
    quantum_train_features = apply_train_preprocessor(
        combined[samples["train"]],
        preprocessing,
    )
    quantum_test_self_features = quantum_test_features
    quantum_kernel, _ = _quantum_kernel(config)
    quantum_started = time.perf_counter()
    quantum_test_kernel = quantum_kernel.evaluate(quantum_test_features, quantum_train_features)
    quantum_test_self = quantum_kernel.evaluate(quantum_test_self_features)
    quantum_kernel_seconds = time.perf_counter() - quantum_started
    validate_cross_kernel(quantum_test_kernel)
    quantum_test_diagnostics = validate_square_kernel(quantum_test_self)

    tanimoto_started = time.perf_counter()
    tanimoto_test_kernel = tanimoto_kernel_matrix(
        fingerprints[samples["test"]],
        fingerprints[samples["train"]],
    )
    tanimoto_test_self = tanimoto_kernel_matrix(
        fingerprints[samples["test"]],
        fingerprints[samples["test"]],
    )
    tanimoto_kernel_seconds = time.perf_counter() - tanimoto_started
    validate_cross_kernel(tanimoto_test_kernel)
    tanimoto_test_diagnostics = validate_square_kernel(tanimoto_test_self)

    test_matrices = {
        "fidelity_svc": quantum_test_kernel,
        "tanimoto_svm": tanimoto_test_kernel,
    }
    test_kernel_diagnostics = {
        "fidelity": {
            "test_cross_finite": bool(np.isfinite(quantum_test_kernel).all()),
            "test_self": quantum_test_diagnostics,
        },
        "tanimoto": {
            "test_cross_finite": bool(np.isfinite(tanimoto_test_kernel).all()),
            "test_self": tanimoto_test_diagnostics,
        },
    }
    test_labels = _read_labels(inputs["processed_path"], samples["test"])
    if set(np.unique(test_labels)) != {0, 1}:
        raise ValueError("Locked test subset must contain both classes for requested metrics")

    metrics = {}
    for model_name in MODEL_NAMES:
        prediction_started = time.perf_counter()
        probabilities = models[model_name].predict_proba(test_matrices[model_name])[:, 1]
        test_prediction_seconds = time.perf_counter() - prediction_started
        threshold = float(lock["tuning"]["models"][model_name]["threshold"])
        values = calculate_pilot_metrics(test_labels, probabilities, threshold)
        metrics[model_name] = {
            "n_test": int(len(test_labels)),
            "positive_rate": float(test_labels.mean()),
            "threshold": threshold,
            "test_metrics": values,
            "kernel_construction_seconds_train_validation": lock["tuning"]["models"][model_name][
                "kernel_construction_seconds_train_validation"
            ],
            "kernel_construction_seconds_test": (
                quantum_kernel_seconds
                if model_name == "fidelity_svc"
                else tanimoto_kernel_seconds
            ),
            "kernel_construction_seconds_total": (
                lock["tuning"]["models"][model_name][
                    "kernel_construction_seconds_train_validation"
                ]
                + (
                    quantum_kernel_seconds
                    if model_name == "fidelity_svc"
                    else tanimoto_kernel_seconds
                )
            ),
            "training_seconds_final_fit": training_seconds[model_name],
            "validation_prediction_seconds_refit": prediction_seconds[model_name],
            "test_prediction_seconds": test_prediction_seconds,
        }
        _save_predictions(
            output_dir / "test_predictions" / f"{model_name}.csv",
            test_ids,
            test_labels,
            probabilities,
        )
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            model_dir / f"{model_name}_checkpoint.npz",
            support_vectors=models[model_name].support_,
            dual_coefficients=models[model_name].dual_coef_,
            intercept=models[model_name].intercept_,
            classes=models[model_name].classes_,
            C=np.asarray([models[model_name].C]),
            threshold=np.asarray([threshold]),
            train_ids=np.asarray(train_ids, dtype=str),
        )
        import joblib

        joblib.dump(models[model_name], model_dir / f"{model_name}.joblib")
    _write_json(output_dir / "metrics" / "test_metrics.json", metrics)
    _save_kernel(
        output_dir / "kernels" / "fidelity_test.npz",
        quantum_test_kernel,
        test_ids,
        train_ids,
        quantum_test_self,
    )
    _save_kernel(
        output_dir / "kernels" / "tanimoto_test.npz",
        tanimoto_test_kernel,
        test_ids,
        train_ids,
        tanimoto_test_self,
    )

    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["test_metrics"] = metrics
    manifest["test_kernel_diagnostics"] = test_kernel_diagnostics
    manifest["runtime"] = _runtime_info()
    manifest["artifacts"]["test_predictions"] = [
        f"test_predictions/{name}.csv" for name in MODEL_NAMES
    ]
    manifest["artifacts"]["test_kernels"] = [
        "kernels/fidelity_test.npz",
        "kernels/tanimoto_test.npz",
    ]
    manifest["artifacts"]["models"] = [
        f"models/{name}.joblib" for name in MODEL_NAMES
    ]
    manifest["artifacts"]["test_metrics"] = "metrics/test_metrics.json"
    manifest["artifact_checksums"] = {
        path.relative_to(output_dir).as_posix(): _sha256(path)
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    _write_json(manifest_path, manifest)
    for model_name in MODEL_NAMES:
        print(model_name, json.dumps(metrics[model_name], sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    script_path = Path(__file__).resolve()
    if args.phase == "select":
        _selection_phase(ROOT, output_dir, script_path)
    else:
        _evaluation_phase(ROOT, output_dir, script_path)


if __name__ == "__main__":
    main()
