"""Run a fixed-reservoir hERG pilot on the matched fidelity-pilot molecules."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qktox.qm_qrc import (
    ClassicalRandomReservoir,
    calculate_metrics,
    positive_probabilities,
    tune_readout,
)
from qktox.quantum_pilot import apply_train_preprocessor
from qktox.quantum_reservoir import QuantumReservoir


PARTITIONS = ("train", "val", "test")
REFERENCE_LOCKS = (
    "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/locked_configuration.json",
    "results/quantum_benchmarks/herg_karim/fidelity_trainable/locked_configuration.json",
    "results/quantum_benchmarks/herg_karim/vqc_cobyla/locked_configuration.json",
    "results/quantum_benchmarks/herg_karim/vqc_spsa/locked_configuration.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return config


def _pilot_indices(
    root: Path,
    config: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    source_lock_path = root / config["source_pilot"] / "locked_configuration.json"
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if source_lock.get("status") != "locked_for_test_evaluation":
        raise ValueError("The reference fidelity pilot is not locked for test evaluation")
    molecule_ids = source_lock.get("molecule_ids")
    if not isinstance(molecule_ids, dict):
        raise ValueError("The reference pilot lock has no molecule IDs")

    reference_ids = {partition: molecule_ids[partition] for partition in PARTITIONS}
    for relative_path in REFERENCE_LOCKS[1:]:
        lock = json.loads((root / relative_path).read_text(encoding="utf-8"))
        for partition in PARTITIONS:
            if lock.get("molecule_ids", {}).get(partition) != reference_ids[partition]:
                raise ValueError(
                    f"{relative_path}: {partition} IDs differ from the fidelity pilot"
                )

    indices: dict[str, np.ndarray] = {}
    for partition, ids in reference_ids.items():
        if len(ids) != int(config["sample_sizes"][partition]):
            raise ValueError(f"{partition}: sample size differs from the pilot config")
        values = np.asarray([_row_index(molecule_id) for molecule_id in ids], dtype=np.int64)
        if len(np.unique(values)) != len(values) or np.any(np.diff(values) <= 0):
            raise ValueError(f"{partition}: molecule IDs must be unique and row ordered")
        indices[partition] = values

    frozen_path = root / "data" / "splits" / "herg_karim_scaffold.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8-sig"))
    split = frozen[config["split_name"]]
    for partition in PARTITIONS:
        if not set(indices[partition]).issubset(set(split[partition])):
            raise ValueError(f"{partition}: pilot IDs are outside the frozen scaffold split")
    if any(
        set(indices[left]) & set(indices[right])
        for offset, left in enumerate(PARTITIONS)
        for right in PARTITIONS[offset + 1 :]
    ):
        raise ValueError("Pilot partitions overlap")
    return reference_ids, indices


def _row_index(molecule_id: str) -> int:
    prefix, row = molecule_id.rsplit("_", 1)
    if prefix != "herg_karim" or len(row) != 6 or not row.isdigit():
        raise ValueError(f"Unexpected molecule ID: {molecule_id}")
    return int(row)


def _read_selected_column(path: Path, column: str, indices: np.ndarray) -> list[Any]:
    selected = set(int(index) for index in indices)
    values = pd.read_csv(
        path,
        usecols=[column],
        skiprows=lambda line_number: line_number != 0
        and line_number - 1 not in selected,
    )[column]
    if len(values) != len(indices):
        raise ValueError(f"{path}: could not read all requested {column} rows")
    return values.tolist()


def _read_labels(path: Path, indices: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(
        pd.Series(_read_selected_column(path, "label", indices)),
        errors="coerce",
    )
    if values.isna().any() or not set(values.unique()).issubset({0, 1}):
        raise ValueError("Selected labels must be binary 0/1")
    labels = values.to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Each selected partition must contain both label classes")
    return labels


def _structure_checksums(
    processed_path: Path,
    indices: np.ndarray,
) -> dict[str, str]:
    smiles = _read_selected_column(processed_path, "smiles", indices)
    return {
        f"herg_karim_{int(index):06d}": hashlib.sha256(
            str(structure).encode("utf-8")
        ).hexdigest()
        for index, structure in zip(indices, smiles)
    }


def _load_features(
    root: Path,
    config: dict[str, Any],
    indices: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    feature_dir = root / "data" / "processed" / "features" / "hERG_Karim"
    fingerprint_path = feature_dir / "ecfp4_2048.npz"
    descriptor_path = feature_dir / "rdkit_descriptors.csv"
    descriptor_manifest_path = feature_dir / "feature_manifest.json"
    processed_path = root / "data" / "processed" / "herg_karim.csv"
    split_path = root / "data" / "splits" / "herg_karim_scaffold.json"
    source_root = root / config["source_pilot"]
    preprocessing_path = source_root / "preprocessing" / "scaler_pca_parameters.npz"

    with np.load(fingerprint_path, allow_pickle=False) as archive:
        fingerprints = archive["fingerprints"]
        fingerprint_ids = archive["mol_ids"].astype(str)
    descriptor_manifest = json.loads(descriptor_manifest_path.read_text(encoding="utf-8"))
    descriptor_names = descriptor_manifest["descriptor_names"]
    descriptors = pd.read_csv(
        descriptor_path,
        usecols=["mol_id", *descriptor_names],
        dtype={"mol_id": str},
    )
    descriptor_ids = descriptors["mol_id"].to_numpy(dtype=str)
    expected_ids = np.asarray(
        [f"herg_karim_{index:06d}" for index in range(len(fingerprint_ids))]
    )
    if not np.array_equal(fingerprint_ids, expected_ids):
        raise ValueError("Fingerprint molecule IDs do not match processed row positions")
    if not np.array_equal(descriptor_ids, expected_ids):
        raise ValueError("Descriptor molecule IDs do not match fingerprint IDs")
    if fingerprints.ndim != 2 or fingerprints.shape[1] != 2048:
        raise ValueError("Expected 2,048-bit ECFP4 fingerprints")
    if not np.isin(fingerprints, (0, 1)).all():
        raise ValueError("ECFP4 fingerprint matrix contains non-binary values")

    descriptor_values = descriptors.loc[:, descriptor_names].to_numpy(dtype=np.float64)
    combined = np.concatenate((fingerprints.astype(np.float64), descriptor_values), axis=1)
    with np.load(preprocessing_path, allow_pickle=False) as archive:
        preprocessing = {key: archive[key] for key in archive.files}
    features = {
        partition: apply_train_preprocessor(combined[positions], preprocessing)
        for partition, positions in indices.items()
    }
    if any(
        values.shape[1] != int(config["feature_representation"]["input_dimensions"])
        for values in features.values()
    ):
        raise ValueError("Preprocessed feature dimension differs from pilot configuration")

    inputs = {
        "processed_dataset": processed_path,
        "frozen_splits": split_path,
        "fingerprints": fingerprint_path,
        "descriptors": descriptor_path,
        "descriptor_manifest": descriptor_manifest_path,
        "preprocessing": preprocessing_path,
        "source_fidelity_lock": source_root / "locked_configuration.json",
    }
    return features, inputs


def _runtime_info() -> dict[str, Any]:
    distributions = (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "qiskit",
        "qiskit-algorithms",
        "qiskit-machine-learning",
        "PyYAML",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "scikit_learn": sklearn.__version__,
        "versions": {
            package: importlib.metadata.version(package)
            for package in distributions
        },
    }


def _build_reservoir_features(
    reservoir_type: str,
    layers: int,
    seed: int,
    partitions: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    start = time.perf_counter()
    if reservoir_type == "quantum":
        reservoir = QuantumReservoir(
            n_qubits=4,
            encoding_layers=layers,
            seed=seed,
            entanglement="linear",
        )
        values = {partition: reservoir.transform(matrix) for partition, matrix in partitions.items()}
    elif reservoir_type == "classical":
        reservoir = ClassicalRandomReservoir(
            input_dimension=4,
            output_dimension_per_layer=15,
            seed=seed,
        )
        values = {
            partition: reservoir.transform(matrix, layers)
            for partition, matrix in partitions.items()
        }
    else:
        raise ValueError(f"Unsupported reservoir type: {reservoir_type}")
    return values, reservoir.configuration(), time.perf_counter() - start


def _save_predictions(
    path: Path,
    molecule_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "mol_id": molecule_ids,
            "y_true": labels.astype(np.int8),
            "p_pred": probabilities,
        }
    ).to_csv(path, index=False)


def _write_comparison(root: Path, records: list[dict[str, Any]]) -> None:
    comparison_dir = root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(comparison_dir / "pilot_metrics.csv", index=False)
    lines = [
        "# QM-QRC Structural Pilot",
        "",
        "One matched, sampled holdout from `herg_karim/scaffold_r0` "
        "(64 train, 128 validation, 128 test molecules).",
        "This is a pilot result, not a five-repeat benchmark or evidence of quantum advantage.",
        "**Exploratory only:** the first test metrics were generated before the QM-QRC method "
        "was recorded in `paper/protocol.md`; they are post hoc and must not guide future "
        "model selection.",
        "Readouts were selected by validation AUROC and thresholds by validation balanced accuracy.",
        "",
        "| Model | AUROC | AUPRC | Positive F1 | MCC |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in records:
        if record.get("auroc") is not None and record.get("auroc") != "":
            lines.append(
                f"| {record['model']} | {record['auroc']:.4f} | "
                f"{record['auprc']:.4f} | {record['positive_f1']:.4f} | "
                f"{record['mcc']:.4f} |"
            )
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "pilot_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path, output_override: Path | None = None) -> Path:
    config = _read_yaml(config_path)
    if config.get("dataset_key") != "herg_karim" or config.get("split_name") != "scaffold_r0":
        raise ValueError("This pilot only supports herg_karim/scaffold_r0")
    output_dir = output_override or (ROOT / config["output_dir"])
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment output: {output_dir}"
        )

    molecule_ids, indices = _pilot_indices(ROOT, config)
    train_val_indices = {
        partition: indices[partition] for partition in ("train", "val")
    }
    input_features, input_paths = _load_features(ROOT, config, train_val_indices)
    # Test labels are intentionally not loaded until the selection lock is written.
    train_labels = _read_labels(ROOT / "data" / "processed" / "herg_karim.csv", indices["train"])
    validation_labels = _read_labels(ROOT / "data" / "processed" / "herg_karim.csv", indices["val"])
    train_val_features = {
        "train": input_features["train"],
        "val": input_features["val"],
    }

    seed = int(config["seed"])
    layers_to_evaluate = tuple(int(value) for value in config["reservoir"]["encoding_layers"])
    readout_names = tuple(str(value) for value in config["readouts"]["names"])
    regularization_values = tuple(
        float(value) for value in config["readouts"]["regularization_values"]
    )
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    reservoir_configs: dict[tuple[str, int], dict[str, Any]] = {}
    feature_times: dict[tuple[str, int], float] = {}
    train_validation_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    selections = []

    for reservoir_type in ("quantum", "classical"):
        for layers in layers_to_evaluate:
            transformed, reservoir_config, feature_seconds = _build_reservoir_features(
                reservoir_type,
                layers,
                seed,
                train_val_features,
            )
            train_validation_cache[(reservoir_type, layers)] = transformed
            reservoir_configs[(reservoir_type, layers)] = reservoir_config
            feature_times[(reservoir_type, layers)] = feature_seconds
            selection = tune_readout(
                transformed["train"],
                train_labels,
                transformed["val"],
                validation_labels,
                readout_names=readout_names,
                regularization_values=regularization_values,
                seed=seed,
            )
            chosen[(reservoir_type, layers)] = selection
            selections.append(
                {
                    "reservoir": reservoir_type,
                    "encoding_layers": layers,
                    "selected_readout": selection["readout"],
                    "regularization": selection["regularization"],
                    "validation_auroc": selection["validation_auroc"],
                    "threshold": selection["threshold"],
                    "validation_balanced_accuracy": selection[
                        "validation_balanced_accuracy"
                    ],
                    "validation_metrics": selection["validation_metrics"],
                    "tuning_seconds": selection["tuning_seconds"],
                    "feature_generation_seconds_train_val": feature_seconds,
                    "trials": selection["trials"],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "features").mkdir()
    (output_dir / "models").mkdir()
    (output_dir / "circuit").mkdir()
    _write_json(
        output_dir / "reservoir_configurations.json",
        {
            f"{kind}_{layers}_layers": parameters
            for (kind, layers), parameters in reservoir_configs.items()
        },
    )
    np.savez_compressed(
        output_dir / "features" / "train_validation_reservoir_features.npz",
        **{
            f"{kind}_{layers}_{partition}": matrix
            for (kind, layers), partitions in train_validation_cache.items()
            for partition, matrix in partitions.items()
        },
    )
    for layers in layers_to_evaluate:
        circuit = QuantumReservoir(
            n_qubits=4,
            encoding_layers=layers,
            seed=seed,
            entanglement="linear",
        ).circuit_for(input_features["train"][0])
        with (output_dir / "circuit" / f"quantum_reservoir_{layers}_layers.qpy").open("wb") as stream:
            qpy.dump(circuit, stream)

    input_checksums = {
        name: _sha256(path)
        for name, path in input_paths.items()
    }
    input_checksums["experiment_config"] = _sha256(config_path)
    selection_lock = {
        "architecture": "QM-augmented Quantum Reservoir Classifier structural pilot",
        "dataset": config["dataset_key"],
        "split": config["split_name"],
        "seed": seed,
        "status": "locked_for_test_evaluation",
        "source_pilot": config["source_pilot"],
        "sample_sizes": {partition: len(indices[partition]) for partition in PARTITIONS},
        "molecule_ids": molecule_ids,
        "train_structure_checksums": _structure_checksums(
            input_paths["processed_dataset"], indices["train"]
        ),
        "validation_structure_checksums": _structure_checksums(
            input_paths["processed_dataset"], indices["val"]
        ),
        "input_checksums": input_checksums,
        "preprocessing": "source fidelity scaler/PCA fitted on the identical 64 training IDs",
        "selection_metric": config["readouts"]["selection_metric"],
        "threshold_metric": config["readouts"]["threshold_metric"],
        "readout_regularization_grid": list(regularization_values),
        "reservoir_selections": selections,
        "reservoir_configurations": {
            f"{kind}_{layers}_layers": parameters
            for (kind, layers), parameters in reservoir_configs.items()
        },
        "locked_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "locked_configuration.json", selection_lock)

    # Test features and labels are accessed only after validation choices are locked.
    test_input_features, _ = _load_features(
        ROOT,
        config,
        {"test": indices["test"]},
    )
    test_features: dict[tuple[str, int], np.ndarray] = {}
    test_feature_seconds: dict[tuple[str, int], float] = {}
    for reservoir_type in ("quantum", "classical"):
        for layers in layers_to_evaluate:
            transformed, _, seconds = _build_reservoir_features(
                reservoir_type,
                layers,
                seed,
                {"test": test_input_features["test"]},
            )
            test_features[(reservoir_type, layers)] = transformed["test"]
            test_feature_seconds[(reservoir_type, layers)] = seconds
    np.savez_compressed(
        output_dir / "features" / "test_reservoir_features.npz",
        **{
            f"{kind}_{layers}": matrix
            for (kind, layers), matrix in test_features.items()
        },
    )
    test_labels = _read_labels(
        ROOT / "data" / "processed" / "herg_karim.csv",
        indices["test"],
    )
    test_structure_checksums = _structure_checksums(
        input_paths["processed_dataset"],
        indices["test"],
    )

    records: list[dict[str, Any]] = []
    for model_name, path in (
        ("Fixed Fidelity Kernel", ROOT / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/metrics/test_metrics.json"),
        ("Tanimoto-SVM", ROOT / "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/metrics/test_metrics.json"),
        ("Trainable Fidelity Kernel", ROOT / "results/quantum_benchmarks/herg_karim/fidelity_trainable/metrics/test_metrics.json"),
        ("VQC COBYLA", ROOT / "results/quantum_benchmarks/herg_karim/vqc_cobyla/metrics/test_metrics.json"),
        ("VQC SPSA", ROOT / "results/quantum_benchmarks/herg_karim/vqc_spsa/metrics/test_metrics.json"),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = (
            payload["tanimoto_svm"]["test_metrics"]
            if model_name == "Tanimoto-SVM"
            else payload["fidelity_svc"]["test_metrics"]
            if model_name == "Fixed Fidelity Kernel"
            else payload["test_metrics"]
        )
        records.append({"model": model_name, **metrics})

    final_metrics = {}
    for (reservoir_type, layers), selection in chosen.items():
        model = selection["model"]
        x_test = test_features[(reservoir_type, layers)]
        prediction_start = time.perf_counter()
        test_probabilities = positive_probabilities(model, x_test)
        prediction_seconds = time.perf_counter() - prediction_start
        metrics = calculate_metrics(
            test_labels,
            test_probabilities,
            float(selection["threshold"]),
        )
        name = f"{reservoir_type}_{layers}_layers_{selection['readout']}"
        joblib.dump(model, output_dir / "models" / f"{name}.joblib")
        _save_predictions(
            output_dir / "test_predictions" / f"{name}.csv",
            molecule_ids["test"],
            test_labels,
            test_probabilities,
        )
        validation_probabilities = selection["validation_probabilities"]
        _save_predictions(
            output_dir / "validation_predictions" / f"{name}.csv",
            molecule_ids["val"],
            validation_labels,
            validation_probabilities,
        )
        row = {
            "model": name,
            **metrics,
            "threshold": float(selection["threshold"]),
            "validation_auroc": float(selection["validation_auroc"]),
            "readout_training_seconds": float(selection["tuning_seconds"]),
            "test_prediction_seconds": float(prediction_seconds),
            "train_val_feature_generation_seconds": feature_times[(reservoir_type, layers)],
            "test_feature_generation_seconds": test_feature_seconds[(reservoir_type, layers)],
            "reservoir": reservoir_type,
            "encoding_layers": layers,
            "readout": selection["readout"],
            "regularization": float(selection["regularization"]),
        }
        records.append(row)
        final_metrics[name] = row

    _write_json(
        output_dir / "metrics.json",
        {
            "status": "completed",
            "test_partition": "sampled scaffold_r0 test molecules",
            "n_test": len(test_labels),
            "positive_rate": float(test_labels.mean()),
            "test_structure_checksums": test_structure_checksums,
            "models": final_metrics,
        },
    )
    _write_comparison(output_dir, records)
    manifest = {
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config["dataset_key"],
        "split": config["split_name"],
        "architecture": "fixed four-qubit gate-based QRC and matched classical random reservoir",
        "pilot_only": True,
        "qm_descriptors_included": False,
        "sample_sizes": {partition: len(indices[partition]) for partition in PARTITIONS},
        "input_checksums": input_checksums,
        "runtime": _runtime_info(),
        "reservoir_configuration": str(output_dir / "reservoir_configurations.json"),
        "locked_configuration": str(output_dir / "locked_configuration.json"),
        "metrics": str(output_dir / "metrics.json"),
        "limitations": [
            "single sampled scaffold_r0 holdout; not a five-repeat estimate",
            "no quantum-mechanical descriptors in this first structural pilot",
            "statevector simulation; no hardware or shot-noise claims",
            "random-feature and readout selection are descriptive pilot comparisons",
        ],
    }
    manifest["artifact_checksums"] = {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "experiment_manifest.json"
    }
    _write_json(output_dir / "experiment_manifest.json", manifest)
    print(f"QM-QRC structural pilot completed: {output_dir}")
    for name, metrics in final_metrics.items():
        print(
            f"{name}: AUROC={metrics['auroc']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}, "
            f"positive-F1={metrics['positive_f1']:.4f}, "
            f"MCC={metrics['mcc']:.4f}"
        )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "qm_qrc_pilot.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    run(arguments.config.resolve(), arguments.output.resolve() if arguments.output else None)


if __name__ == "__main__":
    main()
