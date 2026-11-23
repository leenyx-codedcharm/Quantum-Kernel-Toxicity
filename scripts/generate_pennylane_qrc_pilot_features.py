from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results/quantum_benchmarks/herg_karim/pennylane_research/full_pilot"
CONFIG = ROOT / "configs/qm_qrc_pilot.yaml"
PARTITIONS = ("train", "val", "test")
DEPTHS = (2, 4, 6)
N_QUBITS = 4
SEED = 20261119
ATOL = 1e-8
REFERENCE_LOCKS = (
    "results/quantum_fidelity/fidelity_statevector_pilot_r0_final/locked_configuration.json",
    "results/quantum_benchmarks/herg_karim/fidelity_trainable/locked_configuration.json",
    "results/quantum_benchmarks/herg_karim/vqc_cobyla/locked_configuration.json",
    "results/quantum_benchmarks/herg_karim/vqc_spsa/locked_configuration.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def row_index(molecule_id: str) -> int:
    prefix, suffix = molecule_id.rsplit("_", 1)
    if prefix != "herg_karim" or len(suffix) != 6 or not suffix.isdigit():
        raise ValueError(f"Unexpected molecule ID: {molecule_id}")
    return int(suffix)


def read_selected_column(path: Path, column: str, indices: np.ndarray):
    selected = set(map(int, indices))
    series = pd.read_csv(
        path,
        usecols=[column],
        skiprows=lambda line: line != 0 and line - 1 not in selected,
    )[column]
    if len(series) != len(indices):
        raise ValueError(f"Could not load all requested rows from {path}")
    return series.tolist()


def load_frozen_inputs():
    config = json.loads((ROOT / REFERENCE_LOCKS[0]).read_text(encoding="utf-8"))
    if config.get("status") != "locked_for_test_evaluation":
        raise ValueError("Source fidelity pilot is not locked for test evaluation.")
    molecule_ids = config["molecule_ids"]
    ids = {part: list(molecule_ids[part]) for part in PARTITIONS}
    for lock_rel in REFERENCE_LOCKS[1:]:
        lock = json.loads((ROOT / lock_rel).read_text(encoding="utf-8"))
        for part in PARTITIONS:
            if lock.get("molecule_ids", {}).get(part) != ids[part]:
                raise ValueError(f"Molecule IDs differ in lock: {lock_rel}")

    indices = {part: np.asarray([row_index(x) for x in ids[part]], dtype=np.int64)
               for part in PARTITIONS}
    for part in PARTITIONS:
        if len(np.unique(indices[part])) != len(indices[part]):
            raise ValueError(f"Duplicate row indices in {part}")
        if np.any(np.diff(indices[part]) <= 0):
            raise ValueError(f"{part} IDs are not in frozen row order")

    split_path = ROOT / "data/splits/herg_karim_scaffold.json"
    split = json.loads(split_path.read_text(encoding="utf-8-sig"))["scaffold_r0"]
    for part in PARTITIONS:
        if not set(indices[part]).issubset(set(split[part])):
            raise ValueError(f"{part} pilot IDs fall outside frozen scaffold_r0 {part}")
    for i, left in enumerate(PARTITIONS):
        for right in PARTITIONS[i + 1:]:
            if set(indices[left]) & set(indices[right]):
                raise ValueError(f"Partition overlap: {left}/{right}")

    feature_dir = ROOT / "data/processed/features/hERG_Karim"
    fp_path = feature_dir / "ecfp4_2048.npz"
    rdkit_path = feature_dir / "rdkit_descriptors.csv"
    feature_manifest_path = feature_dir / "feature_manifest.json"
    processed_path = ROOT / "data/processed/herg_karim.csv"
    preprocessing_path = ROOT / REFERENCE_LOCKS[0].replace(
        "locked_configuration.json", "preprocessing/scaler_pca_parameters.npz"
    )

    with np.load(fp_path, allow_pickle=False) as archive:
        fingerprints = archive["fingerprints"]
        fp_ids = archive["mol_ids"].astype(str)
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    descriptor_names = feature_manifest["descriptor_names"]
    descriptors = pd.read_csv(rdkit_path, usecols=["mol_id", *descriptor_names],
                               dtype={"mol_id": str})
    descriptor_ids = descriptors["mol_id"].to_numpy(dtype=str)
    expected = np.asarray([f"herg_karim_{i:06d}" for i in range(len(fp_ids))])
    if not np.array_equal(fp_ids, expected) or not np.array_equal(descriptor_ids, expected):
        raise ValueError("Fingerprint/descriptor IDs do not match processed row order.")
    if fingerprints.shape[1] != 2048 or not np.isin(fingerprints, (0, 1)).all():
        raise ValueError("Invalid ECFP4 fingerprint matrix.")
    rdkit = descriptors.loc[:, descriptor_names].to_numpy(dtype=np.float64)
    raw = np.concatenate([fingerprints.astype(np.float64), rdkit], axis=1)
    with np.load(preprocessing_path, allow_pickle=False) as archive:
        prep = {key: archive[key] for key in archive.files}

    from qktox.quantum_pilot import apply_train_preprocessor
    matrices = {}
    for part in PARTITIONS:
        matrices[part] = apply_train_preprocessor(raw[indices[part]], prep)
        if matrices[part].shape != (len(ids[part]), N_QUBITS):
            raise ValueError(f"Unexpected preprocessed matrix shape for {part}: {matrices[part].shape}")
        if not np.isfinite(matrices[part]).all():
            raise ValueError(f"Non-finite preprocessed inputs in {part}")

    labels = {}
    for part in PARTITIONS:
        y = pd.to_numeric(pd.Series(read_selected_column(processed_path, "label", indices[part])),
                          errors="coerce")
        if y.isna().any() or set(y.unique()) != {0, 1}:
            raise ValueError(f"Invalid labels in {part}")
        labels[part] = y.to_numpy(dtype=np.int8)

    checksums = {}
    for part in PARTITIONS:
        smiles = read_selected_column(processed_path, "smiles", indices[part])
        checksums[part] = {
            mol_id: hashlib.sha256(str(smi).encode("utf-8")).hexdigest()
            for mol_id, smi in zip(ids[part], smiles)
        }

    input_paths = [
        processed_path, split_path, fp_path, rdkit_path, feature_manifest_path,
        preprocessing_path, ROOT / REFERENCE_LOCKS[0],
    ]
    return ids, matrices, labels, checksums, input_paths, descriptor_names


def qiskit_backend():
    from qktox.quantum_reservoir import QuantumReservoir

    ids, matrices, labels, checksums, input_paths, descriptor_names = load_frozen_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {}
    configs = {}
    start = time.perf_counter()
    for depth in DEPTHS:
        model = QuantumReservoir(n_qubits=N_QUBITS, encoding_layers=depth,
                                 seed=SEED, entanglement="linear")
        configs[str(depth)] = model.configuration()
        for part in PARTITIONS:
            key = f"quantum_{depth}_{part}"
            arrays[key] = model.transform(matrices[part])
            print(f"Qiskit {depth} layers {part}: {arrays[key].shape}", flush=True)
    elapsed = time.perf_counter() - start
    np.savez_compressed(OUT / "qiskit_reference_features.npz", **arrays)
    np.savez_compressed(OUT / "preprocessed_inputs.npz",
                        **{f"{p}_X": matrices[p] for p in PARTITIONS})
    (OUT / "molecule_ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    (OUT / "labels.json").write_text(json.dumps({p: labels[p].tolist() for p in PARTITIONS},
                                                indent=2), encoding="utf-8")
    (OUT / "qiskit_reservoir_configurations.json").write_text(
        json.dumps(configs, indent=2), encoding="utf-8")
    write_json(OUT / "qiskit_generation.json", {
        "backend": "Qiskit Statevector",
        "qubits": N_QUBITS, "depths": list(DEPTHS), "seed": SEED,
        "entanglement": "linear CNOT", "shots": None,
        "partition_sizes": {p: len(ids[p]) for p in PARTITIONS},
        "input_dimension": N_QUBITS,
        "runtime_seconds": elapsed,
        "input_checksums": {str(p.relative_to(ROOT)): sha256(p) for p in input_paths},
        "descriptor_columns": descriptor_names,
        "qm_descriptors_included": False,
        "scope": "structural-only pilot; ECFP4 plus RDKit, train-fitted scaler/PCA",
    })
    print(f"Qiskit reference generation finished in {elapsed:.2f}s")


def pennylane_backend():
    import pennylane as qml

    ref_path = OUT / "qiskit_reference_features.npz"
    input_path = OUT / "preprocessed_inputs.npz"
    if not ref_path.exists() or not input_path.exists():
        raise FileNotFoundError("Run --backend qiskit first to create frozen inputs/reference.")
    ids = json.loads((OUT / "molecule_ids.json").read_text(encoding="utf-8"))
    reference = np.load(ref_path)
    input_archive = np.load(input_path)
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    arrays = {}
    reports = []
    start = time.perf_counter()

    for depth in DEPTHS:
        rotations = np.asarray(
            json.loads((OUT / "qiskit_reservoir_configurations.json").read_text(encoding="utf-8"))[str(depth)]
            ["rotation_parameters"], dtype=np.float64
        )
        width = 3 * N_QUBITS + N_QUBITS - 1

        @qml.qnode(dev)
        def circuit(row, layer):
            x = np.tanh(row)
            # Recreate the cumulative Qiskit state through the requested layer.
            for current in range(layer + 1):
                for wire in range(N_QUBITS):
                    qml.RY(np.pi * x[wire], wires=wire)
                    qml.RZ(0.5 * np.pi * x[wire], wires=wire)
                    qml.RX(rotations[current, wire, 0], wires=wire)
                    qml.RY(rotations[current, wire, 1], wires=wire)
                    qml.RZ(rotations[current, wire, 2], wires=wire)
                for wire in range(N_QUBITS - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            obs = [qml.expval(qml.PauliX(wires=w)) for w in range(N_QUBITS)]
            obs += [qml.expval(qml.PauliY(wires=w)) for w in range(N_QUBITS)]
            obs += [qml.expval(qml.PauliZ(wires=w)) for w in range(N_QUBITS)]
            obs += [qml.expval(qml.PauliZ(wires=w) @ qml.PauliZ(wires=w + 1))
                    for w in range(N_QUBITS - 1)]
            return tuple(obs)

        for part in PARTITIONS:
            X = input_archive[f"{part}_X"]
            F = np.empty((len(X), depth * width), dtype=np.float64)
            for i, row in enumerate(X):
                for layer in range(depth):
                    F[i, layer * width:(layer + 1) * width] = np.asarray(
                        circuit(row, layer), dtype=np.float64)
            arrays[f"quantum_{depth}_{part}"] = F
            expected = reference[f"quantum_{depth}_{part}"]
            diff = np.abs(expected - F)
            passed = bool(np.allclose(expected, F, rtol=0.0, atol=ATOL))
            reports.append({
                "depth": depth, "partition": part,
                "molecules": len(X), "feature_shape": list(F.shape),
                "max_absolute_error": float(diff.max()),
                "mean_absolute_error": float(diff.mean()),
                "tolerance": ATOL, "passed": passed,
            })
            print(f"PennyLane {depth} layers {part}: {F.shape}; "
                  f"max error={diff.max():.3e}; {'PASS' if passed else 'FAIL'}",
                  flush=True)

    elapsed = time.perf_counter() - start
    np.savez_compressed(OUT / "pennylane_features.npz", **arrays)
    all_passed = all(r["passed"] for r in reports)
    write_json(OUT / "pennylane_parity_report.json", {
        "backend": "PennyLane Lightning Statevector",
        "pennylane_version": qml.__version__,
        "qubits": N_QUBITS, "depths": list(DEPTHS),
        "runtime_seconds": elapsed, "absolute_tolerance": ATOL,
        "all_passed": all_passed, "comparisons": reports,
        "partition_sizes": {p: len(ids[p]) for p in PARTITIONS},
        "qm_descriptors_included": False,
    })
    print("\nOVERALL:", "PASSED" if all_passed else "FAILED")
    print("Report:", OUT / "pennylane_parity_report.json")
    if not all_passed:
        raise SystemExit("Full pilot parity failed; do not train readouts.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["qiskit", "pennylane"], required=True)
    args = parser.parse_args()
    if args.backend == "qiskit":
        qiskit_backend()
    else:
        pennylane_backend()


if __name__ == "__main__":
    main()
