from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/quantum_benchmarks/herg_karim/pennylane_research/qrc_depth_validation"
OUT.mkdir(parents=True, exist_ok=True)

N_QUBITS = 4
DEPTHS = (2, 4, 6)
SEED = 20261119
INPUT_SEED = 20261123
ATOL = 1e-8


def make_inputs():
    return np.random.default_rng(INPUT_SEED).normal(0.0, 0.7, size=(8, N_QUBITS))


def run_qiskit():
    sys.path.insert(0, str(ROOT / "src"))
    from qktox.quantum_reservoir import QuantumReservoir

    X = make_inputs()
    arrays = {"inputs": X}
    summaries = {}
    for depth in DEPTHS:
        model = QuantumReservoir(
            n_qubits=N_QUBITS,
            encoding_layers=depth,
            seed=SEED,
            entanglement="linear",
        )
        features = model.transform(X)
        arrays[f"features_{depth}"] = features
        arrays[f"rotations_{depth}"] = model.rotation_parameters
        summaries[str(depth)] = {
            "shape": list(features.shape),
            "output_dimension": model.output_dimension,
            "seed": SEED,
            "entanglement": "linear",
        }
        print(f"Qiskit {depth} layers: {features.shape}")

    np.savez_compressed(OUT / "qiskit_reference.npz", **arrays)
    (OUT / "qiskit_reference.json").write_text(
        json.dumps({"qubits": N_QUBITS, "depths": summaries}, indent=2),
        encoding="utf-8",
    )
    print("Qiskit references saved:", OUT)


def run_pennylane():
    import pennylane as qml

    ref_path = OUT / "qiskit_reference.npz"
    if not ref_path.exists():
        raise FileNotFoundError("Run --backend qiskit first.")

    ref = np.load(ref_path)
    X = ref["inputs"]
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    all_reports = []
    save_arrays = {"inputs": X}

    for depth in DEPTHS:
        rotations = ref[f"rotations_{depth}"]
        expected = ref[f"features_{depth}"]
        actual = np.empty_like(expected)
        width = 3 * N_QUBITS + N_QUBITS - 1

        for layer in range(depth):
            @qml.qnode(dev)
            def circuit(row):
                values = np.tanh(row)
                # Apply the complete circuit prefix so each layer's observables
                # are measured on the same cumulative state as the Qiskit code.
                for current_layer in range(layer + 1):
                    for wire in range(N_QUBITS):
                        qml.RY(np.pi * values[wire], wires=wire)
                        qml.RZ(0.5 * np.pi * values[wire], wires=wire)
                        qml.RX(rotations[current_layer, wire, 0], wires=wire)
                        qml.RY(rotations[current_layer, wire, 1], wires=wire)
                        qml.RZ(rotations[current_layer, wire, 2], wires=wire)
                    for wire in range(N_QUBITS - 1):
                        qml.CNOT(wires=[wire, wire + 1])

                measurements = []
                measurements.extend(
                    qml.expval(qml.PauliX(wires=w)) for w in range(N_QUBITS)
                )
                measurements.extend(
                    qml.expval(qml.PauliY(wires=w)) for w in range(N_QUBITS)
                )
                measurements.extend(
                    qml.expval(qml.PauliZ(wires=w)) for w in range(N_QUBITS)
                )
                measurements.extend(
                    qml.expval(
                        qml.PauliZ(wires=w) @ qml.PauliZ(wires=w + 1)
                    )
                    for w in range(N_QUBITS - 1)
                )
                return tuple(measurements)

            for row_index, row in enumerate(X):
                actual[row_index, layer * width:(layer + 1) * width] = np.asarray(
                    circuit(row), dtype=np.float64
                )

        difference = np.abs(expected - actual)
        passed = bool(np.allclose(expected, actual, rtol=0.0, atol=ATOL))
        save_arrays[f"features_{depth}"] = actual

        report = {
            "layers": depth,
            "feature_shape": list(actual.shape),
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "tolerance": ATOL,
            "passed": passed,
        }
        all_reports.append(report)
        print(
            f"PennyLane {depth} layers: {actual.shape}; "
            f"max error={difference.max():.3e}; "
            f"{'PASS' if passed else 'FAIL'}"
        )

    np.savez_compressed(OUT / "pennylane_features.npz", **save_arrays)
    final = {
        "comparison": "Qiskit vs PennyLane quantum reservoir depth validation",
        "pennylane_version": qml.__version__,
        "qubits": N_QUBITS,
        "depths": all_reports,
        "all_passed": all(item["passed"] for item in all_reports),
    }
    (OUT / "parity_report.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    print("\nOVERALL:", "PASSED" if final["all_passed"] else "FAILED")
    print("Report:", OUT / "parity_report.json")
    if not final["all_passed"]:
        raise SystemExit("At least one depth failed numerical parity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["qiskit", "pennylane"], required=True)
    args = parser.parse_args()
    if args.backend == "qiskit":
        run_qiskit()
    else:
        run_pennylane()
