from pathlib import Path
import argparse, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/quantum_benchmarks/herg_karim/pennylane_research/qrc_parity"
OUT.mkdir(parents=True, exist_ok=True)
N, LAYERS, SEED, ATOL = 4, 2, 20261119, 1e-8

def inputs():
    return np.random.default_rng(20261123).normal(0, .7, (5, N))

def qiskit_run():
    sys.path.insert(0, str(ROOT / "src"))
    from qktox.quantum_reservoir import QuantumReservoir
    X = inputs()
    model = QuantumReservoir(n_qubits=N, encoding_layers=LAYERS, seed=SEED, entanglement="linear")
    F = model.transform(X)
    np.savez_compressed(OUT / "qiskit_reference.npz", inputs=X, features=F,
                        rotation_parameters=model.rotation_parameters)
    print("Qiskit reference saved; shape:", F.shape)

def pennylane_run():
    import pennylane as qml
    path = OUT / "qiskit_reference.npz"
    if not path.exists():
        raise FileNotFoundError("Run the Qiskit backend first: " + str(path))
    data = np.load(path)
    X, expected, rotations = data["inputs"], data["features"], data["rotation_parameters"]
    dev = qml.device("lightning.qubit", wires=N)

    @qml.qnode(dev)
    def step(row, layer):
        x = np.tanh(row)
        # Qiskit accumulates each reservoir step on the same state.
        # Rebuild the complete circuit from layer 0 through this layer.
        for current_layer in range(layer + 1):
            for w in range(N):
                qml.RY(np.pi*x[w], wires=w)
                qml.RZ(.5*np.pi*x[w], wires=w)
                qml.RX(rotations[current_layer,w,0], wires=w)
                qml.RY(rotations[current_layer,w,1], wires=w)
                qml.RZ(rotations[current_layer,w,2], wires=w)
            for w in range(N-1):
                qml.CNOT(wires=[w,w+1])
        obs = [qml.expval(qml.PauliX(wires=w)) for w in range(N)]
        obs += [qml.expval(qml.PauliY(wires=w)) for w in range(N)]
        obs += [qml.expval(qml.PauliZ(wires=w)) for w in range(N)]
        obs += [qml.expval(qml.PauliZ(wires=w) @ qml.PauliZ(wires=w+1)) for w in range(N-1)]
        return tuple(obs)

    F = np.empty_like(expected)
    width = 3*N + N-1
    for i, row in enumerate(X):
        for layer in range(LAYERS):
            F[i,layer*width:(layer+1)*width] = np.asarray(step(row,layer), dtype=float)

    diff = np.abs(expected-F)
    passed = bool(np.allclose(expected,F,rtol=0,atol=ATOL))
    np.savez_compressed(OUT/"pennylane_result.npz", inputs=X, features=F, absolute_difference=diff)
    report = {
        "comparison":"Qiskit vs PennyLane QRC numerical parity",
        "qubits":N,"encoding_layers":LAYERS,"seed":SEED,
        "entanglement":"linear CNOT","shots":None,
        "feature_shape":list(F.shape),"absolute_tolerance":ATOL,
        "max_absolute_error":float(diff.max()),
        "mean_absolute_error":float(diff.mean()),
        "parity_passed":passed,"pennylane_version":qml.__version__
    }
    (OUT/"parity_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not passed:
        raise SystemExit("Parity failed. Do not benchmark this port yet.")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--backend",choices=["qiskit","pennylane"],required=True)
    a=ap.parse_args()
    qiskit_run() if a.backend=="qiskit" else pennylane_run()
