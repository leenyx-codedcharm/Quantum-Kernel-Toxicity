from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    balanced_accuracy_score, accuracy_score, matthews_corrcoef,
    precision_score, recall_score, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/quantum_benchmarks/herg_karim/pennylane_research/full_pilot"
OUT = ROOT / "results/quantum_benchmarks/herg_karim/pennylane_research/hybrid_vqc_adam"
N_QUBITS = 4
N_LAYERS = 2
SEEDS = (20261119, 20261120)
EPOCHS = 100
PATIENCE = 18
LEARNING_RATE = 0.025
SHOTS = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_pilot(include_test_labels=False):
    feature_path = PILOT / "preprocessed_inputs.npz"
    ids_path = PILOT / "molecule_ids.json"
    processed_path = ROOT / "data/processed/herg_karim.csv"
    for path in (feature_path, ids_path, processed_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. First run the full-pilot feature generation script."
            )
    z = np.load(feature_path)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    X = {p: np.asarray(z[f"{p}_X"], dtype=np.float64) for p in ("train", "val", "test")}
    requested = ("train", "val", "test") if include_test_labels else ("train", "val")
    y = {}
    for partition in requested:
        row_indices = np.asarray(
            [int(molecule_id.rsplit("_", 1)[1]) for molecule_id in ids[partition]],
            dtype=np.int64,
        )
        selected = set(map(int, row_indices))
        labels = pd.read_csv(
            processed_path,
            usecols=["label"],
            skiprows=lambda line: line != 0 and line - 1 not in selected,
        )["label"]
        values = pd.to_numeric(labels, errors="coerce")
        if len(values) != len(row_indices) or values.isna().any():
            raise ValueError(f"Could not load valid {partition} labels.")
        y[partition] = values.to_numpy(dtype=np.int64)
    for partition in ("train", "val", "test"):
        if X[partition].shape != (len(ids[partition]), N_QUBITS):
            raise ValueError(f"{partition} feature shape does not match frozen IDs.")
        if not np.isfinite(X[partition]).all():
            raise ValueError(f"Non-finite {partition} features.")
        if partition in y and set(np.unique(y[partition])) != {0, 1}:
            raise ValueError(f"Invalid {partition} labels.")
    return X, y, ids


def make_qnode():
    dev = qml.device("lightning.qubit", wires=N_QUBITS, shots=SHOTS)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(x, theta):
        # PennyLane-native data re-uploading VQC.
        # Each layer re-encodes the four PCA inputs, applies trainable rotations,
        # then entangles neighbouring qubits.
        for layer in range(N_LAYERS):
            for wire in range(N_QUBITS):
                qml.RY(x[wire], wires=wire)
                qml.RZ(x[wire], wires=wire)
                qml.Rot(
                    theta[layer, wire, 0],
                    theta[layer, wire, 1],
                    theta[layer, wire, 2],
                    wires=wire,
                )
            for wire in range(N_QUBITS - 1):
                qml.CNOT(wires=[wire, wire + 1])

        outputs = [qml.expval(qml.PauliZ(wires=w)) for w in range(N_QUBITS)]
        outputs += [
            qml.expval(qml.PauliZ(wires=w) @ qml.PauliZ(wires=w + 1))
            for w in range(N_QUBITS - 1)
        ]
        return tuple(outputs)

    return circuit


def sigmoid(x):
    return 1.0 / (1.0 + pnp.exp(-pnp.clip(x, -30.0, 30.0)))


def select_threshold(y, probabilities):
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scores = [balanced_accuracy_score(y, probabilities >= t) for t in candidates]
    best = max(range(len(candidates)),
               key=lambda i: (scores[i], -abs(float(candidates[i]) - 0.5), -float(candidates[i])))
    return float(candidates[best]), float(scores[best])


def make_predictor(circuit):
    def predict(X, theta, head, bias):
        rows = []
        for x in X:
            quantum_features = pnp.stack(circuit(x, theta))
            rows.append(sigmoid(pnp.dot(head, quantum_features) + bias))
        return pnp.stack(rows)
    return predict


def validation_metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "positive_f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "AUROC": float(roc_auc_score(y, prob)),
        "AUPRC": float(average_precision_score(y, prob)),
    }


def run_select():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {OUT}")
    OUT.mkdir(parents=True)
    X, y, ids = load_pilot()
    circuit = make_qnode()
    predict = make_predictor(circuit)
    optimizer = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    candidates = []
    best_global = None
    start_all = time.perf_counter()

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        theta = pnp.array(rng.uniform(-0.25, 0.25, size=(N_LAYERS, N_QUBITS, 3)),
                          requires_grad=True)
        head = pnp.array(rng.normal(0.0, 0.08, size=2 * N_QUBITS - 1),
                         requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)
        y_train = pnp.array(y["train"], dtype=float)
        x_train = pnp.array(X["train"], dtype=float)

        def loss_fn(t, w, b):
            probs = predict(x_train, t, w, b)
            eps = 1e-7
            return -pnp.mean(y_train * pnp.log(probs + eps) +
                             (1.0 - y_train) * pnp.log(1.0 - probs + eps))

        best_seed = None
        best_auc = -np.inf
        stale = 0
        history = []
        seed_start = time.perf_counter()

        for epoch in range(1, EPOCHS + 1):
            theta, head, bias = optimizer.step(loss_fn, theta, head, bias)
            val_prob = np.asarray(predict(pnp.array(X["val"]), theta, head, bias), dtype=float)
            val_auc = float(roc_auc_score(y["val"], val_prob))
            val_loss = float(-np.mean(
                y["val"] * np.log(np.clip(val_prob, 1e-7, 1.0)) +
                (1 - y["val"]) * np.log(np.clip(1 - val_prob, 1e-7, 1.0))
            ))
            history.append({"epoch": epoch, "train_loss": float(loss_fn(theta, head, bias)),
                            "validation_auroc": val_auc, "validation_log_loss": val_loss})

            if val_auc > best_auc + 1e-6:
                best_auc = val_auc
                stale = 0
                best_seed = {
                    "theta": np.asarray(theta).copy(),
                    "head": np.asarray(head).copy(),
                    "bias": float(bias),
                    "epoch": epoch,
                    "validation_probabilities": val_prob.copy(),
                    "validation_auroc": val_auc,
                    "validation_log_loss": val_loss,
                }
            else:
                stale += 1

            if epoch == 1 or epoch % 10 == 0:
                print(f"seed={seed} epoch={epoch} train_loss={history[-1]['train_loss']:.4f} "
                      f"val_AUROC={val_auc:.4f}", flush=True)
            if stale >= PATIENCE:
                break

        threshold, val_bacc = select_threshold(y["val"], best_seed["validation_probabilities"])
        seed_record = {
            "seed": seed,
            "best_epoch": best_seed["epoch"],
            "epochs_run": len(history),
            "validation_auroc": best_seed["validation_auroc"],
            "validation_log_loss": best_seed["validation_log_loss"],
            "validation_balanced_accuracy_at_selected_threshold": val_bacc,
            "selected_threshold": threshold,
            "runtime_seconds": time.perf_counter() - seed_start,
        }
        candidates.append(seed_record)
        (OUT / f"training_history_seed_{seed}.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        if best_global is None or best_seed["validation_auroc"] > best_global["validation_auroc"]:
            best_global = {
                **best_seed, "seed": seed, "threshold": threshold,
                "validation_balanced_accuracy": val_bacc,
            }

    np.savez_compressed(
        OUT / "selected_model.npz",
        theta=best_global["theta"], head=best_global["head"],
        bias=np.asarray(best_global["bias"]),
    )
    lock = {
        "status": "validation_locked_test_not_evaluated",
        "architecture": "PennyLane hybrid data-reuploading VQC",
        "dataset": "herg_karim",
        "split": "scaffold_r0",
        "qubits": N_QUBITS,
        "reupload_layers": N_LAYERS,
        "entanglement": "linear CNOT",
        "observables": "four local Z plus three adjacent ZZ",
        "readout": "jointly trained linear sigmoid head",
        "optimizer": "PennyLane Adam",
        "learning_rate": LEARNING_RATE,
        "max_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "shots": SHOTS,
        "seed_candidates": candidates,
        "selected_seed": best_global["seed"],
        "selected_epoch": best_global["epoch"],
        "validation_auroc": best_global["validation_auroc"],
        "validation_threshold": best_global["threshold"],
        "validation_balanced_accuracy": best_global["validation_balanced_accuracy"],
        "sample_sizes": {p: len(y[p]) for p in ("train", "val", "test")},
        "qm_descriptors_included": False,
        "test_labels_accessed_for_selection": False,
        "runtime_seconds": time.perf_counter() - start_all,
        "pennylane_version": qml.__version__,
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print("\nSELECTION COMPLETE. TEST SET NOT EVALUATED.")
    print(json.dumps(lock, indent=2))


def run_evaluate():
    lock_path = OUT / "selection_lock.json"
    model_path = OUT / "selected_model.npz"
    if not lock_path.exists() or not model_path.exists():
        raise FileNotFoundError("Run --phase select first.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "validation_locked_test_not_evaluated":
        raise ValueError("Selection lock is not in the expected state.")
    X, y, ids = load_pilot(include_test_labels=True)
    model = np.load(model_path)
    circuit = make_qnode()
    predict = make_predictor(circuit)
    prob = np.asarray(predict(pnp.array(X["test"]), pnp.array(model["theta"]),
                              pnp.array(model["head"]), float(model["bias"])), dtype=float)
    threshold = float(lock["validation_threshold"])
    metrics = validation_metrics(y["test"], prob, threshold)
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y["test"], pred, labels=[0, 1])
    pd = __import__("pandas")
    pd.DataFrame({
        "mol_id": ids["test"], "y_true": y["test"],
        "p_pred": prob, "y_pred": pred,
    }).to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame(cm, index=["actual_negative", "actual_positive"],
                 columns=["predicted_negative", "predicted_positive"]).to_csv(
                     OUT / "test_confusion_matrix.csv"
                 )
    report = {
        "status": "test_evaluated_after_validation_lock",
        "selection_lock_sha256": sha256(lock_path),
        "selected_seed": lock["selected_seed"],
        "selected_epoch": lock["selected_epoch"],
        "validation_auroc": lock["validation_auroc"],
        "validation_threshold": threshold,
        "test_metrics": metrics,
        "test_molecules": len(y["test"]),
        "qm_descriptors_included": False,
    }
    (OUT / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nFINAL TEST RESULTS")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["select", "evaluate"], required=True)
    args = parser.parse_args()
    if args.phase == "select":
        run_select()
    else:
        run_evaluate()


if __name__ == "__main__":
    main()
