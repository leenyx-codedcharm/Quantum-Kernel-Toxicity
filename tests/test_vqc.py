import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qiskit")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.vqc import build_data_reuploading_circuit, build_vqc_circuits
from qktox.trainable_kernel import build_trainable_fidelity_kernel
from qktox.projected_quantum import hybrid_kernel, projected_pauli_representation


def test_four_qubit_vqc_circuits_are_four_qubit_with_fixed_ablation():
    circuits = build_vqc_circuits(
        qubits=4,
        feature_repetitions=1,
        feature_entanglement="linear",
        ansatz_repetitions=2,
        ansatz_entanglement="linear",
    )
    assert circuits["qubits"] == 4
    assert circuits["feature_map_parameter_count"] == 4
    assert circuits["ansatz_parameter_count"] == 12
    assert circuits["total_parameter_count"] == 16
    assert circuits["depth"] > 0
    assert circuits["gate_count"] > 0
    assert circuits["two_qubit_gate_count"] > 0


def test_vqc_rejects_non_four_qubit_configuration():
    with np.testing.assert_raises_regex(ValueError, "exactly four qubits"):
        build_vqc_circuits(qubits=3)


def test_data_reuploading_circuit_reports_parameters_and_entanglement():
    pilot = build_data_reuploading_circuit(
        qubits=4,
        reupload_layers=2,
        entanglement="linear",
    )
    assert pilot["qubits"] == 4
    assert pilot["reupload_layers"] == 2
    assert pilot["feature_map_parameter_count"] == 4
    assert pilot["ansatz_parameter_count"] == 8
    assert pilot["total_parameter_count"] == 12
    assert pilot["gate_count"] > 0


def test_projected_pauli_features_and_hybrid_kernel_are_valid():
    features = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    representation = projected_pauli_representation(features)
    assert representation.shape == (2, 7)
    assert np.isfinite(representation).all()

    tanimoto = np.array([[1.0, 0.25], [0.25, 1.0]])
    quantum = np.array([[1.0, 0.5], [0.5, 1.0]])
    combined = hybrid_kernel(tanimoto, quantum, alpha=0.5)
    np.testing.assert_allclose(combined, combined.T)
    assert np.linalg.eigvalsh(combined).min() >= -1e-8


def test_trainable_fidelity_kernel_separates_data_and_training_parameters():
    pilot = build_trainable_fidelity_kernel(
        qubits=4,
        reps=1,
        entanglement="linear",
    )
    assert pilot["qubit_count"] == 4
    assert pilot["feature_parameter_count"] == 4
    assert pilot["training_parameter_count"] == 4
    assert pilot["total_parameter_count"] == 8
    assert pilot["kernel"].training_parameters == pilot["training_parameters"]
    assert pilot["depth"] > 0
    assert pilot["gate_count"] > 0
    data = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.8, 0.5, 0.2, 0.7]],
        dtype=np.float64,
    )
    pilot["kernel"].assign_training_parameters(
        dict(zip(pilot["training_parameters"], np.zeros(4)))
    )
    untrained = pilot["kernel"].evaluate(data)
    pilot["kernel"].assign_training_parameters(
        dict(zip(pilot["training_parameters"], np.full(4, 0.7)))
    )
    trained_parameters = pilot["kernel"].evaluate(data)
    assert not np.allclose(untrained, trained_parameters)
