"""Trainable fidelity-quantum-kernel construction helpers."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import TrainableFidelityStatevectorKernel


def build_trainable_fidelity_kernel(
    qubits: int = 4,
    reps: int = 1,
    entanglement: str = "linear",
) -> dict[str, Any]:
    """Build a fixed ZZ encoding followed by a trainable local rotation layer."""
    if qubits not in (2, 4):
        raise ValueError("Trainable fidelity pilot requires two or four qubits")
    feature_map = zz_feature_map(
        feature_dimension=qubits,
        reps=reps,
        entanglement=entanglement,
    )
    trainable = ParameterVector("theta", qubits)
    circuit = QuantumCircuit(qubits, name="trainable_zz_feature_map")
    for qubit in range(qubits):
        circuit.ry(trainable[qubit], qubit)
    circuit.compose(feature_map, inplace=True)

    # Keep data parameters separate from the training parameters.
    feature_parameters = list(feature_map.parameters)
    training_parameters = list(trainable)
    if any(parameter in feature_parameters for parameter in training_parameters):
        raise ValueError("Data and trainable circuit parameter sets overlap")

    kernel = TrainableFidelityStatevectorKernel(
        feature_map=circuit,
        training_parameters=training_parameters,
        shots=None,
        enforce_psd=True,
    )
    two_qubit_gate_count = sum(
        count
        for name, count in circuit.count_ops().items()
        if circuit.count_ops().get(name, 0) and name in {"cx", "cz", "ecr", "swap"}
    )
    return {
        "feature_map": feature_map,
        "circuit": circuit,
        "kernel": kernel,
        "training_parameters": training_parameters,
        "qubit_count": qubits,
        "feature_parameter_count": len(feature_parameters),
        "training_parameter_count": len(training_parameters),
        "total_parameter_count": len(circuit.parameters),
        "depth": circuit.depth(),
        "gate_count": sum(circuit.count_ops().values()),
        "two_qubit_gate_count": two_qubit_gate_count,
    }
