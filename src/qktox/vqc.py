"""Four-qubit variational quantum classifier construction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.circuit import ParameterVector, QuantumCircuit


def build_vqc_circuits(
    qubits: int = 4,
    feature_repetitions: int = 1,
    feature_entanglement: str = "linear",
    ansatz_repetitions: int = 2,
    ansatz_entanglement: str = "linear",
) -> dict[str, Any]:
    """Build a four-qubit ZZ encoder and RealAmplitudes ansatz."""
    if qubits != 4:
        raise ValueError("The initial VQC pilot requires exactly four qubits")
    feature_map = zz_feature_map(
        feature_dimension=qubits,
        reps=feature_repetitions,
        entanglement=feature_entanglement,
    )
    ansatz = real_amplitudes(
        num_qubits=qubits,
        reps=ansatz_repetitions,
        entanglement=ansatz_entanglement,
    )
    circuit = feature_map.compose(ansatz)
    two_qubit_names = {
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.num_qubits == 2
    }
    return {
        "feature_map": feature_map,
        "ansatz": ansatz,
        "circuit": circuit,
        "feature_map_parameter_count": len(feature_map.parameters),
        "ansatz_parameter_count": len(ansatz.parameters),
        "total_parameter_count": len(circuit.parameters),
        "qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "gate_count": sum(circuit.count_ops().values()),
        "two_qubit_gate_count": sum(
            circuit.count_ops().get(name, 0)
            for name in two_qubit_names
        ),
    }


def count_vqc_circuit_properties(feature_map: Any, ansatz: Any) -> dict[str, Any]:
    """Return reproducible circuit size and parameter properties."""
    circuit = feature_map.compose(ansatz)
    two_qubit_names = {
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.num_qubits == 2
    }
    return {
        "qubit_count": circuit.num_qubits,
        "depth": circuit.depth(),
        "gate_count": sum(circuit.count_ops().values()),
        "two_qubit_gate_count": sum(
            circuit.count_ops().get(name, 0)
            for name in two_qubit_names
        ),
        "feature_map_parameter_count": len(feature_map.parameters),
        "ansatz_parameter_count": len(ansatz.parameters),
        "total_parameter_count": len(circuit.parameters),
    }


def build_data_reuploading_circuit(
    qubits: int = 4,
    reupload_layers: int = 2,
    entanglement: str = "linear",
) -> dict[str, Any]:
    """Build a shallow four-qubit data-reuploading feature and ansatz circuit."""
    if qubits not in (2, 4):
        raise ValueError("Data re-uploading supports exactly two or four qubits")
    if reupload_layers not in (2, 4, 6):
        raise ValueError("Data re-uploading layers must be 2, 4, or 6")
    if entanglement not in ("none", "linear", "full"):
        raise ValueError("Entanglement must be none, linear, or full")

    feature_parameters = ParameterVector("x", qubits)
    trainable_parameters = ParameterVector("theta", qubits * reupload_layers)
    circuit = QuantumCircuit(qubits)
    parameter_index = 0
    for layer in range(reupload_layers):
        for qubit in range(qubits):
            circuit.ry(feature_parameters[qubit], qubit)
            circuit.rz(trainable_parameters[parameter_index], qubit)
            parameter_index += 1
        if entanglement == "linear" and qubits > 1:
            for qubit in range(qubits - 1):
                circuit.cx(qubit, qubit + 1)
        elif entanglement == "full" and qubits > 1:
            for source in range(qubits):
                for target in range(source + 1, qubits):
                    circuit.cx(source, target)

    two_qubit_gate_count = sum(
        count
        for name, count in circuit.count_ops().items()
        if name in {"cx", "cz", "ecr", "swap"}
    )
    return {
        "circuit": circuit,
        "feature_parameters": list(feature_parameters),
        "trainable_parameters": list(trainable_parameters),
        "qubits": qubits,
        "reupload_layers": reupload_layers,
        "entanglement": entanglement,
        "feature_map_parameter_count": len(feature_parameters),
        "ansatz_parameter_count": len(trainable_parameters),
        "total_parameter_count": len(circuit.parameters),
        "depth": circuit.depth(),
        "gate_count": sum(circuit.count_ops().values()),
        "two_qubit_gate_count": two_qubit_gate_count,
    }
