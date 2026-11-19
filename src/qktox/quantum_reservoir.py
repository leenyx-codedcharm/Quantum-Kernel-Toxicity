"""Fixed gate-based quantum reservoir feature maps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


@dataclass
class QuantumReservoir:
    """Map fixed-width molecular vectors to fixed quantum-reservoir observables."""

    n_qubits: int = 4
    encoding_layers: int = 4
    seed: int = 20261119
    entanglement: str = "linear"
    max_encoding_layers: int = 6
    rotation_parameters: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_qubits != 4:
            raise ValueError("The initial QM-QRC pilot requires exactly four qubits")
        if self.encoding_layers not in (2, 4, 6):
            raise ValueError("encoding_layers must be one of 2, 4, or 6")
        if self.max_encoding_layers < self.encoding_layers:
            raise ValueError("max_encoding_layers cannot be smaller than encoding_layers")
        if self.entanglement not in {"none", "linear", "full"}:
            raise ValueError("entanglement must be 'none', 'linear', or 'full'")
        rng = np.random.default_rng(self.seed)
        self.rotation_parameters = rng.uniform(
            -np.pi,
            np.pi,
            size=(self.max_encoding_layers, self.n_qubits, 3),
        )

    @property
    def features_per_layer(self) -> int:
        """Return X, Y, Z single-qubit expectations and adjacent ZZ correlations."""
        return 3 * self.n_qubits + self.n_qubits - 1

    @property
    def output_dimension(self) -> int:
        return self.features_per_layer * self.encoding_layers

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Encode rows repeatedly and collect observables after each fixed step."""
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_qubits:
            raise ValueError("features must be a 2-D matrix with one column per qubit")
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("features must be non-empty and finite")

        observables = _reservoir_observables(self.n_qubits)
        output = np.empty((len(values), self.output_dimension), dtype=np.float64)
        for row_index, row in enumerate(values):
            circuit = QuantumCircuit(self.n_qubits)
            for layer in range(self.encoding_layers):
                circuit.compose(self._step_circuit(row, layer), inplace=True)
                state = Statevector.from_instruction(circuit)
                start = layer * self.features_per_layer
                output[row_index, start : start + self.features_per_layer] = [
                    float(np.real(state.expectation_value(observable)))
                    for observable in observables
                ]
        return output

    def circuit_for(self, features: np.ndarray) -> QuantumCircuit:
        """Build the full assigned circuit for one input row."""
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (self.n_qubits,) or not np.isfinite(values).all():
            raise ValueError("one finite input value per qubit is required")
        circuit = QuantumCircuit(self.n_qubits)
        for layer in range(self.encoding_layers):
            circuit.compose(self._step_circuit(values, layer), inplace=True)
        return circuit

    def _step_circuit(self, features: np.ndarray, layer: int) -> QuantumCircuit:
        circuit = QuantumCircuit(self.n_qubits)
        for qubit, value in enumerate(np.tanh(features)):
            circuit.ry(float(np.pi * value), qubit)
            circuit.rz(float(0.5 * np.pi * value), qubit)
        for qubit, angles in enumerate(self.rotation_parameters[layer]):
            circuit.rx(float(angles[0]), qubit)
            circuit.ry(float(angles[1]), qubit)
            circuit.rz(float(angles[2]), qubit)
        _append_entanglement(circuit, self.entanglement)
        return circuit

    def configuration(self) -> dict[str, object]:
        """Return the exact fixed-circuit specification and random parameters."""
        return {
            "architecture": "gate_based_quantum_reservoir",
            "qubits": self.n_qubits,
            "encoding_layers": self.encoding_layers,
            "max_encoding_layers": self.max_encoding_layers,
            "seed": self.seed,
            "entanglement": self.entanglement,
            "input_encoding": "RY(pi*tanh(x_i)); RZ(0.5*pi*tanh(x_i))",
            "reservoir_step": "fixed random RX-RY-RZ per qubit, then selected entanglers",
            "observables_per_step": [
                "X_i",
                "Y_i",
                "Z_i",
                "Z_i Z_(i+1) for adjacent qubits",
            ],
            "rotation_parameters": self.rotation_parameters.tolist(),
            "trainable_reservoir_parameters": False,
        }


def _reservoir_observables(n_qubits: int) -> list[SparsePauliOp]:
    observables = [
        SparsePauliOp.from_sparse_list([(axis, [qubit], 1.0)], n_qubits)
        for axis in ("X", "Y", "Z")
        for qubit in range(n_qubits)
    ]
    observables.extend(
        SparsePauliOp.from_sparse_list([("ZZ", [qubit, qubit + 1], 1.0)], n_qubits)
        for qubit in range(n_qubits - 1)
    )
    return observables


def _append_entanglement(circuit: QuantumCircuit, entanglement: str) -> None:
    if entanglement == "none":
        return
    if entanglement == "linear":
        for qubit in range(circuit.num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        return
    for control in range(circuit.num_qubits):
        for target in range(control + 1, circuit.num_qubits):
            circuit.cz(control, target)
