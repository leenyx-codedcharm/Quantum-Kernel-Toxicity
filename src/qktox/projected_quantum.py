"""Projected Pauli-observable quantum representations and hybrid kernels."""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import SparsePauliOp, Statevector


def projected_pauli_representation(
    features: np.ndarray,
    qubits: int = 4,
    entanglement: str = "linear",
) -> np.ndarray:
    """Construct Z and selected ZZ expectation-value quantum features."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != qubits:
        raise ValueError("Projected features must be a 2-D matrix with one column per qubit")
    if not np.isfinite(values).all():
        raise ValueError("Projected features must be finite")
    feature_map = zz_feature_map(
        feature_dimension=qubits,
        reps=1,
        entanglement=entanglement,
    )
    pauli_observables = [
        SparsePauliOp.from_sparse_list(
            [("Z", [index], 1.0)],
            qubits,
        )
        for index in range(qubits)
    ]
    representations = []
    for row in values:
        circuit = feature_map.assign_parameters(dict(zip(feature_map.parameters, row)))
        state = Statevector(circuit)
        z_values = []
        for observable in pauli_observables:
            z_values.append(float(np.real(state.expectation_value(observable))))
        zz_values = []
        for source in range(qubits - 1):
            label = ["I"] * qubits
            label[source] = "Z"
            label[source + 1] = "Z"
            label = ["I"] * qubits
            label[source] = "Z"
            label[source + 1] = "Z"
            observable = SparsePauliOp("".join(label))
            zz_values.append(float(np.real(state.expectation_value(observable))))
        representations.append(z_values + zz_values)
    return np.asarray(representations, dtype=np.float64)


def hybrid_kernel(
    tanimoto: np.ndarray,
    quantum: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return alpha*Tanimoto + (1-alpha)*quantum and validate the Gram matrix."""
    tanimoto_values = np.asarray(tanimoto, dtype=np.float64)
    quantum_values = np.asarray(quantum, dtype=np.float64)
    if tanimoto_values.shape != quantum_values.shape:
        raise ValueError("Hybrid kernel matrices must have identical shapes")
    if tanimoto_values.ndim != 2 or tanimoto_values.shape[0] != tanimoto_values.shape[1]:
        raise ValueError("Hybrid kernel matrices must be square")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Hybrid kernel alpha must be in [0, 1]")
    combined = alpha * tanimoto_values + (1.0 - alpha) * quantum_values
    if not np.isfinite(combined).all():
        raise ValueError("Hybrid kernel contains non-finite values")
    if not np.allclose(combined, combined.T, rtol=0.0, atol=1e-8):
        raise ValueError("Hybrid kernel is not symmetric")
    minimum_eigenvalue = float(np.linalg.eigvalsh(combined).min())
    if minimum_eigenvalue < -1e-8:
        raise ValueError(
            f"Hybrid kernel is not positive semidefinite: {minimum_eigenvalue}"
        )
    return combined
