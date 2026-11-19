import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qiskit")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.quantum_reservoir import QuantumReservoir


def test_quantum_reservoir_is_reproducible_fixed_and_has_expected_shape():
    features = np.array([[0.1, -0.2, 0.3, -0.4], [0.4, 0.3, -0.2, 0.1]])
    first = QuantumReservoir(encoding_layers=2, seed=17)
    second = QuantumReservoir(encoding_layers=2, seed=17)

    first_values = first.transform(features)
    np.testing.assert_array_equal(first_values, second.transform(features))
    assert first_values.shape == (2, 30)
    assert first.configuration()["trainable_reservoir_parameters"] is False


def test_quantum_reservoir_depths_share_seeded_parameters_and_expand_features():
    values = np.array([[0.2, -0.1, 0.3, 0.5]])
    shallow = QuantumReservoir(encoding_layers=2, seed=23)
    deep = QuantumReservoir(encoding_layers=6, seed=23)

    np.testing.assert_array_equal(
        shallow.rotation_parameters, deep.rotation_parameters
    )
    assert shallow.transform(values).shape == (1, 30)
    assert deep.transform(values).shape == (1, 90)


def test_quantum_reservoir_validates_dimensions_and_entanglement():
    with pytest.raises(ValueError, match="exactly four qubits"):
        QuantumReservoir(n_qubits=3)
    with pytest.raises(ValueError, match="entanglement"):
        QuantumReservoir(entanglement="ring")
    with pytest.raises(ValueError, match="one column per qubit"):
        QuantumReservoir().transform(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        QuantumReservoir().transform(np.full((1, 4), np.nan))
