import numpy as np

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

print("Qiskit quantum kernel test")

# Four-dimensional input = four qubits
feature_map = zz_feature_map(
    feature_dimension=4,
    reps=1,
    entanglement="linear"
)

kernel = FidelityStatevectorKernel(
    feature_map=feature_map
)

X = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.8, 0.2, 0.5, 0.1],
    [0.3, 0.7, 0.2, 0.9]
])

K = kernel.evaluate(X)

print("Kernel matrix shape:", K.shape)
print("Kernel matrix:")
print(K)

assert K.shape == (4, 4)
assert np.allclose(K, K.T)
assert np.allclose(np.diag(K), 1.0)
assert np.isfinite(K).all()

print("Quantum kernel test PASSED")
