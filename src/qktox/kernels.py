"""Molecular similarity kernels shared by classical and quantum pilots."""

from __future__ import annotations

import numpy as np


def tanimoto_kernel_matrix(
    query: np.ndarray,
    reference: np.ndarray,
    block_size: int = 256,
) -> np.ndarray:
    """Build a dense binary Tanimoto kernel in row blocks.

    The result occupies 8 * n_query * n_reference bytes. A temporary float32
    product is limited to block_size * n_reference values.
    """
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("Tanimoto inputs must be 2-D matrices with the same bit dimension")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not np.isin(query, (0, 1)).all() or not np.isin(reference, (0, 1)).all():
        raise ValueError("Tanimoto inputs must contain only binary fingerprint values")
    query_float = np.asarray(query, dtype=np.float32)
    reference_float = np.asarray(reference, dtype=np.float32)
    query_counts = query_float.sum(axis=1, dtype=np.float32)
    reference_counts = reference_float.sum(axis=1, dtype=np.float32)
    result = np.empty((len(query_float), len(reference_float)), dtype=np.float64)
    for start in range(0, len(query_float), block_size):
        stop = min(start + block_size, len(query_float))
        intersections = query_float[start:stop] @ reference_float.T
        unions = (
            query_counts[start:stop, None]
            + reference_counts[None, :]
            - intersections
        )
        np.divide(intersections, unions, out=intersections, where=unions > 0)
        intersections[unions == 0] = 1.0
        result[start:stop] = intersections
    return result
