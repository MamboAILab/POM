"""The two helpers imported by platemo_bbob_problems; no external framework."""
import numpy as np


class Problem:
    def __init__(self, D=None, M=1, **kwargs):
        self.D = D
        self.M = M
        self.setting()


def orth(matrix):
    """SVD orthonormal column basis, as used by the local BBOB definitions."""
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(singular_values.dtype).eps * singular_values.max(initial=0)
    return u[:, singular_values > tolerance]
