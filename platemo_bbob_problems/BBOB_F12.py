import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F12(Problem):
    """
    BBOB F12: Bent Cigar Function
    """

    def __init__(self, **kwargs):
        self.xopt_val = kwargs.get('xopt', 0)
        super().__init__(**kwargs)

    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30
        if np.isscalar(self.xopt_val):
            self.x_opt = np.full(self.D, self.xopt_val)
        else:
            self.x_opt = np.asarray(self.xopt_val)
            self.D = len(self.x_opt)
        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)
        self.R = orth(np.random.RandomState(2).rand(self.D, self.D))

    def cal_obj(self, X):
        N, D = X.shape
        Z = self.Tasy(0.5, (X - self.x_opt) @ self.R) @ self.R
        pop_obj = Z[:, 0] ** 2 + 1e6 * np.sum(Z[:, 1:] ** 2, axis=1)
        return pop_obj.reshape(-1, self.M)

    @staticmethod
    def Tasy(beta, X):
        X = np.asarray(X).copy()
        D = X.shape[1]
        if D == 1:
            return X
        dim_seq = np.arange(D) / (D - 1)
        X2 = X ** (1 + beta * dim_seq * np.sqrt(np.maximum(X, 0)))
        X[X > 0] = X2[X > 0]
        return X

