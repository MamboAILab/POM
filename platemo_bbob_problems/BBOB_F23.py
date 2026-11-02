import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F23(Problem):
    """
    BBOB F23: Katsuura Function
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
        self.Q = orth(np.random.RandomState(1).rand(self.D, self.D))
        self.R = orth(np.random.RandomState(2).rand(self.D, self.D))

    def cal_obj(self, X):
        N, D = X.shape
        scaling_diag = 100 ** (0.5 * np.arange(D) / (D - 1))
        Z = (X - self.x_opt) @ (self.Q @ np.diag(scaling_diag) @ self.R)
        j = 2.0 ** np.arange(1, 33)
        frac = np.sum(np.abs(j[None, None, :] * Z[:, :, None] - np.round(j[None, None, :] * Z[:, :, None])) / j[None, None, :], axis=2)
        factors = (1 + (np.arange(1, D + 1)[None, :] * frac)) ** (10 / (D ** 1.2))
        pop_obj = 10 / (D ** 2) * np.prod(factors, axis=1) - 10 / (D ** 2) + self.fpen(X)
        return pop_obj.reshape(-1, self.M)

    @staticmethod
    def fpen(X):
        return np.sum(np.maximum(0, np.abs(X) - 5) ** 2, axis=1)

