import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F16(Problem):
    """
    BBOB F16: Weierstrass Function
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
        Z = self.Tosz((X - self.x_opt) @ self.R)
        scaling_diag = 0.01 ** (0.5 * np.arange(D) / (D - 1))
        Z = Z @ (self.R @ np.diag(scaling_diag) @ self.Q)
        k = np.arange(12)
        f0 = np.sum((0.5 ** k) * np.cos(np.pi * (3 ** k)))
        Z3 = Z[:, :, None]
        k3 = k[None, None, :]
        inner = np.sum((0.5 ** k3) * np.cos(2 * np.pi * (3 ** k3) * (Z3 + 0.5)), axis=2)
        pop_obj = 10 * (np.mean(inner, axis=1) - f0) ** 3 + 10 / D * self.fpen(X)
        return pop_obj.reshape(-1, self.M)

    @staticmethod
    def Tosz(X):
        X = np.asarray(X)
        Xh = np.zeros_like(X)
        mask = X != 0
        Xh[mask] = np.log(np.abs(X[mask]))
        C1 = np.where(X > 0, 10.0, 5.5)
        C2 = np.where(X > 0, 7.9, 3.1)
        return np.sign(X) * np.exp(Xh + 0.049 * (np.sin(C1 * Xh) + np.sin(C2 * Xh)))

    @staticmethod
    def fpen(X):
        return np.sum(np.maximum(0, np.abs(X) - 5) ** 2, axis=1)

