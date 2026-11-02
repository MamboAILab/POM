import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F15(Problem):
    """
    BBOB F15: Rastrigin Function
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
        Z = self.Tasy(0.2, Z)
        scaling_diag = 10 ** (0.5 * np.arange(D) / (D - 1))
        Z = Z @ (self.R @ np.diag(scaling_diag) @ self.Q)
        pop_obj = 10 * (D - np.sum(np.cos(2 * np.pi * Z), axis=1)) + np.sum(Z ** 2, axis=1)
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
    def Tasy(beta, X):
        X = np.asarray(X).copy()
        D = X.shape[1]
        if D == 1:
            return X
        dim_seq = np.arange(D) / (D - 1)
        X2 = X ** (1 + beta * dim_seq * np.sqrt(np.maximum(X, 0)))
        X[X > 0] = X2[X > 0]
        return X

