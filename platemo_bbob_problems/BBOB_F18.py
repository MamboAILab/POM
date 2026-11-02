import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F18(Problem):
    """
    BBOB F18: Ill-Conditioned Schaffers F7 Function
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
        Z = self.Tasy(0.5, (X - self.x_opt) @ self.R)
        scaling_diag = 1000 ** (0.5 * np.arange(D) / (D - 1))
        Z = Z @ (np.diag(scaling_diag) @ self.Q)
        S = np.sqrt(Z[:, :-1] ** 2 + Z[:, 1:] ** 2)
        pop_obj = np.mean(np.sqrt(S) + np.sqrt(S) * np.sin(50 * S ** 0.2) ** 2, axis=1) ** 2
        pop_obj += 10 * self.fpen(X)
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

    @staticmethod
    def fpen(X):
        return np.sum(np.maximum(0, np.abs(X) - 5) ** 2, axis=1)

