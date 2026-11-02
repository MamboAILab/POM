import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F22(Problem):
    """
    BBOB F22: Gallagher's Gaussian 21-hi Peaks Function
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30
        self.x_opt = 4.2096874633 / 2 * np.sign(np.random.RandomState(1).rand(self.D) - 0.5)
        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)
        alpha = 1000 ** (2 * np.arange(20) / 19)
        perm = np.random.RandomState(2).permutation(20)
        self.alpha = np.r_[1e6, alpha[perm]]
        self.C = np.array([
            np.diag(a ** (0.5 * np.arange(self.D) / (self.D - 1))) / (a ** 0.25)
            for a in self.alpha
        ])
        self.y = np.r_[
            np.random.RandomState(3).rand(1) * 7.84 - 3.92,
            np.random.RandomState(4).rand(20) * 9.8 - 4.9,
        ]
        self.R = orth(np.random.RandomState(2).rand(self.D, self.D))

    def cal_obj(self, X):
        N, D = X.shape
        Xs = X
        w = np.r_[10.0, 1.1 + 8 * np.arange(20) / 19]
        temp = np.zeros((N, len(self.C)))
        Rt = self.R.T
        for i in range(len(self.C)):
            diff = Xs - self.y[i]
            transformed = diff @ Rt @ self.C[i] @ self.R
            temp[:, i] = w[i] * np.exp(-0.5 * np.mean(transformed * diff, axis=1))
        pop_obj = self.Tosz(10 - np.max(temp, axis=1)) ** 2 + self.fpen(Xs)
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

