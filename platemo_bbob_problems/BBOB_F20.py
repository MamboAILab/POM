import numpy as np
from .compat import Problem


class BBOB_F20(Problem):
    """
    BBOB F20: Schwefel Function
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30
        state = np.random.RandomState(1)
        self.x_opt = 4.2096874633 / 2 * np.sign(state.rand(self.D) - 0.5)
        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)

    def cal_obj(self, X):
        N, D = X.shape
        Xs = X
        Xopt = np.tile(self.x_opt, (N, 1))
        Xtrans = 2 * np.sign(Xopt) * Xs
        Z = np.zeros_like(Xtrans)
        Z[:, 0] = Xtrans[:, 0]
        Z[:, 1:] = Xtrans[:, 1:] + 0.25 * (Xtrans[:, :-1] - 2 * np.abs(Xopt[:, :-1]))
        scaling_diag = 10 ** (0.5 * np.arange(D) / (D - 1))
        Z = 100 * ((Z - 2 * np.abs(Xopt)) @ np.diag(scaling_diag) + 2 * np.abs(Xopt))
        pop_obj = -0.01 * np.mean(Z * np.sin(np.sqrt(np.abs(Z))), axis=1)
        pop_obj += 4.189828872724339 + 100 * self.fpen(Z / 100)
        return pop_obj.reshape(-1, self.M)

    @staticmethod
    def fpen(X):
        return np.sum(np.maximum(0, np.abs(X) - 5) ** 2, axis=1)

