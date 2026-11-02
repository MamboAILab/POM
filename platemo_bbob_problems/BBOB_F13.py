import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F13(Problem):
    """
    BBOB F13: Sharp Ridge Function
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
        scaling_diag = 10 ** (0.5 * np.arange(D) / (D - 1))
        M_trans = self.Q @ np.diag(scaling_diag) @ self.R
        Z = (X - self.x_opt) @ M_trans
        pop_obj = Z[:, 0] ** 2 + 100 * np.sqrt(np.sum(Z[:, 1:] ** 2, axis=1))
        return pop_obj.reshape(-1, self.M)

