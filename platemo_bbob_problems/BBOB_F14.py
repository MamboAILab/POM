import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F14(Problem):
    """
    BBOB F14: Different Powers Function
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
        Z = (X - self.x_opt) @ self.R
        powers = 2 + 4 * np.arange(D) / (D - 1)
        pop_obj = np.sqrt(np.sum(np.abs(Z) ** powers, axis=1))
        return pop_obj.reshape(-1, self.M)

