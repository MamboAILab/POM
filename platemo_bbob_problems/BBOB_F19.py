import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F19(Problem):
    """
    BBOB F19: Composite Griewank-Rosenbrock Function F8F2
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
        Z = max(1.0, np.sqrt(D) / 8.0) * ((X - self.x_opt) @ self.R) + 0.5
        S = 100 * (Z[:, :-1] ** 2 - Z[:, 1:]) ** 2 + (Z[:, :-1] - 1) ** 2
        pop_obj = 10 * np.mean(S / 4000 - np.cos(S), axis=1) + 10
        return pop_obj.reshape(-1, self.M)

