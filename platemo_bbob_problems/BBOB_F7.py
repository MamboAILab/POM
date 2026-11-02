import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F7(Problem):
    """
    BBOB F7: Step Ellipsoidal Function (闃舵妞悆鍑芥暟)
    鐗圭偣: 鍖呭惈璁稿绂绘暎闃舵骞冲彴锛屾搴﹀眬閮ㄤ负0銆?
    娴嬭瘯绠楁硶: 鍦ㄧ鏁?涓嶈繛缁櫙瑙備腑鐨勫叏灞€鎼滅储鑳藉姏銆?
    """

    def __init__(self, **kwargs):
        self.xopt_val = kwargs.get('xopt', 0)
        super().__init__(**kwargs)

    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        # 1. 鍒濆鍖?x_opt
        if np.isscalar(self.xopt_val):
            self.x_opt = np.full(self.D, self.xopt_val)
        else:
            self.x_opt = np.asarray(self.xopt_val)
            self.D = len(self.x_opt)

        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)

        # 2. 鐢熸垚姝ｄ氦鏃嬭浆鐭╅樀 Q 鍜?R
        state_q = np.random.RandomState(1)
        state_r = np.random.RandomState(2)
        self.Q = orth(state_q.rand(self.D, self.D))
        self.R = orth(state_r.rand(self.D, self.D))

    def cal_obj(self, X):
        N, D = X.shape

        diff = X - self.x_opt
        # Lambda 鏄?D x D 瀵硅闃碉紝瀵硅绾垮厓绱犱负 10^(0.5 * (i-1)/(D-1))
        exponent_lambda = 0.5 * np.arange(D) / (D - 1)
        Lambda = np.diag(10 ** exponent_lambda)

        # Z 褰㈢姸涓?(N, D)
        Z = diff @ (Lambda @ self.R)

        # 3. 闃舵鎿嶄綔 (Step operation) -> Zh
        Zh = np.floor(0.5 + 10 * Z) / 10
        # 瀵瑰簲 MATLAB: Zh(abs(Z)>0.5) = floor(0.5+Z(abs(Z)>0.5))
        mask = np.abs(Z) > 0.5
        Zh[mask] = np.floor(0.5 + Z[mask])

        # 4. 鍐嶆鏃嬭浆 Z_rot = Z * Q
        Z_rot = Z @ self.Q

        # 5. 璁＄畻鐩爣鍑芥暟缁勪欢
        # term1: |Zh_1| / 10000
        term1 = np.abs(Zh[:, 0]) / 1.0e4

        # term2: sum( 10^(2*(i-1)/(D-1)) * Z_rot_i^2 )
        exponent_term2 = 2 * np.arange(D) / (D - 1)
        weights_term2 = 10 ** exponent_term2
        term2 = np.sum(weights_term2 * (Z_rot ** 2), axis=1)

        # 6. 鏈€缁堣绠? 0.1 * max(term1, term2) + penalty
        step_ellipsoidal = 0.1 * np.maximum(term1, term2)
        penalty = self.fpen(X)

        pop_obj = step_ellipsoidal + penalty
        return pop_obj.reshape(-1, self.M)

    @staticmethod
    def fpen(X):
        """杈圭晫鎯╃綒鍑芥暟"""
        return np.sum(np.maximum(0, np.abs(X) - 5) ** 2, axis=1)

