import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F10(Problem):
    """
    BBOB F10: Rotated Ellipsoidal Function (鏃嬭浆妞悆鍑芥暟)
    鐗圭偣: 楂樺害鐥呮€?(Ill-conditioned)銆佹棆杞€﹀悎銆佸寘鍚潪绾挎€ф尟鑽°€?
    娴嬭瘯绠楁硶: 鍦ㄦ棆杞悗鐨勫潗鏍囩郴涓鐞嗘瀬搴︿笉骞宠　姊害鐨勮兘鍔涖€?
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

        # 2. 鐢熸垚鏃嬭浆鐭╅樀 R (Seed=2)
        state_r = np.random.RandomState(2)
        self.R = orth(state_r.rand(self.D, self.D))

    def cal_obj(self, X):
        """
        璁＄畻鐩爣鍊?
        """
        N, D = X.shape

        # 1. 鍩虹浣嶇Щ

        # 2. 鍧愭爣杞崲涓庢棆杞? Z = Tosz( (X - xopt) * R )
        diff = X - self.x_opt
        rotated_diff = diff @ self.R
        Z = self.Tosz(rotated_diff)

        # 3. 璁＄畻鏉冮噸绯绘暟 10^(6 * (i-1)/(D-1))
        exponent = 6.0 * np.arange(D) / (D - 1)
        weights = 10 ** exponent

        # 4. 鍔犳潈骞虫柟鍜?sum(weights * Z^2)
        # 鍒╃敤 NumPy 骞挎挱鏈哄埗锛寃eights (D,) 浼氫綔鐢ㄤ簬 Z^2 (N, D) 鐨勬瘡涓€琛?
        pop_obj = np.sum(weights * (Z ** 2), axis=1)

        return pop_obj.reshape(-1, self.M)

    @staticmethod
    def Tosz(X):
        """BBOB transform."""
        X = np.asarray(X)
        mask_nonzero = (X != 0)
        Xh = np.zeros_like(X)
        Xh[mask_nonzero] = np.log(np.abs(X[mask_nonzero]))
        C1 = np.where(X > 0, 10.0, 5.5)
        C2 = np.where(X > 0, 7.9, 3.1)
        return np.sign(X) * np.exp(Xh + 0.049 * (np.sin(C1 * Xh) + np.sin(C2 * Xh)))

