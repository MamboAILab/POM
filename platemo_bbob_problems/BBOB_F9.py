import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F9(Problem):
    """
    BBOB F9: Rotated Rosenbrock Function (鏃嬭浆缃楁．甯冪綏鍏嬪嚱鏁?
    鐗圭偣: 鍦?Rosenbrock 鐨勫熀纭€涓婂鍔犱簡鏃嬭浆鍙樻崲锛屼娇鍙橀噺闂翠骇鐢熷叏鑰﹀悎銆?
    娴嬭瘯绠楁硶: 鍦ㄦ棆杞潪绾挎€у北璋蜂腑鐨勫潗鏍囨棤鍏虫悳绱㈣兘鍔涖€?
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

        # 2. 鐢熸垚鏃嬭浆鐭╅樀 R (Seed=2 涓?MATLAB BBOB 鏍囧噯涓€鑷?
        state_r = np.random.RandomState(2)
        # orth 浣跨敤 SVD 鍒嗚В浠庨殢鏈虹煩闃典腑鎻愬彇姝ｄ氦鍩?
        self.R = orth(state_r.rand(self.D, self.D))

    def cal_obj(self, X):
        """
        璁＄畻鐩爣鍊?
        """
        N, D = X.shape

        # 1. 鍩虹浣嶇Щ

        # 2. 鍧愭爣杞崲 Z = max(1, sqrt(D)/8) * (X - xopt) * R + 0.5
        # 娉ㄦ剰锛欶9 鐨勫钩绉婚噺鏄?0.5锛岃€?F8 鏄?1.0
        scale = max(1.0, np.sqrt(D) / 8.0)
        diff = X - self.x_opt

        # 鎵ц鐭╅樀涔樻硶锛堟棆杞級
        Z = scale * (diff @ self.R) + 0.5

        # 3. 璁＄畻 Rosenbrock 鍏紡閮ㄥ垎
        Z_current = Z[:, :-1]  # z_i
        Z_next = Z[:, 1:]  # z_{i+1}

        term1 = 100.0 * (Z_current ** 2 - Z_next) ** 2
        term2 = (Z_current - 1.0) ** 2

        pop_obj = np.sum(term1 + term2, axis=1)

        return pop_obj.reshape(-1, self.M)

