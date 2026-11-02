import numpy as np
from .compat import orth
from .compat import Problem


class BBOB_F6(Problem):
    """
    BBOB F6: Attractive Sector Function (寮曞姏鎵囧尯鍑芥暟)
    鐗圭偣: 楂樺害闈炲绉般€佹棆杞€﹀悎銆佸叿鏈夋瀬闄″抄鐨勮剨绾裤€?
    娴嬭瘯绠楁硶: 澶勭悊闈炲绉版€т互鍙婇潪鍧愭爣杞村榻愮殑鎼滅储绌洪棿鐨勮兘鍔涖€?
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

        # 2. 鐢熸垚姝ｄ氦鏃嬭浆鐭╅樀 Q 鍜?R (瀵瑰簲 MATLAB 鐨?orth(rand(...)))
        # 浣跨敤鍥哄畾鐨勯殢鏈虹瀛愪繚璇佸彲閲嶅鎬?
        state_q = np.random.RandomState(1)
        state_r = np.random.RandomState(2)

        # scipy.linalg.orth 浣滅敤绛夊悓浜?MATLAB 鐨?orth
        self.Q = orth(state_q.rand(self.D, self.D))
        self.R = orth(state_r.rand(self.D, self.D))

    def cal_obj(self, X):
        """
        璁＄畻鐩爣鍊?
        """
        N, D = X.shape

        # 1. 鍩虹浣嶇Щ (Standard BBOB shift)

        # 2. 鍧愭爣杞崲 Z = (X - Xopt) * (Q * D_scaling * R)
        # 璁＄畻涓棿缂╂斁瀵硅闃?
        scaling_diag = 10 ** (0.5 * np.arange(D) / (D - 1))
        # 缁勫悎杞崲鐭╅樀 M_trans = Q @ diag @ R
        M_trans = self.Q @ np.diag(scaling_diag) @ self.R

        # 璁＄畻 Z (N x D)
        Z = (X - self.x_opt) @ M_trans

        # 3. 寮曞姏鎵囧尯缂╂斁 S
        # 濡傛灉 Z_i * xopt_i > 0锛屽垯璇ョ淮搴︾殑缂╂斁鍥犲瓙涓?100锛屽惁鍒欎负 1
        S = np.ones((N, D))
        # 瀵瑰簲 MATLAB: S(Z.*Xopt>0) = 100
        # 鍒╃敤骞挎挱鏈哄埗瀵规瘮 Z 鍜?self.x_opt
        mask = (Z * self.x_opt) > 0
        S[mask] = 100.0

        # 4. 璁＄畻鍔犳潈骞虫柟鍜?
        # MATLAB: sum((S.*Z).^2, 2)
        sum_sq = np.sum((S * Z) ** 2, axis=1)

        # 5. 搴旂敤 Tosz 鍙樻崲鍜?0.9 娆″箓
        pop_obj = self.Tosz(sum_sq) ** 0.9

        return pop_obj.reshape(-1, self.M)

    @staticmethod
    def Tosz(X):
        """BBOB transform."""
        X = np.asarray(X)
        res = np.zeros_like(X)
        mask_nonzero = (X != 0)
        # 澶勭悊涓€缁村拰澶氱淮鎯呭喌
        Xh = np.zeros_like(X)
        Xh[mask_nonzero] = np.log(np.abs(X[mask_nonzero]))
        C1 = np.where(X > 0, 10.0, 5.5)
        C2 = np.where(X > 0, 7.9, 3.1)
        res = np.sign(X) * np.exp(Xh + 0.049 * (np.sin(C1 * Xh) + np.sin(C2 * Xh)))
        return res

