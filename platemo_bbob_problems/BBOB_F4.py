import numpy as np
from .compat import Problem


class BBOB_F4(Problem):
    """
    BBOB F4: B眉che-Rastrigin Function (B眉che-鎷夋柉鐗归噷閲戝嚱鏁?
    鐗圭偣: 澶氬嘲銆侀潪瀵圭О缂╂斁銆佸甫杈圭晫鎯╃綒銆?
    """

    def __init__(self, **kwargs):
        self.xopt_val = kwargs.get('xopt', 0)
        super().__init__(**kwargs)

    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        # 澶勭悊 xopt 缁村害
        if np.isscalar(self.xopt_val):
            self.x_opt = np.full(self.D, self.xopt_val)
        else:
            self.x_opt = np.asarray(self.xopt_val)
            self.D = len(self.x_opt)

        # 鏍囧噯 BBOB 鑼冨洿 [-5, 5]
        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)

    def cal_obj(self, X):
        """
        璁＄畻鐩爣鍊?
        """
        N, D = X.shape

        # 1. 璁＄畻棰濆鍋忕Щ鍚戦噺 c 骞跺簲鐢?

        # 2. 璁＄畻 Z = Tosz(X_true - xopt)
        diff = X - self.x_opt
        Z = self.Tosz(diff)

        # 3. 璁＄畻缂╂斁鐭╅樀 S
        # S_base = 10^(0.5 * (i-1)/(D-1))
        exponent = 0.5 * np.arange(D) / (D - 1)
        S = np.tile(10 ** exponent, (N, 1))

        # 4. 瀵瑰鏁扮淮 (Index 0, 2, 4...) 杩涜闈炲绉扮缉鏀?
        # Matlab: S(:,1:2:end) = 10.^(Z(:,1:2:end)>0).*S(:,1:2:end)
        # 娉ㄦ剰 Python 绱㈠紩浠?0 寮€濮嬶紝1:2:end 瀵瑰簲 0, 2, 4...
        odd_dims = np.arange(0, D, 2)
        Z_odd = Z[:, odd_dims]
        S[:, odd_dims] = (10 ** (Z_odd > 0)) * S[:, odd_dims]

        # 搴旂敤缂╂斁
        Z = S * Z

        # 5. 璁＄畻涓诲嚱鏁伴儴鍒?(Rastrigin)
        rastrigin_part = 10 * (D - np.sum(np.cos(2 * np.pi * Z), axis=1)) + np.sum(Z ** 2, axis=1)

        # 6. 璁＄畻鎯╃綒閮ㄥ垎 (Penalty)
        penalty_part = 100 * self.fpen(X)

        pop_obj = rastrigin_part + penalty_part
        return pop_obj.reshape(-1, self.M)

    @staticmethod
    def Tosz(X):
        """BBOB 鎸崱鍙樻崲"""
        X = np.asarray(X)
        res = np.zeros_like(X)
        mask_nonzero = (X != 0)
        Xh = np.zeros_like(X)
        Xh[mask_nonzero] = np.log(np.abs(X[mask_nonzero]))
        C1 = np.where(X > 0, 10.0, 5.5)
        C2 = np.where(X > 0, 7.9, 3.1)
        res = np.sign(X) * np.exp(Xh + 0.049 * (np.sin(C1 * Xh) + np.sin(C2 * Xh)))
        return res

    @staticmethod
    def fpen(X):
        """杈圭晫鎯╃綒鍑芥暟: sum(max(0, |x|-5)^2)"""
        return np.sum(np.maximum(0, np.abs(X) - 5) ** 2, axis=1)

