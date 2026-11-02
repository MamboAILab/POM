import numpy as np
from .compat import Problem


class BBOB_F3(Problem):
    """
    BBOB F3: Rastrigin Function (鎷夋柉鐗归噷閲戝嚱鏁?
    鐗圭偣: 澶氬嘲銆侀珮搴︽尟鑽°€佸叿鏈夋槑鏄剧殑闈炲绉版€с€?
    鍖呭惈浜?Tosz (鎸崱鍙樻崲) 鍜?Tasy (闈炲绉板彉鎹?銆?
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

        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)

    def cal_obj(self, X):
        """
        璁＄畻鐩爣鍊?
        """
        diff = X - self.x_opt

        # 4. 鎵ц缁勫悎鍙樻崲: Z = Lambda * Tasy(beta, Tosz(diff))
        # Tosz: 鎸崱鍙樻崲
        z_osz = self.Tosz(diff)
        # Tasy: 闈炲绉板彉鎹?(beta = 0.2)
        z_asy = self.Tasy(0.2, z_osz)

        # Lambda: 鎸囨暟鏉冮噸 (10^(0.5 * (i-1)/(D-1)))
        exponent = 0.5 * np.arange(self.D) / (self.D - 1)
        Lambda = 10 ** exponent
        Z = Lambda * z_asy

        # 5. Rastrigin 鍑芥暟鍏紡璁＄畻
        # f(x) = 10 * (D - sum(cos(2*pi*Z))) + sum(Z^2)
        part1 = 10 * (self.D - np.sum(np.cos(2 * np.pi * Z), axis=1))
        part2 = np.sum(Z ** 2, axis=1)

        pop_obj = part1 + part2

        return pop_obj.reshape(-1, self.M)

    @staticmethod
    def Tosz(X):
        """BBOB 鎸崱鍙樻崲"""
        X = np.asarray(X).copy()
        mask_nonzero = (X != 0)
        Xh = np.zeros_like(X)
        Xh[mask_nonzero] = np.log(np.abs(X[mask_nonzero]))
        C1 = np.where(X > 0, 10.0, 5.5)
        C2 = np.where(X > 0, 7.9, 3.1)
        return np.sign(X) * np.exp(Xh + 0.049 * (np.sin(C1 * Xh) + np.sin(C2 * Xh)))

    @staticmethod
    def Tasy(beta, X):
        """BBOB transform."""
        X = np.asarray(X)
        D = X.shape[1]
        if D == 1:
            return X

        # 璁＄畻鎸囨暟閮ㄥ垎
        # 瀵瑰簲 MATLAB: 1 + beta * (0:D-1)/(D-1) * sqrt(|X|)
        dim_seq = np.arange(D) / (D - 1)
        exponent = 1 + beta * dim_seq * np.sqrt(np.maximum(X, 0))

        # 鍙樻崲鏍稿績锛氫粎瀵?X > 0 鐨勯儴鍒嗗簲鐢ㄥ箓杩愮畻
        # 娉ㄦ剰锛歁ATLAB 鐗堜唬鐮佹渶鍚庣畝鍖栦负 X = X2; 鎴戜滑闇€瑕佹纭鐞嗙鍙?
        X2 = X ** exponent
        X[X > 0] = X2[X > 0]
        return X

