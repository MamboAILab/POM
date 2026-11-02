import numpy as np
from .compat import Problem


class BBOB_F2(Problem):
    """
    BBOB F2: Ellipsoidal Function (妞悆鍑芥暟)
    鐗圭偣: 鍗曞嘲銆侀潪瀵圭О銆佺梾鎬佽皟鑺?(Ill-conditioned)銆?
    寮曞叆浜?Tosz 鍙樻崲鏉ュ鍔犲眬閮ㄦ尟鑽°€?
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
        # 1. 鍩虹浣嶇Щ (Matlab: X = X - linspace(...))

        # 2. 璁＄畻 Z = Tosz(X - xopt)
        Z_input = X - self.x_opt
        Z = self.Tosz(Z_input)

        # 3. 璁＄畻鏉冮噸绯绘暟 10^(6 * (i-1)/(D-1))
        # Matlab: 10.^(6*repmat(0:obj.D-1,size(X,1),1)/(obj.D-1))
        exponent = 6 * np.arange(self.D) / (self.D - 1)
        weights = 10 ** exponent  # 褰㈢姸涓?(D,)

        # 4. 鍔犳潈骞虫柟鍜? sum(weights * Z^2)
        # 鍒╃敤骞挎挱鏈哄埗锛寃eights (D,) 浼氳嚜鍔ㄤ綔鐢ㄤ簬 Z^2 (N, D) 鐨勬瘡涓€琛?
        pop_obj = np.sum(weights * (Z ** 2), axis=1)

        return pop_obj.reshape(-1, self.M)

    @staticmethod
    def Tosz(X):
        """
        BBOB 鏍囧噯闈炵嚎鎬ф尟鑽″彉鎹?Tosz
        """
        X = np.asarray(X)
        res = np.zeros_like(X)

        # 鎵惧嚭闈為浂鍏冪礌绱㈠紩锛岄伩鍏?log(0)
        mask_nonzero = (X != 0)
        Xh = np.zeros_like(X)
        Xh[mask_nonzero] = np.log(np.abs(X[mask_nonzero]))

        # 瀹氫箟绯绘暟 C1, C2
        C1 = np.where(X > 0, 10.0, 5.5)
        C2 = np.where(X > 0, 7.9, 3.1)

        # 鎵ц鍙樻崲: sign(x) * exp(Xh + 0.049 * (sin(C1*Xh) + sin(C2*Xh)))
        res = np.sign(X) * np.exp(Xh + 0.049 * (np.sin(C1 * Xh) + np.sin(C2 * Xh)))

        # 澶勭悊 X=0 鐨勬儏鍐碉紙铏界劧姒傜巼鏋佷綆锛屼絾闇€淇濇寔椴佹鎬э級
        # sign(0) 鏄?0锛屾墍浠?res 鍦?X=0 澶勫凡缁忔槸 0锛岀鍚堝師鎰?
        return res

