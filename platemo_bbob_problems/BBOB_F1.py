import numpy as np
from .compat import Problem

class BBOB_F1(Problem):
    """
    BBOB F1: Sphere Function
    鍏紡: f(x) = sum( (x_i - x_opt_i)^2 )
    鐗圭偣: 鍗曞嘲銆佸绉般€佸彲鍒嗙銆?
    """
    def __init__(self, **kwargs):
        # 棰勮 xopt 鍙傛暟锛堝搴?MATLAB 鐨?obj.ParameterSet(0)锛?
        # 濡傛灉 kwargs 涓病鏈変紶 xopt锛岄粯璁や娇鐢?0
        self.xopt_val = kwargs.get('xopt', 0)
        super().__init__(**kwargs)

    def setting(self):
        self.M = 1
        # 濡傛灉缁村害 D 鏈寚瀹氾紝榛樿涓?30
        if self.D is None:
            self.D = 30

        # 澶勭悊 xopt 鐨勭淮搴﹀榻?
        if np.isscalar(self.xopt_val):
            # 濡傛灉鏄爣閲忥紝骞抽摵涓?(D,) 鍚戦噺
            self.x_opt = np.full(self.D, self.xopt_val)
        else:
            # 濡傛灉鏄暟缁勶紝寮哄埗鍖归厤 D
            self.x_opt = np.asarray(self.xopt_val)
            self.D = len(self.x_opt)

        # 璁剧疆杈圭晫 [-5, 5]
        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)

    def cal_obj(self, X):
        """
        璁＄畻鐩爣鍊?
        X: 鍐崇瓥鍙橀噺鐭╅樀 (N, D)
        """
        # 1. 瀵瑰簲 MATLAB 鐨?X = X - linspace(0, obj.upper(1), obj.D)
        # 鐢熸垚鍋忕Щ鍚戦噺 (D,)

        # 2. 瀵瑰簲 MATLAB 鐨?Z = X - repmat(obj.xopt, size(X,1), 1)
        # NumPy 鐨勫箍鎾満鍒朵細鑷姩澶勭悊 repmat 鐨勯€昏緫锛屼笉闇€瑕佹墜鍔ㄥ钩閾?
        Z = X - self.x_opt

        # 3. 瀵瑰簲 MATLAB 鐨?PopObj = sum(Z.^2, 2)
        # 璁＄畻娆у紡璺濈鐨勫钩鏂瑰拰
        pop_obj = np.sum(Z**2, axis=1)

        # 杩斿洖 (N, 1) 鐭╅樀
        return pop_obj.reshape(-1, self.M)

