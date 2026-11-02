import numpy as np
from .compat import Problem


class BBOB_F8(Problem):
    """
    BBOB F8: Rosenbrock Function (缃楁．甯冪綏鍏嬪嚱鏁?
    鐗圭偣: 钁楀悕鐨勨€滈钑夊嚱鏁扳€濓紝鍏锋湁鐙獎寮洸鐨勫北璋枫€?
    娴嬭瘯绠楁硶: 鍦ㄥ己鑰﹀悎缁村害鍜岄潪绾挎€у北璋蜂腑鐨勬悳绱㈣兘鍔涖€?
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

    def cal_obj(self, X):
        """
        璁＄畻鐩爣鍊?
        """
        N, D = X.shape

        # 1. 鍩虹浣嶇Щ (涓庡叾浠?BBOB 闂淇濇寔涓€鑷?

        # 2. BBOB 鐗规湁鐨勫潗鏍囩缉鏀句笌骞崇Щ: Z = max(1, sqrt(D)/8) * (X - xopt) + 1
        # 杩欎釜鍙樻崲灏嗘渶浼樿В浠?xopt 鏄犲皠鍒颁簡鍏?1 鍚戦噺澶?
        scale = max(1.0, np.sqrt(D) / 8.0)
        Z = scale * (X - self.x_opt) + 1.0

        # 3. 璁＄畻 Rosenbrock 鍏紡: sum( 100*(Z_i^2 - Z_{i+1})^2 + (Z_i - 1)^2 )
        # 娉ㄦ剰: i 浠?1 鍒?D-1
        Z_current = Z[:, :-1]  # Z_i
        Z_next = Z[:, 1:]  # Z_{i+1}

        term1 = 100.0 * (Z_current ** 2 - Z_next) ** 2
        term2 = (Z_current - 1.0) ** 2

        pop_obj = np.sum(term1 + term2, axis=1)

        return pop_obj.reshape(-1, self.M)

