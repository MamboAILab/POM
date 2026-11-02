import numpy as np
from .compat import Problem


class BBOB_F5(Problem):
    """
    BBOB F5: Linear Slope (绾挎€у潯闈㈠嚱鏁?
    鐗圭偣: 绾嚎鎬с€佹棤灞€閮ㄦ瀬鍊笺€佹渶浼樿В浣嶄簬杈圭晫椤剁偣銆?
    娴嬭瘯绠楁硶: 杈圭晫鎼滅储鑳藉姏銆?
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        # BBOB F5 鐨?xopt 鏄敱绗﹀彿鍐冲畾鐨勯《鐐?(+5 鎴?-5)
        # 涓轰簡淇濊瘉鍙噸澶嶆€э紝浣跨敤鍥哄畾鐨勯殢鏈虹瀛愭ā鎷?MATLAB 鐨?RandStream
        state = np.random.RandomState(1)
        self.x_opt = 5 * np.sign(state.rand(self.D) - 0.5)

        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)

    def cal_obj(self, X):
        """
        璁＄畻鐩爣鍊?
        """
        N, D = X.shape

        # 1. 鍩虹浣嶇Щ (涓庡叾浠?BBOB 闂淇濇寔涓€鑷?

        # 2. 澶勭悊 Z 閫昏緫: 濡傛灉 X_opt * X >= 25锛屽垯 Z = X_opt
        # 杩欏搴?MATLAB 鐨?Z(Xopt.*X>=25) = Xopt(Xopt.*X>=25)
        # 鎰忔€濇槸濡傛灉褰撳墠瑙ｅ凡缁忚揪鍒版垨瓒呰繃浜嗘渶浼樿竟鐣岋紝姊害灏变笉鍐嶅彉鍖?
        Z = X.copy()
        mask = (self.x_opt * X) >= 25
        Z[mask] = np.tile(self.x_opt, (N, 1))[mask]

        # 3. 璁＄畻缂╂斁鍥犲瓙 S: sign(xopt) * 10^((i-1)/(D-1))
        exponent = np.arange(D) / (D - 1)
        S = np.sign(self.x_opt) * (10 ** exponent)

        # 4. 璁＄畻鐩爣鍑芥暟: sum(5 * abs(S) - S * Z)
        # MATLAB: sum(5*abs(S)-S.*Z, 2)
        # S (D,) 浼氳嚜鍔ㄥ箍鎾埌 Z (N, D)
        pop_obj = np.sum(5 * np.abs(S) - S * Z, axis=1)

        return pop_obj.reshape(-1, self.M)

