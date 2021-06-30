# 2乗和誤差

import numpy as np

# E=1210∑k=1 (yk−tk)2：全ての要素の差を2乗してから和をとることを意味する。その総和に1/2を掛けたものが2乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2) # 「**」はべき乗のこと　ここでは「** 2」なので2乗するということ

# 予想の結果である確率のこと ソフトマックス関数の結果と仮定
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 一番左が0 一番右側が9 なので0.6は2の確立
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

# 正解のラベル 正解を1、それ以外を0で表すことを「one-hot」という
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


print(mean_squared_error(np.array(y1), np.array(t)))
# y1の場合：0.09750000000000003
# y2の場合：0.5975
# この結果の意味は「0」に近づくほど正解に近い、もしくは正解（0なら完全に正解）
# 「1」に近づくほど不正解に近い、もしくは不正解（1なら完全に不正解）

# 本よりも詳しい解説：https://www.anarchive-beta.com/entry/2020/06/16/180000