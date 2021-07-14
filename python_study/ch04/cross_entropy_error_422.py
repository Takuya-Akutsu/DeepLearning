# 2乗和誤差

import numpy as np

# E=−∑k tklogyk
def cross_entropy_error(y, t):
    delta = 1e-7
    return  -np.sum(t * np.log(y + delta))

# 予想の結果である確率のこと ソフトマックス関数の結果と仮定
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 一番左が0 一番右側が9 なので0.6は2の確立
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
y3 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# 正解のラベル 正解を1、それ以外を0で表すことを「one-hot」という
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

print(np.log(2))
print(cross_entropy_error(np.array(y1), np.array(t)))
print(cross_entropy_error(np.array(y2), np.array(t)))
print(cross_entropy_error(np.array(y3), np.array(t)))
# y1の場合：0.09750000000000003
# y2の場合：0.5975
# この結果の意味は「0」に近づくほど正解に近い、もしくは正解（0なら完全に正解）
# 「1」に近づくほど不正解に近い、もしくは不正解（1なら完全に不正解）

# 本よりも詳しい解説：https://www.anarchive-beta.com/entry/2020/06/16/180000