import numpy as np
import matplotlib.pyplot as plt

# 式(6.2)
def f(x, y):
    return x ** 2 / 20.0 + y ** 2

# 式(6.2)の勾配
def df(x, y):
    # 偏微分
    dx = x / 10.0 # df / dx
    dy = 2.0 * y # df / dy
    return dx, dy # (値を2つ出力！)

# 等高線用の値
x = np.arange(-10, 10, 0.01) # x軸の値
y = np.arange(-5, 5, 0.01) # y軸の値
X, Y = np.meshgrid(x, y) # 格子状の点に変換
Z = f(X, Y)

# 作図
plt.contour(X, Y, Z) # 等高線
plt.plot(0, 0, '+') # 最小値の点
plt.xlim(-10, 10) # x軸の範囲
plt.ylim(-10, 10) # y軸の範囲
plt.xlabel("x") # x軸ラベル
plt.ylabel("y") # y軸ラベル
plt.title("$f(x, y) = \\frac{1}{20} x^2 + y^2$", fontsize=20) # タイトル
plt.show()