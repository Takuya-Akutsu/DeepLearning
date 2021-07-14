# 必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 偏微分の関数
def function_2(x):
    return x[0]**2+x[1]**2

fig = plt.figure()
ax = Axes3D(fig)

X = np.arange(-3.0, 3.0, 0.25)
Y = np.arange(-3.0, 3.0, 0.25)

X, Y = np.meshgrid(X, Y)
Z = X ** 2 +Y ** 2

# 軸ラベル
ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_zlabel("f(x)")

# 表示する領域の範囲
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
# ax.set_zlim(0, 18)

#角度
ax.view_init(25, -120)
ax.plot_wireframe(X, Y, Z)

# 表示
plt.show()