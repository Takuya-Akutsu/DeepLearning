# この項で利用するライブラリを読み込む
import numpy as np
import matplotlib.pyplot as plt

# RMSPorpの実装
class RMSProp:

    # インスタンス変数を定義
    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr # 学習率
        self.decay_rate = decay_rate # 減衰率
        self.h = None # 過去の勾配の2乗和
    
    # パラメータの更新メソッドを定義
    def update(self, params, grads):
        # hの初期化
        if self.h is None: # 初回のみ
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val) # 全ての要素が0
        
        # パラメータの値を更新
        for key in params.keys():
            self.h[key] *= self.decay_rate # 式(1)の前の項
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key] # 式(1)の後の項
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) # 式(2)

# 式(6.2)
def f(x, y):
    return x ** 2 / 20.0 + y ** 2

# 式(6.2)の勾配(偏微分)
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
# plt.contour(X, Y, Z) # 等高線
# plt.plot(0, 0, '+') # 最小値の点
# plt.xlim(-10, 10) # x軸の範囲
# plt.ylim(-10, 10) # y軸の範囲
# plt.xlabel("x") # x軸ラベル
# plt.ylabel("y") # y軸ラベル
# plt.title("$f(x, y) = \\frac{1}{20} x^2 + y^2$", fontsize=20) # タイトル
# plt.show()

# パラメータの初期値を指定
params = {}
params['x'] = -7.0
params['y'] = 2.0

# 勾配の初期値を指定
grads = {}
grads['x'] = 0
grads['y'] = 0

# 学習率を指定
lr = 0.1

# 減衰率
decay_rate = 0.99

# インスタンスを作成
optimizer = RMSProp(lr=lr, decay_rate=decay_rate)

# 試行回数を指定
iter_num = 30

# 更新値の記録用リストを初期化
x_history = []
y_history = []

# 初期値を保存
x_history.append(params['x'])
y_history.append(params['y'])

# 関数の最小値を探索
for _ in range(iter_num):
    
    # 勾配を計算
    grads['x'], grads['y'] = df(params['x'], params['y'])
    
    # パラメータを更新
    optimizer.update(params, grads)
    
    # パラメータを記録
    x_history.append(params['x'])
    y_history.append(params['y'])

# 作図
plt.plot(x_history, y_history, 'o-') # パラメータの推移
plt.contour(X, Y, Z) # 等高線
plt.plot(0, 0, '+') # 最小値の点
plt.xlim(-10, 10) # t軸の範囲
plt.ylim(-10, 10) # y軸の範囲
plt.xlabel("t") # t軸ラベル
plt.ylabel("y") # y軸ラベル
plt.title("MRSProp", fontsize=20) # タイトル
plt.text(6, 6, "$\\eta=$" + str(lr) + "\n$\\beta=$" + str(decay_rate) + "\niteration:" + str(iter_num)) # メモ
plt.show()