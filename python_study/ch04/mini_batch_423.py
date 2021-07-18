# ミニバッチとは大量データから無作為に対象をピックアップし分析すること
# テレビの視聴率なんかもそれにあたる（任意の1000世帯みたいな）
# こうすることで正確な結果は得られないが効率よくおおよその結果を得られる

import sys, os
# sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize = True, one_hot_label = True)

# print(x_train.shape)
# print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch)
print(t_batch)