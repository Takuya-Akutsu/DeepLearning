import numpy as np

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a) #指数関数
# print(exp_a)

sum_exp_a = np.sum(exp_a) #指数関数の和
# print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)