import numpy as np
import softmax_function as sm_f

a = np.array([0.3, 2.9, 4.0])
y = sm_f.softmax(a)

# print(y)

print(np.sum(y))