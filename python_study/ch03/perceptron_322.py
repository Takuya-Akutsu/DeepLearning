import numpy as np

x = np.array([-1.0, 1.0, 2.0])
y = x > 0
'''
print(y)
'''

y = y.astype(np.int64)
print(y)