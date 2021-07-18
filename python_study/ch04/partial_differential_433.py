import numpy as np

def numerical_diff(f, x):
    h = 1e-4 #0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

def function_tmp1(x0):
    return x0 * x0 + 4.0 **2.0

def function_tmp2(x1):
    return 3.0 **2.0 + x1 * x1

print(numerical_diff(function_tmp1, 3.0))
print(numerical_diff(function_tmp2, 4.0))
# ↓↓↓↓↓↓ 上の引数を渡した場合のnumerical_diffの計算内容 ↓↓↓↓↓↓
# def numerical_diff(function_tmp1, 3.0):
#     h = 1e-4
#     return (function_tmp1(3.0 + 1e-4) - function_tmp1(3.0 - 1e-4)) / (2 * 1e-4)