import numpy as np
import perceptron_233_AND as AND
import perceptron_233_OR as OR
import perceptron_233_NAND as NAND

def XOR(x1, x2):
    s1 = NAND.NAND(x1, x2)
    s2 = OR.OR(x1, x2)
    y = AND.AND(s1, s2)
    return y

print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))