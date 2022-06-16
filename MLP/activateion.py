
def step(a):
    if a <= 0:
        return 0
    else:
        return 1

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

print(step(0))