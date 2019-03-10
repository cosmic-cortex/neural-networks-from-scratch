import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - x)
