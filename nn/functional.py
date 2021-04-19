import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return x * (x > 0)


def relu_prime(x):
    return 1 * (x > 0)


def leaky_relu(x, alpha):
    return x * (x > 0) + alpha * x * (x <= 0)


def leaky_relu_prime(x, alpha):
    return 1 * (x > 0) + alpha * (x <= 0)
