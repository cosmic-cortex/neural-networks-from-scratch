import numpy as np
from .functional import *
from .layers import Function


class Sigmoid(Function):
    def forward(self, X):
        return sigmoid(X)

    def backward(self, dY):
        return dY * self.grad["X"]

    def local_grad(self, X):
        grads = {"X": sigmoid_prime(X)}
        return grads


class ReLU(Function):
    def forward(self, X):
        return relu(X)

    def backward(self, dY):
        return dY * self.grad["X"]

    def local_grad(self, X):
        grads = {"X": relu_prime(X)}
        return grads


class LeakyReLU(Function):
    def forward(self, X):
        return leaky_relu(X)

    def backward(self, dY):
        return dY * self.grad["X"]

    def local_grad(self, X):
        grads = {"X": leaky_relu_prime(X)}
        return grads


class Softmax(Function):
    def forward(self, X):
        exp_x = np.exp(X)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.cache["X"] = X
        self.cache["output"] = probs
        return probs

    def backward(self, dY):
        dX = []

        for dY_row, grad_row in zip(dY, self.grad["X"]):
            dX.append(np.dot(dY_row, grad_row))

        return np.array(dX)

    def local_grad(self, X):
        grad = []

        for prob in self.cache["output"]:
            prob = prob.reshape(-1, 1)
            grad_row = -np.dot(prob, prob.T)
            grad_row_diagonal = prob * (1 - prob)
            np.fill_diagonal(grad_row, grad_row_diagonal)
            grad.append(grad_row)

        grad = np.array(grad)
        return {"X": grad}
