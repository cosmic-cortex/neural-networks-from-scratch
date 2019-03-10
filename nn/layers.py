import numpy as np

from math import sqrt
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        self.input = x
        self.output = self.forward(x)
        return self.output

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x, Dy):
        pass

    @abstractmethod
    def init_weights(self, *args, **kwargs):
        pass

    @abstractmethod
    def gradX(self, x):
        pass

    @abstractmethod
    def gradW(self, x):
        pass


class Linear(Layer):
    __slots__ = ['weight', 'bias']

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.init_weights(in_dim, out_dim)

    def init_weights(self, in_dim, out_dim):
        scale = 1 / sqrt(in_dim)
        self.weight = scale * (np.random.rand(in_dim, out_dim) - 0.5)
        self.bias = scale * (np.random.rand(1, out_dim) - 0.5)

    def forward(self, x):
        """
        Forward pass for the linear layer.

        Args:
            x: numpy.array of shape (1, in_dim)

        Returns:
            y: numpy.array of shape (1, out_dim)
        """
        return np.dot(x, self.weight) + self.bias

    def backward(self, x, Dy):
        pass

    def gradX(self, x):
        return self.weight

    def gradW(self, x):
        pass


if __name__ == '__main__':
    linear = Linear(5, 2)
