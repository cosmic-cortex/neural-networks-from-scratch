import numpy as np

from math import sqrt


class Layer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        self.input = x
        self.output = self.forward(x)
        return self.output

    def forward(self, x):
        pass

    def backward(self, x, Dy):
        pass

    def init_weights(self, *args, **kwargs):
        pass

    def update_weights(self, *args, **kwargs):
        pass

    def gradX(self, x):
        pass

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

    def update_weights(self, *args, **kwargs):
        pass

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
