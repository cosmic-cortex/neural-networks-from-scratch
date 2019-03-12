import numpy as np

from math import sqrt

from .functional import sigmoid, sigmoid_prime


class Function:
    """
    Abstract model of a differentiable function.
    """
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x):
        """
        Forward pass with input x. Calculates the output value and the
        gradient at the input as well.

        Args:
            x: numpy.ndarray, the input value.

        Returns:
            y: numpy.ndarray, the output value.
        """
        pass

    def backward(self, dy):
        """
        Backward pass. Computes the local gradient at the input value
        after forward pass.

        Args:
            dy: numpy.ndarray, the upward gradient.

        Returns:
            dX: numpy.ndarray, the _global_ gradient.
        """
        pass

    def gradX(self, x):
        """
        Calculates the local derivative of the function at x.

        Args:
            x: numpy.ndarray, the input data.

        Returns:
            gradX: numpy.ndarray, output of the layer.
        """
        pass


class Layer(Function):
    """
    Abstract model of a neural network layer. In addition to Function, a Layer
    also has weights and gradients with respect to the weights.
    """
    def init_weights(self, *args, **kwargs):
        pass

    def update_weights(self, *args, **kwargs):
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
        self.gradX_local = self.gradX(x)
        self.output = np.dot(x, self.weight) + self.bias
        return self.output

    def backward(self, dy):
        return dy.dot(self.weight.T)

    def gradX(self, x):
        return self.weight

    def gradW(self, x):
        pass


class Sigmoid(Layer):
    def forward(self, x):
        return sigmoid(x)

    def dx(self, x):
        return np.diag(sigmoid_prime(x).reshape(-1))


if __name__ == '__main__':
    linear = Linear(5, 2)
