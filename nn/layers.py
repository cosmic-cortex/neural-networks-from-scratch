import numpy as np

from math import sqrt

from .functional import sigmoid, sigmoid_prime


class Function:
    """
    Abstract model of a differentiable function.
    """
    def __init__(self, *args, **kwargs):
        # initializing cache for intermediate results
        # helps with gradient calculation
        self.cache = {}

    def __call__(self, *args, **kwargs):
        self.output = self.forward(*args, **kwargs)
        self.gradX_local = self.gradX(*args, **kwargs)
        return self.output

    def forward(self, *args, **kwargs):
        """
        Forward pass of the function. Calculates the output value and the
        gradient at the input as well.
        """
        pass

    def backward(self, *args, **kwargs):
        """
        Backward pass. Computes the local gradient at the input value
        after forward pass.
        """
        pass

    def gradX(self, *args, **kwargs):
        """
        Calculates the local derivative of the function at the given input.
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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.init_weights(in_dim, out_dim)

    def init_weights(self, in_dim, out_dim):
        scale = 1 / sqrt(in_dim)
        self.W = scale * np.random.randn(in_dim, out_dim)
        self.b = scale * np.random.randn(1, out_dim)

    def update_weights(self, *args, **kwargs):
        pass

    def forward(self, x):
        """
        Forward pass for the Linear layer.

        Args:
            x: numpy.ndarray of shape (n_batch, in_dim) containing
                the input value.

        Returns:
            y: numpy.ndarray of shape of shape (n_batch, out_dim) containing
                the output value.
        """
        return np.dot(x, self.W) + self.b

    def backward(self, dy):
        """
        Backward pass for the Linear layer.

        Args:
            dy: numpy.ndarray of shape (n_batch, n_out). Global gradient
                backpropagated from the next layer.

        Returns:
            dx: numpy.ndarray of shape (n_batch, n_out). Global gradient
                of the Linear layer.
        """
        # calculating the global gradient, to be propagated backwards
        dx = dy.dot(self.W.T)
        return dx

    def gradX(self, x):
        """
        Local gradient of the Linear layer at x.

        Args:
            x: numpy.ndarray of shape (n_batch, in_dim) containing the
                input data.

        Returns:
            gradX: numpy.ndarray of shape (n_batch, in_dim), containing
                the local gradient at x.
        """
        return self.W

    def gradW(self, x):
        """
        Gradient of the Linear layer with respect to the weights.

        Args:
            x: numpy.ndarray of shape (n_batch, in_dim) containing the
                input data.

        Returns:
            gradW: numpy.ndarray of shape (n_batch, in_dim).
            gradb: numpy.ndarray of shape (n_batch, 1).
        """
        gradW = x
        gradb = np.ones_like(self.b)
        return gradW, gradb


class Sigmoid(Function):
    def forward(self, x):
        return sigmoid(x)

    def backward(self, dy):
        return dy*self.gradX_local

    def gradX(self, x):
        return sigmoid_prime(x)


if __name__ == '__main__':
    linear = Linear(5, 2)
