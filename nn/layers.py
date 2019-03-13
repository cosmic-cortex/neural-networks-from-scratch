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
        # cache for gradients
        self.grad = {}

    def __call__(self, *args, **kwargs):
        # calculating output
        output = self.forward(*args, **kwargs)
        # caching
        self.cache['output'] = output
        self.grad['X'] = self.gradX(*args, **kwargs)
        return output

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = {}

    def init_weights(self, *args, **kwargs):
        pass

    def update_weights(self, lr):
        pass

    def gradW(self, X):
        pass


class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.init_weights(in_dim, out_dim)

    def init_weights(self, in_dim, out_dim):
        scale = 1 / sqrt(in_dim)
        self.weight['W'] = scale * np.random.randn(in_dim, out_dim)
        self.weight['b'] = scale * np.random.randn(1, out_dim)

    def forward(self, X):
        """
        Forward pass for the Linear layer.

        Args:
            X: numpy.ndarray of shape (n_batch, in_dim) containing
                the input value.

        Returns:
            Y: numpy.ndarray of shape of shape (n_batch, out_dim) containing
                the output value.
        """
        # caching variables for backprop
        self.cache['X'] = X
        return np.dot(X, self.weight['W']) + self.weight['b']

    def backward(self, dY):
        """
        Backward pass for the Linear layer.

        Args:
            dY: numpy.ndarray of shape (n_batch, n_out). Global gradient
                backpropagated from the next layer.

        Returns:
            dX: numpy.ndarray of shape (n_batch, n_out). Global gradient
                of the Linear layer.
        """
        # calculating the global gradient, to be propagated backwards
        dX = dY.dot(self.weight['W'].T)
        # calculating the gradient wrt to weights
        self.grad['W'], self.grad['b'] = self.gradW(self.cache['X'])
        return dX

    def gradX(self, X):
        """
        Local gradient of the Linear layer at X.

        Args:
            X: numpy.ndarray of shape (n_batch, in_dim) containing the
                input data.

        Returns:
            gradX: numpy.ndarray of shape (n_batch, in_dim), containing
                the local gradient at X.
        """
        return self.weight['W']

    def gradW(self, X):
        """
        Gradient of the Linear layer with respect to the weights.

        Args:
            X: numpy.ndarray of shape (n_batch, in_dim) containing the
                input data.

        Returns:
            gradW: numpy.ndarray of shape (n_batch, in_dim).
            gradb: numpy.ndarray of shape (n_batch, 1).
        """
        gradW = X
        gradb = np.ones_like(self.weight['b'])
        return gradW, gradb


class Sigmoid(Function):
    def forward(self, X):
        return sigmoid(X)

    def backward(self, dY):
        return dY * self.grad['X']

    def gradX(self, X):
        return sigmoid_prime(X)


if __name__ == '__main__':
    linear = Linear(5, 2)
