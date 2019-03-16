import numpy as np

from math import sqrt


class Function:
    """
    Abstract model of a differentiable function.
    """
    def __init__(self, *args, **kwargs):
        # initializing cache for intermediate results
        # helps with gradient calculation in some cases
        self.cache = {}
        # cache for gradients
        self.grad = {}

    def __call__(self, *args, **kwargs):
        # calculating output
        output = self.forward(*args, **kwargs)
        # calculating and caching local gradients
        self.grad = self.local_grad(*args, **kwargs)
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

    def local_grad(self, *args, **kwargs):
        """
        Calculates the local gradients of the function at the given input.

        Returns:
            grad: dictionary of local gradients.
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
        self.weight_update = {}

    def _init_weights(self, *args, **kwargs):
        pass

    def _update_weights(self, lr):
        """
        Updates the weights using the corresponding _global_ gradients computed during
        backpropagation.

        Args:
             lr: float. Learning rate.
        """
        for weight_key, weight in self.weight.items():
            self.weight[weight_key] = self.weight[weight_key] - lr * self.weight_update[weight_key]
        pass


class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._init_weights(in_dim, out_dim)

    def _init_weights(self, in_dim, out_dim):
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

        output = np.dot(X, self.weight['W']) + self.weight['b']

        # caching variables for backprop
        self.cache['X'] = X
        self.cache['output'] = output

        return output

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
        dX = dY.dot(self.grad['X'].T)
        # calculating the global gradient wrt to weights
        X = self.cache['X']
        dW = self.grad['W'].T.dot(dY)
        db = np.sum(dY, axis=0, keepdims=True)
        # caching the global gradients
        self.weight_update = {'W': dW, 'b': db}

        return dX

    def local_grad(self, X):
        """
        Local gradients of the Linear layer at X.

        Args:
            X: numpy.ndarray of shape (n_batch, in_dim) containing the
                input data.

        Returns:
            grads: dictionary of local gradients with the following items:
                X: numpy.ndarray of shape (n_batch, in_dim).
                W: numpy.ndarray of shape (n_batch, in_dim).
                b: numpy.ndarray of shape (n_batch, 1).
        """
        gradX_local = self.weight['W']
        gradW_local = X
        gradb_local = np.ones_like(self.weight['b'])
        grads = {'X': gradX_local, 'W': gradW_local, 'b': gradb_local}
        return grads


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) \
                           else (kernel_size, kernel_size)
        self.padding = padding
        self._init_weights(in_channels, out_channels, kernel_size)

    def _init_weights(self, in_channels, out_channels, kernel_size):
        scale = 2/sqrt(in_channels*kernel_size[0]*kernel_size[0])

        self.weight = {'W': np.random.normal(scale=scale,
                                             size=(out_channels, in_channels, *kernel_size)),
                       'b': np.zeros(shape=(out_channels, 1))}
