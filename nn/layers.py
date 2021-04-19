import numpy as np

from math import sqrt
from itertools import product

from .utils import zero_pad


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


class Flatten(Function):
    def forward(self, X):
        self.cache["shape"] = X.shape
        n_batch = X.shape[0]
        return X.reshape(n_batch, -1)

    def backward(self, dY):
        return dY.reshape(self.cache["shape"])


class MaxPool2D(Function):
    def __init__(self, kernel_size=(2, 2)):
        super().__init__()
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )

    def __call__(self, X):
        # in contrary to other Function subclasses, MaxPool2D does not need to call
        # .local_grad() after forward pass because the gradient is calculated during it
        return self.forward(X)

    def forward(self, X):
        N, C, H, W = X.shape
        KH, KW = self.kernel_size

        grad = np.zeros_like(X)
        Y = np.zeros((N, C, H // KH, W // KW))

        # for n in range(N):
        for h, w in product(range(0, H // KH), range(0, W // KW)):
            h_offset, w_offset = h * KH, w * KW
            rec_field = X[:, :, h_offset : h_offset + KH, w_offset : w_offset + KW]
            Y[:, :, h, w] = np.max(rec_field, axis=(2, 3))
            for kh, kw in product(range(KH), range(KW)):
                grad[:, :, h_offset + kh, w_offset + kw] = (
                    X[:, :, h_offset + kh, w_offset + kw] >= Y[:, :, h, w]
                )

        # storing the gradient
        self.grad["X"] = grad

        return Y

    def backward(self, dY):
        dY = np.repeat(
            np.repeat(dY, repeats=self.kernel_size[0], axis=2), repeats=self.kernel_size[1], axis=3
        )
        return self.grad["X"] * dY

    def local_grad(self, X):
        # small hack: because for MaxPool calculating the gradient is simpler during
        # the forward pass, it is calculated there and this function just returns the
        # grad dictionary
        return self.grad


class BatchNorm2D(Layer):
    def __init__(self, n_channels, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.n_channels = n_channels
        self._init_weights(n_channels)

    def _init_weights(self, n_channels):
        self.weight["gamma"] = np.ones(shape=(1, n_channels, 1, 1))
        self.weight["beta"] = np.zeros(shape=(1, n_channels, 1, 1))

    def forward(self, X):
        """
        Forward pass for the 2D batchnorm layer.

        Args:
            X: numpy.ndarray of shape (n_batch, n_channels, height, width).

        Returns_
            Y: numpy.ndarray of shape (n_batch, n_channels, height, width).
                Batch-normalized tensor of X.
        """
        mean = np.mean(X, axis=(2, 3), keepdims=True)
        var = np.var(X, axis=(2, 3), keepdims=True) + self.epsilon
        invvar = 1.0 / var
        sqrt_invvar = np.sqrt(invvar)
        centered = X - mean
        scaled = centered * sqrt_invvar
        normalized = scaled * self.weight["gamma"] + self.weight["beta"]

        # caching intermediate results for backprop
        self.cache["mean"] = mean
        self.cache["var"] = var
        self.cache["invvar"] = invvar
        self.cache["sqrt_invvar"] = sqrt_invvar
        self.cache["centered"] = centered
        self.cache["scaled"] = scaled
        self.cache["normalized"] = normalized

        return normalized

    def backward(self, dY):
        """
        Backward pass for the 2D batchnorm layer. Calculates global gradients
        for the input and the parameters.

        Args:
            dY: numpy.ndarray of shape (n_batch, n_channels, height, width).

        Returns:
            dX: numpy.ndarray of shape (n_batch, n_channels, height, width).
                Global gradient wrt the input X.
        """
        # global gradients of parameters
        dgamma = np.sum(self.cache["scaled"] * dY, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dY, axis=(0, 2, 3), keepdims=True)

        # caching global gradients of parameters
        self.weight_update["gamma"] = dgamma
        self.weight_update["beta"] = dbeta

        # global gradient of the input
        dX = self.grad["X"] * dY

        return dX

    def local_grad(self, X):
        """
        Calculates the local gradient for X.

        Args:
            dY: numpy.ndarray of shape (n_batch, n_channels, height, width).

        Returns:
            grads: dictionary of gradients.
        """
        # global gradient of the input
        N, C, H, W = X.shape
        # ppc = pixels per channel, useful variable for further computations
        ppc = H * W

        # gradient for 'denominator path'
        dsqrt_invvar = self.cache["centered"]
        dinvvar = (1.0 / (2.0 * np.sqrt(self.cache["invvar"]))) * dsqrt_invvar
        dvar = (-1.0 / self.cache["var"] ** 2) * dinvvar
        ddenominator = (X - self.cache["mean"]) * (2 * (ppc - 1) / ppc ** 2) * dvar

        # gradient for 'numerator path'
        dcentered = self.cache["sqrt_invvar"]
        dnumerator = (1.0 - 1.0 / ppc) * dcentered

        dX = ddenominator + dnumerator
        grads = {"X": dX}
        return grads


class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._init_weights(in_dim, out_dim)

    def _init_weights(self, in_dim, out_dim):
        scale = 1 / sqrt(in_dim)
        self.weight["W"] = scale * np.random.randn(in_dim, out_dim)
        self.weight["b"] = scale * np.random.randn(1, out_dim)

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

        output = np.dot(X, self.weight["W"]) + self.weight["b"]

        # caching variables for backprop
        self.cache["X"] = X
        self.cache["output"] = output

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
        dX = dY.dot(self.grad["X"].T)
        # calculating the global gradient wrt to weights
        X = self.cache["X"]
        dW = self.grad["W"].T.dot(dY)
        db = np.sum(dY, axis=0, keepdims=True)
        # caching the global gradients
        self.weight_update = {"W": dW, "b": db}

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
        gradX_local = self.weight["W"]
        gradW_local = X
        gradb_local = np.ones_like(self.weight["b"])
        grads = {"X": gradX_local, "W": gradW_local, "b": gradb_local}
        return grads


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.padding = padding
        self._init_weights(in_channels, out_channels, self.kernel_size)

    def _init_weights(self, in_channels, out_channels, kernel_size):
        scale = 2 / sqrt(in_channels * kernel_size[0] * kernel_size[1])

        self.weight = {
            "W": np.random.normal(scale=scale, size=(out_channels, in_channels, *kernel_size)),
            "b": np.zeros(shape=(out_channels, 1)),
        }

    def forward(self, X):
        """
        Forward pass for the convolution layer.

        Args:
            X: numpy.ndarray of shape (N, C, H_in, W_in).

        Returns:
            Y: numpy.ndarray of shape (N, F, H_out, W_out).
        """
        if self.padding:
            X = zero_pad(X, pad_width=self.padding, dims=(2, 3))

        self.cache["X"] = X

        N, C, H, W = X.shape
        KH, KW = self.kernel_size
        out_shape = (N, self.out_channels, 1 + (H - KH) // self.stride, 1 + (W - KW) // self.stride)
        Y = np.zeros(out_shape)
        for n in range(N):
            for c_w in range(self.out_channels):
                for h, w in product(range(out_shape[2]), range(out_shape[3])):
                    h_offset, w_offset = h * self.stride, w * self.stride
                    rec_field = X[n, :, h_offset : h_offset + KH, w_offset : w_offset + KW]
                    Y[n, c_w, h, w] = (
                        np.sum(self.weight["W"][c_w] * rec_field) + self.weight["b"][c_w]
                    )

        return Y

    def backward(self, dY):
        # calculating the global gradient to be propagated backwards
        # TODO: this is actually transpose convolution, move this to a util function
        X = self.cache["X"]
        dX = np.zeros_like(X)
        N, C, H, W = dX.shape
        KH, KW = self.kernel_size
        for n in range(N):
            for c_w in range(self.out_channels):
                for h, w in product(range(dY.shape[2]), range(dY.shape[3])):
                    h_offset, w_offset = h * self.stride, w * self.stride
                    dX[n, :, h_offset : h_offset + KH, w_offset : w_offset + KW] += (
                        self.weight["W"][c_w] * dY[n, c_w, h, w]
                    )

        # calculating the global gradient wrt the conv filter weights
        dW = np.zeros_like(self.weight["W"])
        for c_w in range(self.out_channels):
            for c_i in range(self.in_channels):
                for h, w in product(range(KH), range(KW)):
                    X_rec_field = X[
                        :, c_i, h : H - KH + h + 1 : self.stride, w : W - KW + w + 1 : self.stride
                    ]
                    dY_rec_field = dY[:, c_w]
                    dW[c_w, c_i, h, w] = np.sum(X_rec_field * dY_rec_field)

        # calculating the global gradient wrt to the bias
        db = np.sum(dY, axis=(0, 2, 3)).reshape(-1, 1)

        # caching the global gradients of the parameters
        self.weight_update["W"] = dW
        self.weight_update["b"] = db

        return dX[:, :, self.padding : -self.padding, self.padding : -self.padding]
