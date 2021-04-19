from .losses import Loss
from .layers import Function, Layer


class Net:
    __slots__ = ["layers", "loss_fn"]

    def __init__(self, layers, loss):
        assert isinstance(loss, Loss), "loss must be an instance of nn.losses.Loss"
        for layer in layers:
            assert isinstance(layer, Function), (
                "layer should be an instance of " "nn.layers.Function or nn.layers.Layer"
            )

        self.layers = layers
        self.loss_fn = loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        """
        Calculates the forward pass by propagating the input through the
        layers.

        Args:
            x: numpy.ndarray. Input of the net.

        Returns:
            output: numpy.ndarray. Output of the net.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, x, y):
        """
        Calculates the loss of the forward pass output with respect to y.
        Should be called after forward pass.

        Args:
            x: numpy.ndarray. Output of the forward pass.
            y: numpy.ndarray. Ground truth.

        Returns:
            loss: numpy.float. Loss value.
        """
        loss = self.loss_fn(x, y)
        return loss

    def backward(self):
        """
        Complete backward pass for the net. Should be called after the forward
        pass and the loss are calculated.

        Returns:
            d: numpy.ndarray of shape matching the input during forward pass.
        """
        d = self.loss_fn.backward()
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d

    def update_weights(self, lr):
        """
        Updates the weights for all layers using the corresponding gradients
        computed during backpropagation.

        Args:
             lr: float. Learning rate.
        """
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer._update_weights(lr)
