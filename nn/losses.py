import numpy as np
from .layers import Function


class Loss(Function):
    def forward(self, x, y):
        """
        Computes the loss of x with respect to y.

        Args:
            x: numpy.ndarray of shape (n_batch, n_dim).
            y: numpy.ndarray of shape (n_batch, n_dim).

        Returns:
            loss: numpy.float.
        """
        pass

    def backward(self):
        """
        Backward pass for the MeanSquareLoss function. Since it should be the final layer
        of an architecture, no input is needed for the backward pass.

        Returns:
            gradX: numpy.ndarray of shape (n_batch, n_dim). Local gradient of the MSE loss.
        """
        return self.gradX_local

    def gradX(self, x, y):
        """
        Local gradient with respect to x at (x, y).

        Args:
            x: numpy.ndarray of shape (n_batch, n_dim).
            y: numpy.ndarray of shape (n_batch, n_dim).

        Returns:
            gradX: numpy.ndarray of shape (n_batch, n_dim).
        """
        pass


class MeanSquareLoss(Loss):
    def forward(self, x, y):
        """
        Computes the mean square error of x with respect to y.

        Args:
            x: numpy.ndarray of shape (n_batch, n_dim).
            y: numpy.ndarray of shape (n_batch, n_dim).

        Returns:
            mse: numpy.float. Mean square error of x with respect to y.
        """
        sum = np.sum((x - y)**2, axis=1, keepdims=True)
        mse_loss = np.mean(sum)
        return mse_loss

    def gradX(self, x, y):
        """
        Local gradient with respect to x at (x, y).

        Args:
            x: numpy.ndarray of shape (n_batch, n_dim).
            y: numpy.ndarray of shape (n_batch, n_dim).

        Returns:
            gradX: numpy.ndarray of shape (n_batch, n_dim). Gradient of MSE wrt X at x and y.
        """
        return 2*(x - y)/x.shape[0]
