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
        Backward pass for the loss function. Since it should be the final layer
        of an architecture, no input is needed for the backward pass.

        Returns:
            gradX: numpy.ndarray of shape (n_batch, n_dim). Local gradient of the loss.
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
            mse_loss: numpy.float. Mean square error of x with respect to y.
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


class CrossEntropyLoss(Loss):
    def forward(self, x, y):
        """
        Computes the cross entropy loss of x with respect to y.

        Args:
            x: numpy.ndarray of shape (n_batch, n_dim).
            y: numpy.ndarray of shape (n_batch, 1). Should contain class labels
                for each data point in x.

        Returns:
            crossentropy_loss: numpy.float. Cross entropy loss of x with respect to y.
        """
        # calculating crossentropy
        exp_x = np.exp(x)
        probs = exp_x/np.sum(exp_x, axis=1, keepdims=True)
        log_probs = -np.log(probs[:, y])
        crossentropy_loss = np.mean(log_probs)

        # caching for backprop
        self.cache['probs'] = probs
        self.cache['y'] = y

        return crossentropy_loss

    def gradX(self, x, y):
        probs = self.cache['probs']
        ones = np.zeros_like(probs)
        for row_idx, col_idx in enumerate(y):
            ones[row_idx, col_idx] = 1.0
        return (probs - ones)/len(x)
