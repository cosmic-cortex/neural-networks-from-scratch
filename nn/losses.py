class Loss:
    """
    Abstract model of a loss function.
    """
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x, y):
        pass

    def backward(self):
        pass

    def gradX(self, x, y):
        pass


class MeanSquareLoss(Loss):
    def forward(self, x, y):
        """
        Computes the mean square error of x with respect to y.

        Args:
            x: numpy.ndarray of shape (n_batch, 1). Should be the output of a
                Linear layer.
            y: numpy.ndarray of shape (n_batch, 1).

        Returns:
            mse: numpy.ndarray of shape (1, 1). Mean square error of x with respect
                to y.
        """
        return np.mean((x - y)**2)

    def gradX(self, x, y):
        """
        Local gradient with respect to x at (x, y).

        Args:
            x: numpy.ndarray of shape (n_batch, 1). Should be the output of a
                Linear layer.
            y: numpy.ndarray of shape (n_batch, 1).

        Returns:
            gradX: numpy.ndarray of shape (n_batch, 1). Gradient of MSE wrt X at x and y.
        """
        return 2*(x - y)/x.shape[1]
