class Loss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x, y):
        self.input = x
        self.output = self.forward(x, y)
        return self.output

    def forward(self, x, y):
        pass

    def backward(self, x):
        pass

    def gradX(self, x, y):
        pass


class MeanSquareLoss(Loss):
    def forward(self, x, y):
        return np.mean((x - y)**2)

    def gradX(self, x, y):
        return 2*(x - y)/x.shape[1]
