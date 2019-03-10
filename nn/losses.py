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

    def gradX(self, x):
        pass


class MeanSquareLoss(Loss):
    def forward(self, x, y):
        return np.sum((x - y)**2)
