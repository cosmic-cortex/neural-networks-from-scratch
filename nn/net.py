class Net:
    __slots__ = ['layers', 'loss']

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self):
        pass
