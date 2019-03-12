class Net:
    __slots__ = ['layers', 'loss']

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, y):
        # propagating through layers
        for layer in self.layers:
            x = layer(x)
        # calculating the loss
        x = self.loss(x, y)
        return x

    def backward(self):
        dy = self.loss.backward()
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy
