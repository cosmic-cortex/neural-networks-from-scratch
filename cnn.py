import numpy as np

from nn.layers import *
from nn.losses import CrossEntropyLoss
from nn.activations import ReLU
from nn.net import Net

net = Net(layers=[Conv2D(3, 8, 3, padding=1),  MaxPool2D(kernel_size=2), ReLU(),
                  Flatten(), Linear(8*3*3, 10)],
          loss=CrossEntropyLoss())

X = np.random.rand(5, 3, 6, 6)
Y_labels = np.random.randint(0, 10, size=(2, 1))

out = net(X)
loss = net.loss(out, Y_labels)
net.backward()
net.update_weights(lr=0.01)