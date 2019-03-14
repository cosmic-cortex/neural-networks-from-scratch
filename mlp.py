import numpy as np
import matplotlib.pyplot as plt

from nn.layers import *
from nn.losses import CrossEntropyLoss
from nn.activations import ReLU
from nn.net import Net

# generating some data
n_class_size = 100
r = 2
X1_offset = np.random.rand(n_class_size, 2) - 0.5
np.sqrt(np.sum(X1_offset**2, axis=1, keepdims=True))
X1_offset = r * X1_offset/np.sqrt(np.sum(X1_offset**2, axis=1, keepdims=True))
X1 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size=n_class_size) + X1_offset
X2 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size=n_class_size)

X = np.concatenate((X1, X2))
Y_labels = np.array([0]*n_class_size + [1]*n_class_size)

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 10))
    plt.scatter(X1[:, 0], X1[:, 1], c='r')
    plt.scatter(X2[:, 0], X2[:, 1], c='b')
    plt.show()

net = Net(layers=[Linear(2, 4), ReLU(), Linear(4, 2)],
          loss=CrossEntropyLoss())

n_epochs = 1000
for epoch_idx in range(n_epochs):
    print("Epoch no. %d" % epoch_idx)
    out = net(X)
    # prediction accuracy
    pred = np.argmax(out, axis=1)
    print("accuracy: %1.4f" % (1 - np.abs(pred - Y_labels).sum()/200))
    loss = net.loss(out, Y_labels)
    print('loss: %1.4f' % loss)
    grad = net.backward()
    net.update_weights(0.01)

