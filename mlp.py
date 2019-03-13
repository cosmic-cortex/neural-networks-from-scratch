import numpy as np

from nn.layers import *
from nn.losses import MeanSquareLoss, CrossEntropyLoss
from nn.net import Net

# generating some data
X = np.concatenate((np.random.multivariate_normal([-1, -1], [[0.1, 0], [0, 0.1]], size=100),
                    np.random.multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]], size=100)))
Y_labels = np.array([0]*100 + [1]*100)

net = Net(layers=[Linear(2, 2)],
          loss=CrossEntropyLoss())

n_epochs = 100
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

