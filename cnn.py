import numpy as np

from nn.layers import *
from nn.losses import CrossEntropyLoss
from nn.activations import ReLU
from nn.net import Net

from keras.datasets import mnist

net = Net(layers=[Conv2D(1, 4, 3, padding=1), MaxPool2D(kernel_size=2), ReLU(), BatchNorm2D(4),
                  Conv2D(4, 8, 3, padding=1), MaxPool2D(kernel_size=2), ReLU(), BatchNorm2D(8),
                  Flatten(), Linear(8*7*7, 10)],
          loss=CrossEntropyLoss())

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshaping
X_train, X_test = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
# normalizing and scaling data
X_train, X_test = X_train.astype('float32')/255, X_test.astype('float32')/255

n_epochs = 100
n_batch = 1000
for epoch_idx in range(n_epochs):
    batch_idx = np.random.choice(range(len(X_train)), size=n_batch, replace=False)
    out = net(X_train[batch_idx])
    loss = net.loss(out, y_train[batch_idx])
    net.backward()
    net.update_weights(lr=0.01)
    print("Epoch no. %d loss =  %2f4" % (epoch_idx, loss))
