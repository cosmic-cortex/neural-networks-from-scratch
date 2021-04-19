import numpy as np

import matplotlib.pyplot as plt

from nn.layers import *
from nn.losses import CrossEntropyLoss
from nn.activations import ReLU, Softmax
from nn.net import Net


# functions for visualization
def plot_data(X1, X2, export_path=None):
    with plt.style.context("seaborn-white"):
        plt.figure(figsize=(10, 10))
        plt.scatter(X1[:, 0], X1[:, 1], c="r", edgecolor="k")
        plt.scatter(X2[:, 0], X2[:, 1], c="b", edgecolor="k")
        plt.title("The data")
        if export_path is None:
            plt.show()
        else:
            plt.savefig(export_path, dpi=500)


def make_grid(X_data, n_res=20):
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    x_meshgrid, y_meshgrid = np.meshgrid(
        np.linspace(x_min, x_max, n_res), np.linspace(y_min, y_max, n_res)
    )

    X_grid = np.concatenate((x_meshgrid.reshape(-1, 1), y_meshgrid.reshape(-1, 1)), axis=1)

    return x_meshgrid, y_meshgrid, X_grid


def plot_classifier(net, X_data, x_meshgrid, y_meshgrid, X_grid, export_path=None):
    y_grid = Softmax()(net(X_grid))[:, 0].reshape(x_meshgrid.shape)
    y_data = net(X_data)
    preds = np.argmax(y_data, axis=1)

    with plt.style.context("seaborn-white"):
        plt.figure(figsize=(5, 5))
        plt.scatter(X_data[preds == 0, 0], X_data[preds == 0, 1], c="b", zorder=1, edgecolor="k")
        plt.scatter(X_data[preds == 1, 0], X_data[preds == 1, 1], c="r", zorder=1, edgecolor="k")
        plt.contourf(x_meshgrid, y_meshgrid, y_grid, zorder=0, cmap="RdBu")
        if not export_path:
            plt.show()
        else:
            plt.savefig(export_path, dpi=500)

        plt.close("all")


# generating some data
n_class_size = 100
r = 2
X1_offset = np.random.rand(n_class_size, 2) - 0.5
np.sqrt(np.sum(X1_offset ** 2, axis=1, keepdims=True))
X1_offset = r * X1_offset / np.sqrt(np.sum(X1_offset ** 2, axis=1, keepdims=True))
X1 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size=n_class_size) + X1_offset
X2 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size=n_class_size)

X = np.concatenate((X1, X2))
Y_labels = np.array([0] * n_class_size + [1] * n_class_size)

plot_data(X1, X2)
# make meshgrid
x_meshgrid, y_meshgrid, X_grid = make_grid(X, n_res=100)

net = Net(
    layers=[Linear(2, 4), ReLU(), Linear(4, 4), ReLU(), Linear(4, 2)], loss=CrossEntropyLoss()
)

n_epochs = 1000
for epoch_idx in range(n_epochs):
    print("Epoch no. %d" % epoch_idx)
    out = net(X)
    # prediction accuracy
    pred = np.argmax(out, axis=1)
    print("accuracy: %1.4f" % (1 - np.abs(pred - Y_labels).sum() / 200))
    loss = net.loss(out, Y_labels)
    print("loss: %1.4f" % loss)
    grad = net.backward()
    net.update_weights(0.1)
    if epoch_idx % 50 == 0:
        plot_classifier(net, X, x_meshgrid, y_meshgrid, X_grid)
