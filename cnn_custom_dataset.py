import numpy as np

from argparse import ArgumentParser

from nn.layers import *
from nn.losses import CrossEntropyLoss
from nn.activations import ReLU
from nn.net import Net
from nn.utils import load_data

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
args = parser.parse_args()

# load images
print('loading data ...')
X, y = load_data(args.dataset)
print('data loaded')
# scaling and converting to float
print('scaling data...')
X = X.astype('float32')/255
print('data scaled')

# split to train and validation datasets
idx = np.arange(len(X))
np.random.shuffle(idx)
X_train, y_train = X[idx[:50000]], y[idx[:50000]]
X_val, y_val = X[idx[50000:]], y[idx[50000:]]

net = Net(layers=[Conv2D(3, 8, 3, padding=1), MaxPool2D(kernel_size=2), ReLU(), BatchNorm2D(8),
                  Conv2D(8, 16, 3, padding=1), MaxPool2D(kernel_size=2), ReLU(), BatchNorm2D(16),
                  Flatten(), Linear(16*13*13, 12)],
          loss=CrossEntropyLoss())

n_epochs = args.epochs
n_batch = args.batch_size
for epoch_idx in range(n_epochs):
    batch_idx = np.random.choice(range(len(X_train)), size=n_batch, replace=False)
    out = net(X_train[batch_idx])
    preds = np.argmax(out, axis=1).reshape(-1, 1)
    accuracy = 100*(preds == y_train[batch_idx]).sum() / n_batch
    loss = net.loss(out, y_train[batch_idx])
    net.backward()
    net.update_weights(lr=args.lr)
    print("Epoch no. %d loss =  %2f4 \t accuracy = %d %%" % (epoch_idx + 1, loss, accuracy))
    if epoch_idx % 10 == 0:
        val_idx = np.random.choice(range(len(X_val)), size=n_batch, replace=False)
        val_out = net.forward(X_val[val_idx])
        val_pred = np.argmax(val_out, axis=1).reshape(-1, 1)
        val_loss = net.loss(val_out, y_val[val_idx])
        val_acc = 100*(val_pred == y_val[val_idx]).sum() / n_batch
        print("Validation loss = %2f4 \t accuracy = %d %%" % (val_loss, val_acc))
