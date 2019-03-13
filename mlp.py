from nn.layers import *
from nn.losses import MeanSquareLoss, CrossEntropyLoss
from nn.net import Net

net = Net(layers=[Linear(2, 2), Sigmoid()],
          loss=MeanSquareLoss())
x = np.random.rand(10, 2)
y = np.random.rand(10, 2)
yl = np.random.randint(0, 2, size=(10, 1))

crossent = CrossEntropyLoss()
crossent(x, yl)

out = net(x)
loss = net.loss(out, y)
grad = net.backward()
net.update_weights(10)
