from nn.layers import *
from nn.losses import MeanSquareLoss
from nn.net import Net

net = Net(layers=[Linear(2, 2), Sigmoid()],
          loss=MeanSquareLoss())
x = np.random.rand(10, 2)
y = np.random.rand(10, 2)

out = net(x)
loss = net.loss(out, y)
grad = net.backward()

pass
