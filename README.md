# neural-networks-from-scratch

# Contents
- [Quickstart](#quickstart)
  - [A simple example CNN](#CNN-example)
  - [The `Net` object](#net)
- [Layers](#layers)
  - [`Linear`](#linear)
  - [`Conv2D`](#conv2d)
  - [`MaxPool2D`](#maxpool2d)
  - [`BatchNorm2D`](#batchnorm2d)
  - [`Flatten`](#flatten)
- [Losses](#losses)
  - [`CrossEntropyLoss`](#crossentropyloss)
  - [`MeanSquareLoss`](#meansquareloss)
  
# Quickstart<a name="quickstart"></a>
## A simple example CNN<a name="CNN-example"></a>
A simple convolutional network for image classification can be found in `CNN_custom_dataset.py`. To try it on your own dataset, you should prepare your images in the following format:
```bash
images_folder
   |-- class_01
       |-- 001.png
       |-- ...
   |-- class_02
       |-- 001.png
       |-- ...
   |-- ...
```
Its required argument is
- `--dataset`: path to the dataset,
while the optional arguments are
- `--epochs`: number of epochs,
- `--batch_size`: size of the training batch,
- `--lr`: learning rate.

## The `Net` object<a name="net"></a>
To define a neural network, the `nn.net.Net` object can be used. Its parameters are
* `layers`: a list of layers from `nn.layers`, for example `[Linear(2, 4), ReLU(), Linear(4, 2)]`,
* `loss`: a loss function from `nn.losses`, for example `CrossEntropyLoss` or `MeanSquareLoss`.
If you would like to train the model with data `X` and label `y`, you should
1) perform the forward pass, during which local gradients are calculated,
2) calculate the loss,
3) perform the backward pass, where global gradients with respect to the variables and layer parameters are calculated,
4) update the weights.

In code, this looks like the following:
```python3
out = net(X)
loss = net.loss(out, y)
net.backward()
net.update_weights(lr)
```

# Layers<a name="layers"></a>
## `Linear`<a name="linear"></a>
## `Conv2D`<a name="conv2d"></a>
## `MaxPool2D`<a name="maxpool2d"></a>
## `BatchNorm2D`<a name="batchnorm2d"></a>
## `Flatten`<a name="flatten"></a>

# Losses<a name="losses"></a>
## CrossEntropyLoss<a name="crossentropyloss"></a>
## MeanSquareLoss<a name="meansquareloss"></a>
