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
- [Activations](#activations)
  
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
The currently implemented layers can be found in `nn.layers`. Each layer is a callable object, where calling performs the forward pass and calculates local gradients. The most important methods are:
- `.forward(X)`: performs the forward pass for X. Instead calling `forward` directly, the layer object should be called directly, which calculates and caches local gradients.
- `.backward(dY)`: performs the backward pass, where `dY` is the gradient propagated backwards from the consequtive layer.
- `.local_grad(X)`: calculates the local gradient of the input.

The input to the layers should always be a `numpy.ndarray` of shape `(n_batch, ...)`. For the 2D layers for images, the input should have shape `(n_batch, n_channels, n_height, n_width)`.

## `Linear`<a name="linear"></a>
A simple fully connected layer. 
Parameters:
- `in_dim`: integer, dimensions of the input.
- `out_dim`: integer, dimensions of the output.

Usage:
- input: `numpy.ndarray` of shape `(N, in_dim)`.
- output: `numpy.ndarray` of shape `(N, out_dim)`.

## `Conv2D`<a name="conv2d"></a>
2D convolutional layer. Parameters:
- `in_channels`: integer, number of channels in the input image.
- `out_channels`: integer, number of filters to be learned.
- `kernel_size`: integer or tuple, the size of the filter to be learned. Defaults to 3.
- `stride`: integer, stride of the convolution. Defaults to 1.
- `padding`: integer, number of zeros to be added to each edge of the images. Defaults to 0.

Usage:
- input: `numpy.ndarray` of shape `(N, C_in, H_in, W_in)`.
- output: `numpy.ndarray` of shape `(N, C_out, H_out, W_out)`.

## `MaxPool2D`<a name="maxpool2d"></a>
2D max pooling layer. Parameters:
- `kernel_size`: integer or tuple, size of the pooling window. Defaults to 2.

Usage:
- input: `numpy.ndarray` of shape `(N, C, H, W)`.
- output: `numpy.ndarray` of shape `(N, C, H//KH, W//KW)` with kernel size `(KH, KW)`.

## `BatchNorm2D`<a name="batchnorm2d"></a>
2D batch normalization layer. Parameters:
- `n_channels`: integer, number of channels.
- `epsilon`: epsilon parameter for BatchNorm, defaults to 1e-5.

Usage:
- input: `numpy.ndarray` of shape `(N, C, H, W)`.
- output: `numpy.ndarray` of shape `(N, C, H, W)`.

## `Flatten`<a name="flatten"></a>
A simple layer which flattens the outputs of a 2D layer for images.

Usage:
- input: `numpy.ndarray` of shape `(N, C, H, W)`.
- output: `numpy.ndarray` of shape `(N, C*H*W)`.

# Losses<a name="losses"></a>
The implemented loss functions are located in `nn.losses`. As Layers, they are callable objects, with predictions and targets as input.

## `CrossEntropyLoss`<a name="crossentropyloss"></a>
Cross-entropy loss. Usage:
- input: `numpy.ndarray` of shape `(N, D)` containing the class scores for each element in the batch.
- output: `float`.

## `MeanSquareLoss`<a name="meansquareloss"></a>
Mean square loss. Usage:
- input: `numpy.ndarray` of shape `(N, D)`.
- output: `numpy.ndarray` of shape `(N, D)`.

# Activations<a name="activations"></a>
The activation layers for the network can be found in `nn.activations`. They are functions, applying the specified activation function elementwisely on a `numpy.ndarray`. Currently, the following activation functions are implemented:
- ReLU
- Leaky ReLU
- Sigmoid
