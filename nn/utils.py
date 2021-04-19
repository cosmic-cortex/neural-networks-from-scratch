import os
import numpy as np

from skimage import io


def zero_pad(X, pad_width, dims):
    """
    Pads the given array X with zeroes at the both end of given dims.

    Args:
        X: numpy.ndarray.
        pad_width: int, width of the padding.
        dims: int or tuple, dimensions to be padded.

    Returns:
        X_padded: numpy.ndarray, zero padded X.
    """
    dims = (dims) if isinstance(dims, int) else dims
    pad = [(0, 0) if idx not in dims else (pad_width, pad_width) for idx in range(len(X.shape))]
    X_padded = np.pad(X, pad, "constant")
    return X_padded


def load_data(folder_path):
    imgs = []
    labels = []
    for class_dir in os.listdir(folder_path):
        class_label = int(class_dir) - 1
        class_path = os.path.join(folder_path, class_dir)
        imgs.append(
            np.array(
                [
                    io.imread(os.path.join(class_path, fname)).transpose((2, 0, 1))
                    for fname in os.listdir(class_path)
                ]
            )
        )
        labels.append(np.array([class_label] * len(os.listdir(class_path))))

    X = np.concatenate(imgs, axis=0)
    y = np.concatenate(labels).reshape(-1, 1)

    return X, y
