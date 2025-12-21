import numpy as np
import os
import struct


def load_idx_images(filename):
    with open(filename, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def load_idx_labels(filename):
    with open(filename, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


def extract_patches(img, patch_size=(4, 4), stride=None):
    """
    Extract patches from an image.

    Args:
        img: numpy array (H, W)
        patch_size: tuple (patch_height, patch_width)
        stride: tuple (stride_h, stride_w) or int. If None, defaults to patch_size

    Returns:
        numpy array (num_patches, patch_height, patch_width)
    """
    H, W = img.shape
    patch_h, patch_w = patch_size

    # Default stride to patch size if not specified
    if stride is None:
        stride_h, stride_w = patch_h, patch_w
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    patches = []
    for i in range(0, H - patch_h + 1, stride_h):
        for j in range(0, W - patch_w + 1, stride_w):
            patch = img[i:i+patch_h, j:j+patch_w]
            patches.append(patch)

    return np.array(patches)


class MNISTPatches:
    def __init__(self, patch_size=(4, 4), stride=None, data_path="_dev-data/datasets/hojjatk/mnist-dataset/versions/1"):
        """
        Downloads dataset and prepare train images.

        Args:
            patch_size: tuple (height, width) for patch dimensions
            stride: int or tuple for stride. If None, uses patch_size
            data_path: path to MNIST dataset
        """
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.stride = stride

        train_images_path = os.path.join(data_path, "train-images.idx3-ubyte")
        train_labels_path = os.path.join(data_path, "train-labels.idx1-ubyte")

        self.X_train = load_idx_images(train_images_path)
        self.y_train = load_idx_labels(train_labels_path)

        # Filter only images with label 0 and take first 100
        zero_mask = self.y_train == 0
        self.X_train = self.X_train[zero_mask][:100]
        self.y_train = self.y_train[zero_mask][:100]

    def get_image(self, idx):
        """Returns an image 28x28 from the train set."""
        return self.X_train[idx]

    def get_patches(self, idx):
        """Returns patches of configured size from the idxth image."""
        img = self.get_image(idx)
        return extract_patches(img, patch_size=self.patch_size, stride=self.stride)