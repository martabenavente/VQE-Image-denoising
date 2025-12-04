import kagglehub
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


def extract_patches_4x4(img, stride=4):
    """
    Args:
        img: numpy array (28, 28)
        stride: jump between parches

    Returns:
        numpy array (num_patches, 4, 4)
    """
    H, W = img.shape
    patches = []

    for i in range(0, H - 4 + 1, stride):
        for j in range(0, W - 4 + 1, stride):
            patch = img[i:i+4, j:j+4]
            patches.append(patch)

    return np.array(patches)


class MNISTPatches:
    def __init__(self, stride=4):
        """Downloads dataset and prepare train images."""
        self.stride = stride

        # Descargar dataset UNA sola vez
        path = kagglehub.dataset_download("hojjatk/mnist-dataset")
        print("Dataset path:", path)

        train_images_path = os.path.join(path, "train-images.idx3-ubyte")
        train_labels_path = os.path.join(path, "train-labels.idx1-ubyte")

        self.X_train = load_idx_images(train_images_path)
        self.y_train = load_idx_labels(train_labels_path)

    def get_image(self, idx):
        """Returns an image 28x28 from the train set."""
        return self.X_train[idx]

    def get_patches(self, idx):
        """Returns a 4x4 patch from the idxth image."""
        img = self.get_image(idx)
        return extract_patches_4x4(img, stride=self.stride)
