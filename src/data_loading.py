import numpy as np

from pathlib import Path


def read_idx_ubyte(filepath):
    """
    Read IDX ubyte file format.

    Args:
        filepath (str or Path): Path to the IDX ubyte file.

    Returns:
        np.ndarray: Numpy array containing the data.

    """
    with open(filepath, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), 'big')
        num_items = int.from_bytes(f.read(4), 'big')

        if magic == 2051:  # Images
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_items, rows, cols)
        elif magic == 2049:  # Labels
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown magic number: {magic}")

    return data


def load_mnist_from_kaggle(dataset_path):
    """
    Load MNIST dataset from Kaggle downloaded files.

    Args:
        dataset_path (str or Path): Path to the directory containing the MNIST IDX files.

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels) as numpy arrays.
    """
    # Construct paths to the IDX files
    train_images_path = Path(dataset_path) / "train-images.idx3-ubyte"
    train_labels_path = Path(dataset_path) / "train-labels.idx1-ubyte"
    test_images_path = Path(dataset_path) / "t10k-images.idx3-ubyte"
    test_labels_path = Path(dataset_path)  / "t10k-labels.idx1-ubyte"

    # Load training data
    train_images = read_idx_ubyte(train_images_path)
    train_labels = read_idx_ubyte(train_labels_path)

    # Load test data
    test_images = read_idx_ubyte(test_images_path)
    test_labels = read_idx_ubyte(test_labels_path)

    return train_images, train_labels, test_images, test_labels