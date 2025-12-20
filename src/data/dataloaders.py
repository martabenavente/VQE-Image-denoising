import torch
import numpy as np

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset

from src.noise_generation import add_gaussian_noise


class NoisyMNISTDataset(Dataset):
    """
    Wraps MNIST dataset and applies noise on-the-fly.

    Args:
        mnist_dataset: Instance of torchvision MNIST dataset
        add_noise_fn: Function that takes a clean image tensor and returns a noisy image tensor
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        clean_img, label = self.mnist_dataset[idx]
        noisy_img = add_gaussian_noise(clean_img.unsqueeze(0)).squeeze(0)
        return clean_img, noisy_img  # (ground_truth, input)


def train_dataset(n_samples = 200, batch_size = 1, target_classes=None):
    """
    Create dataloader for training dataset with noisy MNIST images.

    Args:
        n_samples (int): Number of samples per class (0 and 1) to include in the dataset.
        batch_size (int): Batch size for the dataloader.
        target_classes (list): List of target classes to include in the dataset. Default is [0, 1].

    Returns:
        train_loader: DataLoader for the training dataset.
    """
    if target_classes is None:
        target_classes = [0, 1]
    X_train = MNIST(root='./_dev-data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    idx = np.concatenate([np.where(X_train.targets == target)[0][:n_samples] for target in target_classes])

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]
    X_train = NoisyMNISTDataset(X_train)

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader


def test_dataset(n_samples = 200, batch_size = 1, target_classes=None):
    """
    Create dataloader for test dataset with clean MNIST images.

    Args:
        n_samples (int): Number of samples per class (0 and 1) to include in the dataset.
        batch_size (int): Batch size for the dataloader.
        target_classes (list): List of target classes to include in the dataset. Default is [0, 1].

    Returns:
        test_loader: DataLoader for the test dataset.
    """
    if target_classes is None:
        target_classes = [0, 1]
    X_test = MNIST(root='./_dev-data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    idx = np.concatenate([np.where(X_test.targets == target)[0][:n_samples] for target in target_classes])

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]
    X_test = NoisyMNISTDataset(X_test)

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)
    return test_loader