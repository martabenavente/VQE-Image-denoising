import torch
import numpy as np

from torchvision.datasets import MNIST
from torchvision import transforms
from src.mnist_patches import MNISTPatches
from src.noise_generation import NoiseGenerator
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, List, Dict, Any


class NoisyMNISTPatchesDataset(Dataset):
    """
    Dataset that yields noisy MNIST patches for denoising tasks.

    Args:
        mnist_patches: MNISTPatches instance
        noise_generators: Single NoiseGenerator or list of NoiseGenerators for varied noise
        transform: Optional transform to apply to patches
        flatten: Whether to flatten patches (True) or keep 2D (False)
        normalize: Whether to normalize patches to [0, 1] range
        return_clean: Whether to return (clean, noisy) or (noisy, clean) pairs
    """

    def __init__(
        self,
        mnist_patches: MNISTPatches,
        noise_generators: Union[NoiseGenerator, List[NoiseGenerator]],
        transform: Optional[Any] = None,
        flatten: bool = True,
        normalize: bool = True,
        return_clean: bool = True
    ):
        self.mnist_patches = mnist_patches

        # Handle single or multiple noise generators
        if isinstance(noise_generators, NoiseGenerator):
            self.noise_generators = [noise_generators]
        else:
            self.noise_generators = noise_generators

        self.transform = transform
        self.flatten = flatten
        self.normalize = normalize
        self.return_clean = return_clean

        # Calculate total number of patches
        num_images = len(self.mnist_patches.X_train)
        patches_per_image = len(self.mnist_patches.get_patches(0))
        self.total_patches = num_images * patches_per_image
        self.patches_per_image = patches_per_image

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        # Determine which image and which patch
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        # Get all patches for the image
        patches = self.mnist_patches.get_patches(img_idx)
        patch = patches[patch_idx]  # Shape: (4, 4)

        # Convert to torch tensor
        clean_patch = torch.from_numpy(patch).float()

        # Normalize to [0, 1] if requested
        if self.normalize:
            clean_patch = clean_patch / 255.0

        # Flatten if requested
        if self.flatten:
            clean_patch = clean_patch.flatten()
        else:
            # Add channel dimension for 2D: (1, 4, 4)
            clean_patch = clean_patch.unsqueeze(0)

        # Select noise generator (cycle through if multiple)
        noise_gen = self.noise_generators[idx % len(self.noise_generators)]

        # Apply noise
        noisy_patch = noise_gen(clean_patch)

        # Optionally clip to valid range
        if self.normalize:
            noisy_patch = torch.clamp(noisy_patch, 0.0, 1.0)

        # Apply additional transforms if provided
        if self.transform is not None:
            clean_patch = self.transform(clean_patch)
            noisy_patch = self.transform(noisy_patch)

        # Return in requested order
        if self.return_clean:
            return clean_patch, noisy_patch
        else:
            return noisy_patch, clean_patch


def create_mnist_patches_dataloaders(
    stride: int = 4,
    patch_size: int = (4, 4),
    noise_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    batch_size: int = 32,
    train_split: float = 0.8,
    flatten: bool = True,
    normalize: bool = True,
    num_workers: int = 0,
    shuffle_train: bool = True,
    random_seed: Optional[int] = None,
    data_path: str = "_dev-data/datasets/hojjatk/mnist-dataset/versions/1"
) -> Dict[str, DataLoader]:
    """
    Create train/validation DataLoaders for noisy MNIST patches.

    Args:
        stride: Stride for patch extraction (4x4 patches)
        patch_size: Patch size (default (4, 4))
        noise_config: Configuration for noise generation. Can be:
            - Single dict: {'noise_type': 'gaussian', 'mean': 0.0, 'std': 0.1}
            - List of dicts for multiple noise types applied to different patches
        batch_size: Batch size for DataLoader
        train_split: Fraction of data to use for training (rest for validation)
        flatten: Whether to flatten patches to 1D
        normalize: Whether to normalize patches to [0, 1]
        num_workers: Number of workers for DataLoader
        shuffle_train: Whether to shuffle training data
        random_seed: Random seed for reproducibility
        data_path: Path to MNIST dataset files

    Returns:
        Dictionary with 'train' and 'val' DataLoaders
    """
    # Set random seed if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # Default noise configuration if none provided
    if noise_config is None:
        noise_config = {'noise_type': 'gaussian', 'mean': 0.0, 'std': 0.1}

    # Create noise generators
    if isinstance(noise_config, dict):
        noise_generators = [NoiseGenerator(**noise_config)]
    else:
        noise_generators = [NoiseGenerator(**config) for config in noise_config]

    # Load MNIST patches
    mnist_patches = MNISTPatches(patch_size=patch_size, stride=stride, data_path=data_path)

    # Create full dataset
    full_dataset = NoisyMNISTPatchesDataset(
        mnist_patches=mnist_patches,
        noise_generators=noise_generators,
        flatten=flatten,
        normalize=normalize
    )

    # Split into train/validation
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed) if random_seed else None
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return {
        'train': train_loader,
        'val': val_loader
    }

def train_dataset(n_samples = 200, batch_size = 1):
    X_train = MNIST(root='./_dev-data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # # Leaving only labels 0 and 1
    idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
                    np.where(X_train.targets == 1)[0][:n_samples])

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader


def test_dataset(n_samples = 200, batch_size = 1):
    X_test = MNIST(root='./_dev-data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # Leaving only labels 0 and 1
    idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                    np.where(X_test.targets == 1)[0][:n_samples])

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)
    return test_loader