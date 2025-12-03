import torch

import matplotlib.pyplot as plt


def visualize_noise(img: torch.Tensor, noisy_img: torch.Tensor, title: str = "Noise Comparison"):
    """
    Visualize original and noisy images side by side.

    Args:
        img: Original image tensor
        noisy_img: Image with noise added
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convert to numpy for visualization
    img_np = img.permute(1, 2, 0).numpy() if img.ndim == 3 else img.squeeze().numpy()
    noisy_np = noisy_img.permute(1, 2, 0).numpy() if noisy_img.ndim == 3 else noisy_img.squeeze().numpy()
    noise_np = (noisy_img - img).permute(1, 2, 0).numpy() if img.ndim == 3 else (noisy_img - img).squeeze().numpy()

    # Plot original
    axes[0].imshow(img_np, cmap='gray' if img.shape[0] == 1 else None)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot noise
    axes[1].imshow(noise_np, cmap='gray' if img.shape[0] == 1 else None)
    axes[1].set_title('Noise')
    axes[1].axis('off')

    # Plot noisy image
    axes[2].imshow(noisy_np, cmap='gray' if img.shape[0] == 1 else None)
    axes[2].set_title('Noisy Image')
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


class NoiseGenerator:
    """
    A class to generate different types of noise for image tensors.

    Args:
        noise_type (str): Type of noise to generate ('gaussian' or 'uniform').
        mean (float): Mean for Gaussian noise.
        std (float): Standard deviation for Gaussian noise.
        low (float): Lower bound for Uniform noise.
        high (float): Upper bound for Uniform noise.
    """
    def __init__(self, noise_type='gaussian', mean=0.0, std=1.0, low=-1.0, high=1.0):
        self.noise_type = noise_type

        self.mean = mean
        self.std = std

        self.low = low
        self.high = high

    def generate(self, img: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the input image tensor.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Generated noise tensor added to the input image.
        """
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(img) * self.std + self.mean
        elif self.noise_type == 'uniform':
            noise = torch.empty_like(img).uniform_(self.low, self.high)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

        return noise + img

