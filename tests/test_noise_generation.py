import pytest
import torch

from src.noise_generation import NoiseGenerator
from examples.load_data_example import load_mnist_from_kaggle


class TestNoiseGenerator:
    """Test suite for NoiseGenerator class."""

    def test_gaussian_noise_generation(self):
        """Test Gaussian noise generation."""
        generator = NoiseGenerator(noise_type='gaussian', mean=0.0, std=0.1)
        img = torch.zeros(3, 64, 64)
        noisy_img = generator.generate(img)

        assert noisy_img.shape == img.shape
        assert not torch.equal(noisy_img, img)

    def test_uniform_noise_generation(self):
        """Test Uniform noise generation."""
        generator = NoiseGenerator(noise_type='uniform', low=-1.0, high=1.0)
        img = torch.zeros(3, 64, 64)
        noisy_img = generator.generate(img)

        assert noisy_img.shape == img.shape
        assert not torch.equal(noisy_img, img)

    def test_gaussian_noise_properties(self):
        """Test that Gaussian noise has approximately correct mean and std."""
        generator = NoiseGenerator(noise_type='gaussian', mean=5.0, std=2.0)
        img = torch.zeros(1, 1000, 1000)
        noisy_img = generator.generate(img)
        noise = noisy_img - img

        assert torch.abs(noise.mean() - 5.0) < 0.1
        assert torch.abs(noise.std() - 2.0) < 0.1

    def test_uniform_noise_bounds(self):
        """Test that Uniform noise is within specified bounds."""
        low, high = -2.0, 3.0
        generator = NoiseGenerator(noise_type='uniform', low=low, high=high)
        img = torch.zeros(3, 100, 100)
        noisy_img = generator.generate(img)
        noise = noisy_img - img

        assert noise.min() >= low
        assert noise.max() <= high

    def test_invalid_noise_type(self):
        """Test that invalid noise type raises ValueError."""
        generator = NoiseGenerator(noise_type='invalid')
        img = torch.zeros(3, 64, 64)

        with pytest.raises(ValueError, match="Unsupported noise type"):
            generator.generate(img)

    def test_default_parameters(self):
        """Test that default parameters work correctly."""
        generator = NoiseGenerator()
        img = torch.ones(3, 32, 32)
        noisy_img = generator.generate(img)

        assert noisy_img.shape == img.shape
        assert generator.noise_type == 'gaussian'

    def test_different_image_shapes(self):
        """Test noise generation with different image shapes."""
        generator = NoiseGenerator()
        shapes = [(1, 28, 28), (3, 224, 224), (4, 512, 512)]

        for shape in shapes:
            img = torch.zeros(*shape)
            noisy_img = generator.generate(img)
            assert noisy_img.shape == shape

    def test_noise_addition(self):
        """Test that noise is actually added to the image."""
        generator = NoiseGenerator(noise_type='gaussian', mean=1.0, std=0.1)
        img = torch.zeros(3, 64, 64)
        noisy_img = generator.generate(img)

        assert (noisy_img != img).any()
        assert torch.abs(noisy_img.mean() - 1.0) < 0.2

    def test_with_real_image(self):
        """Test noise generation with a real image tensor."""
        generator = NoiseGenerator(noise_type='uniform', low=0, high=0.3)

        train_images, train_labels, test_images, test_labels = load_mnist_from_kaggle(
            "/home/yeray142/Documents/projects/VQE-Image-denoising/_dev-data/datasets/hojjatk/mnist-dataset/versions/1"
        )

        img = torch.tensor(train_images[0]).unsqueeze(0).float() / 255.0
        noisy_img = generator.generate(img)

        assert noisy_img.shape == img.shape
        assert not torch.equal(noisy_img, img)

    def test_noise_generator_call(self):
        """Test the __call__ method of NoiseGenerator."""
        generator = NoiseGenerator(noise_type='gaussian', mean=0.0, std=0.5)
        img = torch.zeros(3, 64, 64)
        noisy_img = generator(img)

        assert noisy_img.shape == img.shape
        assert not torch.equal(noisy_img, img)
