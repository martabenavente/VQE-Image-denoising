import matplotlib.pyplot as plt

from src.data_loading import load_mnist_from_kaggle


def visualize_samples(images, labels, num_samples=10):
    """Visualize sample images from the dataset."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_samples):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Path to downloaded Kaggle dataset
    dataset_path = "/home/yeray142/Documents/projects/VQE-Image-denoising/_dev-data/datasets/hojjatk/mnist-dataset/versions/1"

    # Load MNIST data
    train_images, train_labels, test_images, test_labels = load_mnist_from_kaggle(dataset_path)

    # Print dataset information
    print(f"Training images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # Visualize some samples
    visualize_samples(train_images, train_labels)