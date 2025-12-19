import matplotlib.pyplot as plt

from src.data.dataloaders import create_mnist_patches_dataloaders


# Single noise type
dataloaders = create_mnist_patches_dataloaders(
    stride=4,
    noise_config={'noise_type': 'gaussian', 'mean': 0.0, 'std': 0.1},
    batch_size=64,
    flatten=True,
    normalize=True,
    data_path="/home/yeray142/Documents/projects/VQE-Image-denoising/_dev-data/datasets/hojjatk/mnist-dataset/versions/1"
)

train_loader = dataloaders['train']
val_loader = dataloaders['val']

# Get a batch from the train loader
batch = next(iter(train_loader))
clean_patches, noisy_patches = batch

print(f"Batch shape - Clean: {clean_patches.shape}, Noisy: {noisy_patches.shape}")
print(f"Clean range: [{clean_patches.min():.3f}, {clean_patches.max():.3f}]")
print(f"Noisy range: [{noisy_patches.min():.3f}, {noisy_patches.max():.3f}]")

# Visualize multiple patches to verify different noise
num_samples = 8
fig, axes = plt.subplots(3, num_samples, figsize=(16, 6))

for i in range(num_samples):
    # Reshape from flattened (16,) to 2D (4, 4) for visualization
    clean = clean_patches[i].reshape(4, 4).numpy()
    noisy = noisy_patches[i].reshape(4, 4).numpy()
    noise = noisy - clean

    # Plot clean patch
    axes[0, i].imshow(clean, cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Clean', fontsize=10)

    # Plot noise only
    axes[1, i].imshow(noise, cmap='RdBu', vmin=-0.3, vmax=0.3)
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Noise', fontsize=10)

    # Plot noisy patch
    axes[2, i].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('Noisy', fontsize=10)

plt.suptitle('Different Noise Applied to MNIST Patches', fontsize=14, y=0.98)
plt.tight_layout()
plt.show()

# Compute noise statistics to verify different magnitudes
print("\nNoise statistics for first 8 samples:")
for i in range(num_samples):
    clean = clean_patches[i].reshape(4, 4)
    noisy = noisy_patches[i].reshape(4, 4)
    noise = (noisy - clean).numpy()
    print(f"Sample {i}: mean={noise.mean():.4f}, std={noise.std():.4f}, "
          f"min={noise.min():.4f}, max={noise.max():.4f}")

print("-" * 50)
print(f"Total patches in train loader: {len(train_loader.dataset)}")
print(f"Total patches in val loader: {len(val_loader.dataset)}")

