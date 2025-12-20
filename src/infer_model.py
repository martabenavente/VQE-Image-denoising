import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.dataloaders import test_dataset
from src.circuits.ansatz_circuit import SimpleAnsatzCircuit
from src.qnn.autoencoder_model import ConvDenoiseNet
from src.utils.metrics import DenoisingMetrics


def load_checkpoint(checkpoint_path: Path, num_qubits: int = 9):
    """Load model from checkpoint."""
    circuit = SimpleAnsatzCircuit(
        num_qubits=num_qubits,
        num_features=num_qubits,
        num_parameters=num_qubits * 2
    )
    model = ConvDenoiseNet(circuit=circuit, quantum=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def infer_single_image(checkpoint_path: str):
    """Test model with a single random MNIST image."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 80)
    print("TESTING QUANTUM DENOISING MODEL")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint_path}")

    # Load model
    print("\nLoading model...")
    model, checkpoint = load_checkpoint(checkpoint_path, num_qubits=4)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    if 'metrics' in checkpoint and checkpoint['metrics']:
        print("\nCheckpoint metrics:")
        for key, value in checkpoint['metrics'].items():
            print(f"  {key}: {value:.6f}")

    # Load single test image
    print("\nLoading test image...")
    test_loader = test_dataset(n_samples=1, batch_size=1, target_classes=[1])
    clean_image, noisy_image = next(iter(test_loader))

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        denoised_image = model(noisy_image)
        denoised_image = denoised_image.view(1, 1, 28, 28)

    # Calculate metrics
    metrics = DenoisingMetrics(data_range=1.0)
    metrics.update(denoised_image, clean_image)
    results = metrics.compute()

    print("\nTest Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.6f}")

    # Visualize results
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    clean_img = clean_image[0, 0].cpu().numpy()
    noisy_img = noisy_image[0, 0].cpu().numpy()
    denoised_img = denoised_image[0, 0].cpu().numpy()

    axes[0].imshow(clean_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Clean Image')
    axes[0].axis('off')

    axes[1].imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Noisy Image')
    axes[1].axis('off')

    axes[2].imshow(denoised_img, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Denoised Image')
    axes[2].axis('off')

    plt.tight_layout()

    # Save visualization
    output_path = checkpoint_path.parent / 'test_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Test quantum denoising model with a single MNIST image'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (e.g., checkpoints/20240101_120000/best_model.pt)'
    )

    args = parser.parse_args()
    infer_single_image(args.checkpoint)


if __name__ == '__main__':
    main()
