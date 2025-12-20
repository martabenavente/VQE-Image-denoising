import pickle
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(history_path: str):
    """Plot training history from pickle file."""
    history_path = Path(history_path)

    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    print("=" * 80)
    print("TRAINING HISTORY VISUALIZATION")
    print("=" * 80)
    print(f"\nLoading history from: {history_path}")

    # Load history
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    print(f"\nAvailable metrics: {list(history.keys())}")

    # Separate train and validation metrics
    train_metrics = {k: v for k, v in history.items() if k.startswith('train_')}
    val_metrics = {k: v for k, v in history.items() if k.startswith('val_')}

    # Determine number of subplots needed
    metric_names = set()
    for key in train_metrics.keys():
        metric_name = key.replace('train_', '')
        metric_names.add(metric_name)

    num_metrics = len(metric_names)

    # Create figure with subplots
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
    if num_metrics == 1:
        axes = [axes]

    # Plot each metric
    for idx, metric_name in enumerate(sorted(metric_names)):
        ax = axes[idx]

        train_key = f'train_{metric_name}'
        val_key = f'val_{metric_name}'

        # Plot training metric
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], 'b-', label=f'Train {metric_name}', linewidth=2)

        # Plot validation metric
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], 'r-', label=f'Val {metric_name}', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name.upper(), fontsize=12)
        ax.set_title(f'{metric_name.upper()} over Training', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Print final values
        if train_key in history:
            print(f"\n{metric_name.upper()}:")
            print(f"  Final train: {history[train_key][-1]:.6f}")
            if val_key in history:
                print(f"  Final val: {history[val_key][-1]:.6f}")
                print(f"  Best val: {min(history[val_key]) if metric_name == 'loss' else max(history[val_key]):.6f}")

    plt.tight_layout()

    # Save plot
    output_path = history_path.parent / 'training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Plot training history from pickle file'
    )
    parser.add_argument(
        '--history',
        type=str,
        required=True,
        help='Path to training_history.pkl file (e.g., checkpoints/20240101_120000/training_history.pkl)'
    )

    args = parser.parse_args()
    plot_training_history(args.history)


if __name__ == '__main__':
    main()
