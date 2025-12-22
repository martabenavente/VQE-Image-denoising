import torch
import json
import pickle

from pathlib import Path
from datetime import datetime

from src.data.dataloaders import train_dataset, test_dataset
from src.circuits.ansatz_circuit import SimpleAnsatzCircuit
from src.training.trainer import QiskitTrainer
from src.utils.loss import DenoisingLoss
from src.utils.metrics import DenoisingMetrics
from src.qnn.autoencoder_model import ConvDenoiseNet
from src.utils.loggers import WandBMetricLogger


def train_model():
    """Train quantum denoising model with optimized configuration."""

    # Training configuration
    config = {
        'batch_size': 16,  # Smaller batches for quantum circuits
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'early_stopping_patience': 10,
        'save_frequency': 5,
        'num_qubits': 4,
        'num_layers': 1,
        "use_wandb": True, 
        "wandb_project": "vqe-image-denoising",
        "wandb_run_name": None,
        "wandb_log_images_every_n": 1 # Add an int in case we want to activate
    }

    print("=" * 80)
    print("QUANTUM IMAGE DENOISING - TRAINING")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints') / datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create dataloaders
    print("\n" + "-" * 80)
    print("Loading MNIST Dataset...")
    print("-" * 80)

    train_loader = train_dataset(n_samples=100, batch_size=config['batch_size'])
    val_loader = test_dataset(n_samples=5, batch_size=config['batch_size'])

    print(f"\nDataset Statistics:")
    print(f"  Total training images: {len(train_loader.dataset):,}")
    print(f"  Total validation images: {len(val_loader.dataset):,}")
    print(f"  Training batches: {len(train_loader):,}")
    print(f"  Validation batches: {len(val_loader):,}")
    print(f"  Patch size: 3x3 (9 features)")

    # Build quantum circuit
    print("\n" + "-" * 80)
    print("Building Quantum Neural Network...")
    print("-" * 80)

    circuit = SimpleAnsatzCircuit(
        num_qubits=config['num_qubits'],
        num_features=config['num_qubits'],
        num_parameters=config['num_qubits'] * 2
    )
    model_wrapper = ConvDenoiseNet(circuit=circuit, quantum=True)

    print(f"\nQuantum Circuit Details:")
    print(f"  Number of qubits: {config['num_qubits']}")
    print(f"  Number of layers: {config['num_layers']}")
    print(f"  Feature parameters: {len(circuit.feature_params)}")
    print(f"  Weight parameters: {len(circuit.weight_params)}")
    print(f"  Total parameters: {len(circuit.feature_params) + len(circuit.weight_params)}")

    # Total params models and trainable params
    total_params = sum(p.numel() for p in model_wrapper.parameters())
    trainable_params = sum(p.numel() for p in model_wrapper.parameters() if p.requires_grad)
    print(f"  Total model parameters: {total_params}")
    print(f"  Trainable model parameters: {trainable_params}")

    # Setup loss and metrics
    loss_fn = DenoisingLoss(loss_type='mse')
    metrics = DenoisingMetrics(data_range=1.0)

    # Setup optimizer with weight decay
    optimizer = torch.optim.Adam(
        model_wrapper.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Setup logger
    wandb_logger = None
    if config.get("use_wandb", False):
        wandb_logger = WandBMetricLogger(
            project=config.get("wandb_project", "vqe-image-denoising"),
            name=config.get("wandb_run_name", None),
            config=config
        )

    # Create trainer
    print("\n" + "-" * 80)
    print("Initializing Trainer...")
    print("-" * 80)

    trainer = QiskitTrainer(
        model=model_wrapper,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        device='auto',
        gradient_clip=config['gradient_clip'],
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=config['early_stopping_patience'],
        early_stopping_metric='loss',
        early_stopping_mode='min',
        logger=wandb_logger,
        log_images_every_n=int(config.get("wandb_log_images_every_n", 0) or 0)
    )
    trainer.set_scheduler(scheduler)

    print(f"  Device: {trainer.device}")
    print(f"  Gradient clipping: {config['gradient_clip']}")
    print(f"  Early stopping patience: {config['early_stopping_patience']} epochs")
    print(f"  Checkpoints saved to: {checkpoint_dir}")

    # Training callback
    def on_epoch_end(epoch, train_metrics, val_metrics):
        """Print detailed metrics after each epoch."""
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Summary:")
        print(f"{'='*80}")

        print(f"\nTraining Metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.6f}")

        if val_metrics:
            print(f"\nValidation Metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.6f}")

        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nCurrent Learning Rate: {current_lr:.6f}")

    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    start_time = datetime.now()

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        verbose=False,
        save_frequency=config['save_frequency'],
        validate_frequency=1,
        on_epoch_end=on_epoch_end
    )

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds() / 60

    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTotal Training Time: {training_duration:.2f} minutes ({training_duration/60:.2f} hours)")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Final metrics
    print(f"\nFinal Training Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
    print(f"Best Validation Loss: {min(history['val_loss']):.6f}")

    # Save history
    with open(checkpoint_dir / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    print(f"\nTraining history saved to: {checkpoint_dir / 'training_history.pkl'}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")

    return trainer, history


if __name__ == '__main__':
    trainer, history = train_model()