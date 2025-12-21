import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any, List
from pathlib import Path
from tqdm import tqdm

from src.noise_generation import add_gaussian_noise


class QiskitTrainer:
    """
    Flexible trainer for Qiskit Machine Learning models using TorchConnector.

    Supports:
    - Custom loss functions and metrics
    - Train/validation/test loops
    - Checkpointing
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping

    Args:
        qnn: EstimatorQNN instance from QNNBuilder
        loss_fn: Loss function (e.g., DenoisingLoss)
        metrics: Metrics calculator (e.g., DenoisingMetrics) or None
        optimizer: PyTorch optimizer. If None, uses Adam with lr=0.001
        device: Device to train on ('auto', 'cpu', 'cuda', etc.)
        gradient_clip: Maximum gradient norm for clipping (None to disable)
        checkpoint_dir: Directory to save checkpoints (None to disable)
        early_stopping_patience: Epochs to wait before early stopping (None to disable)
        early_stopping_metric: Metric to monitor for early stopping ('loss' or metric name)
        early_stopping_mode: 'min' or 'max' for early stopping metric
    """

    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module,
            metrics: Optional[Any] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            device: str = 'auto',
            gradient_clip: Optional[float] = None,
            checkpoint_dir: Optional[Path] = None,
            early_stopping_patience: Optional[int] = None,
            early_stopping_metric: str = 'loss',
            early_stopping_mode: str = 'min',
            logger=None,
            log_images_every_n: int = 0,
            max_log_images: int = 8,
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("Using device:", self.device)
        else:
            self.device = torch.device(device)

        # Wrap QNN in TorchConnector
        self.model = model.to(self.device)

        self.loss_fn = loss_fn.to(self.device)
        self.metrics = metrics

        # Initialize optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer

        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode
        self._early_stopping_counter = 0
        self._best_metric = float('inf') if early_stopping_mode == 'min' else float('-inf')

        # Logger
        self.logger = logger
        self.log_images_every_n = log_images_every_n
        self.max_log_images = max_log_images

        # Learning rate scheduler (can be set via set_scheduler)
        self.scheduler: Optional[Any] = None

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
        }

    def set_scheduler(self, scheduler: Any):
        """
        Set learning rate scheduler.

        Args:
            scheduler: Learning rate scheduler instance
        """
        self.scheduler = scheduler

    def _iter_steps(self, loader: DataLoader, steps: Optional[int]):
        """Iterate over a DataLoader but stop after 'steps' batches."""

        if steps is None:
            yield from loader
        else:
            for i, batch in enumerate(loader):
                if i >= steps:
                    break
                yield batch

    def _train_epoch(
            self,
            train_loader: DataLoader,
            epoch: int,
            verbose: bool = True
    ) -> Dict[str, float]:
        """
        Execute one training epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            verbose: Whether to show progress bar

        Returns:
            Dictionary with average training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_samples = 0

        if self.metrics:
            self.metrics.reset()

        iterator = tqdm(train_loader, desc=f'Epoch {epoch} [Train]') if verbose else train_loader
        for batch_idx, batch in enumerate(iterator):
            images, noisy_images = batch

            # Move to device
            images = images.to(self.device)
            noisy_images = noisy_images.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(noisy_images)
            loss = self.loss_fn(outputs, images)
            loss.backward()

            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            self.optimizer.step()
            
            bs = noisy_images.size(0)
            num_samples += bs
            epoch_loss += loss.item() * bs

            outputs = outputs.detach()
            if outputs.shape != images.shape:
                outputs = outputs.reshape(images.shape)

            if self.metrics:
                self.metrics.update(outputs.cpu(), images.cpu())

            if verbose:
                iterator.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / max(1, num_samples)
        results = {'loss': avg_loss}

        if self.metrics:
            metric_values = self.metrics.compute()
            results.update(metric_values)

        return results

    def _validate_epoch(
            self,
            val_loader: DataLoader,
            epoch: int,
            verbose: bool = True,
            prefix: str = 'Val',
            steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Execute one validation/test epoch.

        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            verbose: Whether to show progress bar
            prefix: Prefix for progress bar description ('Val' or 'Test')

        Returns:
            Dictionary with average validation metrics
        """
        self.model.eval()
        epoch_loss = 0.0

        if self.metrics:
            self.metrics.reset()

        base_iter = self._iter_steps(val_loader, steps)
        iterator = tqdm(base_iter, desc=f"Epoch {epoch} [{prefix}]") if verbose else base_iter

        epoch_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                images, noisy_images = batch
                inputs = noisy_images.to(self.device)
                targets = images.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                bs = inputs.size(0)
                num_samples += bs
                epoch_loss += loss.item() * bs


                if outputs.shape != targets.shape:
                    outputs = outputs.reshape(targets.shape)

                if self.metrics:
                    self.metrics.update(outputs.cpu(), targets.cpu())

                if verbose:
                    iterator.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / max(1, num_samples)
        results = {'loss': avg_loss}


        if self.metrics:
            metric_values = self.metrics.compute()
            results.update(metric_values)

        return results

    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """
        Check if early stopping criterion is met.

        Args:
            metrics: Dictionary with validation metrics

        Returns:
            True if training should stop, False otherwise
        """
        if self.early_stopping_patience is None:
            return False

        current_metric = metrics[self.early_stopping_metric]

        # Check if metric improved
        improved = False
        if self.early_stopping_mode == 'min':
            if current_metric < self._best_metric:
                improved = True
                self._best_metric = current_metric
        else:  # mode == 'max'
            if current_metric > self._best_metric:
                improved = True
                self._best_metric = current_metric

        if improved:
            self._early_stopping_counter = 0
        else:
            self._early_stopping_counter += 1

        return self._early_stopping_counter >= self.early_stopping_patience

    def save_checkpoint(
            self,
            epoch: int,
            metrics: Optional[Dict[str, float]] = None,
            is_best: bool = False,
            filename: Optional[str] = None
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
            filename: Custom filename (None for default naming)
        """
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save checkpoint
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pt'

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint

    def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100,
            verbose: bool = True,
            save_frequency: int = 10,
            validate_frequency: int = 1,
            on_epoch_end: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of training epochs
            verbose: Whether to show progress bars and print metrics
            save_frequency: Save checkpoint every N epochs (0 to disable)
            validate_frequency: Validate every N epochs
            on_epoch_end: Callback function called at end of each epoch
                          with signature: on_epoch_end(epoch, train_metrics, val_metrics)

        Returns:
            Training history dictionary
        """

        try:
            for epoch in tqdm(range(1, epochs + 1)):
                # Training
                train_metrics = self._train_epoch(train_loader, epoch, verbose)

                # w&b logging
                if getattr(self, "logger", None) is not None:
                    try:
                        wb_train = {f"train/{k}": float(v) for k, v in train_metrics.items()}
                        try:
                            wb_train["train/lr"] = float(self.optimizer.param_groups[0]["lr"])
                        except Exception:
                            pass
                        self.logger.log_scalars(wb_train, step=epoch)
                    except Exception as e:
                        print(f"w&b train logging failed: {e}")

                self.history['train_loss'].append(train_metrics['loss'])

                for key, value in train_metrics.items():
                    if key != 'loss':
                        history_key = f'train_{key}'
                        if history_key not in self.history:
                            self.history[history_key] = []
                        self.history[history_key].append(value)

                # Validation
                val_metrics = None
                if val_loader is not None and epoch % validate_frequency == 0:
                    val_metrics = self._validate_epoch(val_loader, epoch, verbose)

                    # w&b logging
                    if getattr(self, "logger", None) is not None:
                        try:
                            wb_val = {f"val/{k}": float(v) for k, v in val_metrics.items()}
                            self.logger.log_scalars(wb_val, step=epoch)
                        except Exception as e:
                            print(f"w&b val logging failed: {e}")

                    self.history['val_loss'].append(val_metrics['loss'])

                    # image logging
                    if (
                        getattr(self, "logger", None) is not None
                        and self.log_images_every_n
                        and epoch % self.log_images_every_n == 0
                    ):
                        try:
                            clean, noisy = next(iter(val_loader))
                            clean = clean[: self.max_log_images].to(self.device)
                            noisy = noisy[: self.max_log_images].to(self.device)

                            self.model.eval()
                            with torch.no_grad():
                                denoised = self.model(noisy)

                            if denoised.dim() == 2:
                                denoised_img = denoised.detach().view(len(clean), 1, 28, 28)
                            else:
                                denoised_img = denoised.detach()

                            clean_img = clean.detach()
                            noisy_img = noisy.detach()
                            if hasattr(self.logger, "log_image_examples"):
                                self.logger.log_image_examples(
                                    clean=clean_img.cpu().numpy(),
                                    noisy=noisy_img.cpu().numpy(),
                                    denoised=denoised_img.cpu().numpy(),
                                    step=epoch,
                                    key="examples/denoising",
                                )
                        except Exception as e:
                            print(f"w&b image logging failed: {e}")

                    for key, value in val_metrics.items():
                        if key != 'loss':
                            history_key = f'val_{key}'
                            if history_key not in self.history:
                                self.history[history_key] = []
                            self.history[history_key].append(value)

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        # ReduceLROnPlateau needs a metric
                        metric = val_metrics['loss'] if val_metrics else train_metrics['loss']
                        self.scheduler.step(metric)
                    else:
                        self.scheduler.step()

                # Print epoch summary
                if verbose:
                    summary = f"Epoch {epoch}/{epochs} - Train Loss: {train_metrics['loss']:.6f}"
                    if val_metrics:
                        summary += f" - Val Loss: {val_metrics['loss']:.6f}"
                    print(summary)

                # Checkpointing
                if save_frequency > 0 and epoch % save_frequency == 0:
                    self.save_checkpoint(epoch, val_metrics or train_metrics)
                if val_metrics and self.checkpoint_dir:
                    is_best = (val_metrics[self.early_stopping_metric] == self._best_metric)
                    if is_best:
                        self.save_checkpoint(epoch, val_metrics, is_best=True)

                # Early stopping
                if val_metrics and self._check_early_stopping(val_metrics):
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break

                # Custom callback
                if on_epoch_end is not None:
                    on_epoch_end(epoch, train_metrics, val_metrics)
        
        finally:
            if getattr(self, "logger", None) is not None:
                try:
                    self.logger.close()
                except Exception as e:
                    print(f"w&b close failed: {e}")

        return self.history

    def evaluate(
            self,
            test_loader: DataLoader,
            verbose: bool = True,
            evaluation_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        TODO: Update this method to handle autoencoder style evaluation.

        Args:
            test_loader: DataLoader for test data
            verbose: Whether to show progress bar

        Returns:
            Dictionary with test metrics
        """
        test_metrics = self._validate_epoch(test_loader, 0, verbose, prefix='Test', steps=evaluation_steps)

        if verbose:
            print("\nTest Results:")
            for key, value in test_metrics.items():
                print(f"  {key}: {value:.6f}")

        return test_metrics

    def predict(
            self,
            inputs: torch.Tensor,
            batch_size: Optional[int] = None,
            predict_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Make predictions on input data.
        TODO: Update this method to handle autoencoder style evaluation.

        Args:
            inputs: Input tensor (N, ...)
            batch_size: Batch size for prediction (None for all at once)

        Returns:
            Model predictions
        """
        self.model.eval()

        with torch.no_grad():
            if isinstance(inputs, DataLoader):
                outs = []
                for batch in self._iter_steps(inputs, predict_steps):
                    # batch = (clean, noisy)
                    noisy = batch[1].to(self.device)
                    out = self.model(noisy).detach().cpu()
                    outs.append(out)
                return torch.cat(outs, dim=0) if outs else torch.empty(0)

            x = inputs
            if batch_size is None:
                return self.model(x.to(self.device)).detach().cpu()

            outs = []
            for i in range(0, len(x), batch_size):
                xb = x[i:i + batch_size].to(self.device)
                outs.append(self.model(xb).detach().cpu())
            return torch.cat(outs, dim=0) if outs else torch.empty(0)