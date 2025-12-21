import torch
import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Dict


def _calculate_mse(denoised: torch.Tensor, clean: torch.Tensor) -> float:
    """Calculate Mean Squared Error."""
    return torch.mean((denoised - clean) ** 2).item()


def _calculate_mae(denoised: torch.Tensor, clean: torch.Tensor) -> float:
    """Calculate Mean Absolute Error."""
    return torch.mean(torch.abs(denoised - clean)).item()


class DenoisingMetrics:
    """Metrics calculator for image denoising evaluation."""

    def __init__(self, data_range: float = 1.0):
        """
        Args:
            data_range: Range of pixel values (1.0 for normalized [0,1], 255 for [0,255])
        """
        self.count = None
        self.metrics_sum = None
        self.reset()

        self.data_range = data_range

    def reset(self):
        """Reset accumulated metrics."""
        self.metrics_sum = {
            'psnr': 0.0,
            'ssim': 0.0,
            'mse': 0.0,
            'mae': 0.0
        }
        self.count = 0

    def update(self, denoised: torch.Tensor, clean: torch.Tensor):
        """
        Update metrics with a new batch.

        Args:
            denoised: Model output (B, C, H, W)
            clean: Ground truth clean images (B, C, H, W)
        """
        # Handle flattened patches (B, features) -> reshape to (B, 1, sqrt(features), sqrt(features))
        if denoised.dim() == 2:
            batch_size, num_features = denoised.shape
            patch_size = int(num_features ** 0.5)

            if patch_size * patch_size != num_features:
                raise ValueError(f"Cannot reshape {num_features} features into square patch")

            denoised = denoised.view(batch_size, 1, patch_size, patch_size)
            clean = clean.view(batch_size, 1, patch_size, patch_size)

        batch_size = denoised.shape[0]

        # Calculate batch metrics
        psnr_val = self._calculate_psnr(denoised, clean)
        ssim_val = self._calculate_ssim(denoised, clean)
        mse = _calculate_mse(denoised, clean)
        mae = _calculate_mae(denoised, clean)

        # Accumulate
        self.metrics_sum['psnr'] += psnr_val * batch_size
        self.metrics_sum['ssim'] += ssim_val * batch_size
        self.metrics_sum['mse'] += mse * batch_size
        self.metrics_sum['mae'] += mae * batch_size
        self.count += batch_size

    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics over all accumulated batches.

        Returns:
            Dictionary with metric names and values
        """
        if self.count == 0:
            return {k: 0.0 for k in self.metrics_sum.keys()}

        return {
            metric: value / self.count
            for metric, value in self.metrics_sum.items()
        }

    def _calculate_psnr(self, denoised: torch.Tensor, clean: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio using skimage."""
        batch_psnr = 0.0

        for i in range(len(denoised)):
            org = np.transpose(clean[i], (1, 2, 0)).detach().cpu().numpy()
            denoise = np.transpose(denoised[i], (1, 2, 0)).detach().cpu().numpy()
            batch_psnr += psnr(org, denoise, data_range=self.data_range)

        return batch_psnr / len(denoised)

    def _calculate_ssim(self, denoised: torch.Tensor, clean: torch.Tensor) -> float:
        """Calculate Structural Similarity Index using skimage."""
        batch_ssim = 0.0

        for i in range(len(denoised)):
            org = np.transpose(clean[i], (1, 2, 0)).detach().cpu().numpy()
            denoise = np.transpose(denoised[i], (1, 2, 0)).detach().cpu().numpy()

            # Handle grayscale (H, W, 1) by squeezing
            if org.shape[2] == 1:
                org = org.squeeze(axis=2)
                denoise = denoise.squeeze(axis=2)
                batch_ssim += ssim(org, denoise, data_range=self.data_range)
            else:
                batch_ssim += ssim(org, denoise, data_range=self.data_range, channel_axis=2)

        return batch_ssim / len(denoised)

    @staticmethod
    def compute_single_batch(
            denoised: torch.Tensor,
            clean: torch.Tensor,
            data_range: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute metrics for a single batch without accumulation.

        Args:
            denoised: Model output (B, C, H, W)
            clean: Ground truth clean images (B, C, H, W)
            data_range: Range of pixel values

        Returns:
            Dictionary with metric values
        """
        metrics = DenoisingMetrics(data_range=data_range)
        metrics.update(denoised, clean)
        return metrics.compute()