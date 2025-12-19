import torch
import numpy as np

from skimage.metrics import structural_similarity
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
        psnr = self._calculate_psnr(denoised, clean)
        ssim = self._calculate_ssim(denoised, clean)
        mse = _calculate_mse(denoised, clean)
        mae = _calculate_mae(denoised, clean)

        # Accumulate
        self.metrics_sum['psnr'] += psnr * batch_size
        self.metrics_sum['ssim'] += ssim * batch_size
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
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = torch.mean((denoised - clean) ** 2)
        if mse == 0:
            return 100.0  # Perfect match
        psnr = 20 * torch.log10(self.data_range / torch.sqrt(mse))
        return psnr.item()

    def _calculate_ssim(self, denoised: torch.Tensor, clean: torch.Tensor) -> float:
        """Calculate Structural Similarity Index."""
        # Move to CPU and convert to numpy
        denoised_np = denoised.detach().cpu().numpy()
        clean_np = clean.detach().cpu().numpy()

        # Determine appropriate window size based on image size
        _, _, height, width = denoised_np.shape
        min_dim = min(height, width)

        # SSIM requires odd window size, max 7, must be <= min dimension
        if min_dim >= 7:
            win_size = 7
        elif min_dim >= 5:
            win_size = 5
        elif min_dim >= 3:
            win_size = 3
        else:
            # For very small patches, return MSE-based approximation
            return 1.0 - _calculate_mse(denoised, clean)

        ssim_values = []
        for i in range(denoised_np.shape[0]):
            # Handle grayscale vs RGB
            if denoised_np.shape[1] == 1:  # Grayscale
                img1 = denoised_np[i, 0]
                img2 = clean_np[i, 0]
                ssim = structural_similarity(
                    img1, img2,
                    data_range=self.data_range,
                    win_size=win_size
                )
            else:  # RGB or multi-channel
                # Transpose from (C, H, W) to (H, W, C)
                img1 = np.transpose(denoised_np[i], (1, 2, 0))
                img2 = np.transpose(clean_np[i], (1, 2, 0))
                ssim = structural_similarity(
                    img1, img2,
                    data_range=self.data_range,
                    channel_axis=-1,
                    win_size=win_size
                )
            ssim_values.append(ssim)

        return float(np.mean(ssim_values))

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