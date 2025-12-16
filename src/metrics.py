from __future__ import annotations
import os
import csv
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np


def mse(a: np.ndarray, b: np.ndarray):
    """
    Compute Mean Squared Error (MSE) between two images.

    Args:
        a: First image array
        b: Second image array
    """

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        a: First image array
        b: Second image array
        data_range: Maximum possible pixel value
    """

    err = mse(a, b)
    if err == 0.0:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(err))


def ssim_metric(a: np.ndarray, b: np.ndarray, data_range: float = 1.0):
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Args:
        a: First image array
        b: Second image array
        data_range: Maximum possible pixel value (1.0 if normalized)
    """

    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError as e:
        raise ImportError("Instala scikit-image: pip install scikit-image") from e

    return float(ssim(a, b, data_range=data_range))



## (Optional) Quite useful if we binarize, otherwise it doesn't make sense.
def pixel_accuracy(a: np.ndarray, b: np.ndarray, threshold: float = 0.5):
    """
    Compute pixel-wise accuracy between two images after binarization.

    Args:
        a: First image array
        b: Second image array
        threshold: Threshold for binarization
    """

    aa = (a >= threshold).astype(np.int32)
    bb = (b >= threshold).astype(np.int32)
    return float(np.mean(aa == bb))


def compute_image_metrics(clean: np.ndarray, noisy: Optional[np.ndarray] = None, denoised: Optional[np.ndarray] = None,  
                          data_range: float = 1.0, threshold: float = 0.5,): 
    """
    Compute a set of image quality metrics for denoising evaluation.

    Args:
        clean: Ground-truth clean image
        noisy: Noisy image (optional)
        denoised: Denoised image (optional)
        data_range: Maximum possible pixel value
        threshold: Threshold for pixel accuracy
    """
     
    metrics: Dict[str, float] = {}

    if noisy is not None:
        metrics["mse_noisy"] = mse(noisy, clean)
        metrics["psnr_noisy"] = psnr(noisy, clean, data_range=data_range)
        metrics["ssim_noisy"] = ssim_metric(noisy, clean, data_range)
        metrics["acc_noisy"] = pixel_accuracy(noisy, clean, threshold=threshold)

    if denoised is not None:
        metrics["mse_denoised"] = mse(denoised, clean)
        metrics["psnr_denoised"] = psnr(denoised, clean, data_range=data_range)
        metrics["ssim_denoised"] = ssim_metric(denoised, clean, data_range)
        metrics["acc_denoised"] = pixel_accuracy(denoised, clean, threshold=threshold)

    if ("mse_noisy" in metrics) and ("mse_denoised" in metrics):
        metrics["mse_improvement"] = metrics["mse_noisy"] - metrics["mse_denoised"]
    if ("psnr_noisy" in metrics) and ("psnr_denoised" in metrics):
        metrics["psnr_improvement"] = metrics["psnr_denoised"] - metrics["psnr_noisy"]
    if ("ssim_noisy" in metrics) and ("ssim_denoised" in metrics):
        metrics["ssim_improvement"] = metrics["ssim_denoised"] - metrics["ssim_noisy"]

    return metrics


@dataclass
class CSVMetricLogger:
    """
    Logger for saving metrics to a CSV file.
    """

    csv_path: str
    fieldnames: Optional[list[str]] = None

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = None
        self._wrote_header = os.path.getsize(self.csv_path) > 0

    def log(self, **row: Any):
        if self._writer is None:
            if self.fieldnames is None:
                self.fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            if not self._wrote_header:
                self._writer.writeheader()
                self._wrote_header = True

        if self.fieldnames is not None:
            row = {k: row.get(k, None) for k in self.fieldnames}

        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class WandBMetricLogger:
    """
    Logger for metrics and images using Weights & Biases (wandb).
    """
    def __init__(self, project: str, name: Optional[str] = None, config: Optional[dict] = None):
        """
        Initialize a wandb run.

        Args:
            project: W&B project name
            name: Optional run name
            config: Optional configuration dictionary
        """

        try:
            import wandb  # type: ignore
        except ImportError as e:
            raise ImportError("Instala wandb con: pip install wandb") from e

        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, config=config)

    def log(self, step: Optional[int] = None, **metrics: Any):
        """
        Log scalar metrics to wandb.

        Args:
            step: Optional step index
            **metrics: Key-value metric pairs
        """

        if step is None:
            self.wandb.log(metrics)
        else:
            self.wandb.log(metrics, step=step)


    def close(self):
        self.run.finish()
