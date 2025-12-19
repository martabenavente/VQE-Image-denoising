import wandb
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


class WandBMetricLogger:
    """
    Logger utility for Weights & Biases.
    Handles scalar metrics, histories, and image examples.
    """

    def __init__(self, project: str, name: Optional[str] = None, config: Optional[dict] = None):
        self.run = wandb.init(project=project, name=name, config=config)

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """
        Log scalar metrics to Weights & Biases.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional training step or epoch.
        """
        
        payload = {f"{prefix}{k}": v for k, v in metrics.items()}
        wandb.log(payload, step=step)

    def log_image_examples(self, clean: np.ndarray, noisy: Optional[np.ndarray] = None, denoised: Optional[np.ndarray] = None, 
                           max_images: int = 8, step: Optional[int] = None, key: str = "examples"):
        """
        Log example images to Weights & Biases.

        Args:
            clean: Clean reference images
            noisy: Noisy input images (optional)
            denoised: Denoised output images (optional)
            step: Optional training step.
        """
        
        rows = []
        for i in range(min(max_images, len(clean))):
            row = {"clean": wandb.Image(clean[i])}
            if noisy is not None:
                row["noisy"] = wandb.Image(noisy[i])
            if denoised is not None:
                row["denoised"] = wandb.Image(denoised[i])
            rows.append(row)

        wandb.log({key: rows}, step=step)

    ## This should be used as a summary to visualize COMPLETE histories at the end of training/eval,
    ## not during it as w&b already creates plots when scalars are logged.
    def log_history(self, history: Dict[str, list], prefix: str = "history/"):
        """
        Log training or evaluation history to Weights & Biases as line plots.

        Args:
            history: Dictionary mapping metric names to a list of values
                recorded over epochs or steps.
                Example:
                    {
                        "train/loss": [0.9, 0.7, 0.5],
                        "val/loss": [1.0, 0.8, 0.6],
                        "train/psnr": [...],
                    }
            prefix: Prefix added to each metric name in W&B.
        """

        for key, values in history.items():
            table = wandb.Table(
                data=[[i, v] for i, v in enumerate(values)],
                columns=["epoch", "value"]
            )
            wandb.log({
                f"{prefix}{key}": wandb.plot.line(
                    table, "epoch", "value", title=key
                )
            })

    def close(self):
        self.run.finish()
