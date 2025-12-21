import wandb
import numpy as np

from typing import Dict, Optional


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
        self.run.log(payload, step=step)

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
        images = []
        for i in range(min(max_images, len(clean))):
            # Squeeze single-channel dimensions if present
            clean_img = np.squeeze(clean[i])

            caption = f"Sample {i}"

            # Create side-by-side comparison
            if noisy is not None and denoised is not None:
                noisy_img = np.squeeze(noisy[i])
                denoised_img = np.squeeze(denoised[i])

                # Concatenate horizontally: noisy | denoised | clean
                comparison = np.concatenate([noisy_img, denoised_img, clean_img], axis=1)
                images.append(wandb.Image(comparison, caption=caption))
            elif denoised is not None:
                denoised_img = np.squeeze(denoised[i])
                comparison = np.concatenate([denoised_img, clean_img], axis=1)
                images.append(wandb.Image(comparison, caption=caption))
            else:
                images.append(wandb.Image(clean_img, caption=caption))

        self.run.log({key: images}, step=step)

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
            self.run.log({
                f"{prefix}{key}": wandb.plot.line(
                    table, "epoch", "value", title=key
                )
            })

    def close(self):
        self.run.finish()
