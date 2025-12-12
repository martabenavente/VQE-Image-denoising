import torch
import torch.nn as nn


class DenoisingLoss(nn.Module):
    """
    Loss function for image denoising tasks.

    Args:
        loss_type: Type of loss ('mse', 'l1', 'combined')
        alpha: Weight for MSE in combined loss (1-alpha for L1)
    """
    def __init__(self, loss_type: str = 'mse', alpha: float = 0.84):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, denoised: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between denoised and clean images.

        Args:
            denoised: Model output (B, C, H, W)
            clean: Ground truth clean images (B, C, H, W)

        Returns:
            Loss value as scalar tensor
        """
        if self.loss_type == 'mse':
            return self.mse_loss(denoised, clean)
        elif self.loss_type == 'l1':
            return self.l1_loss(denoised, clean)
        elif self.loss_type == 'combined':
            mse = self.mse_loss(denoised, clean)
            l1 = self.l1_loss(denoised, clean)
            return self.alpha * mse + (1 - self.alpha) * l1
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")