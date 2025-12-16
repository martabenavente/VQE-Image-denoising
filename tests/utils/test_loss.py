import pytest
import torch

from src.utils.loss import DenoisingLoss


class TestDenoisingLoss:
    """Test suite for DenoisingLoss class."""

    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for testing."""
        batch_size = 2
        channels = 3
        height = 32
        width = 32

        denoised = torch.randn(batch_size, channels, height, width)
        clean = torch.randn(batch_size, channels, height, width)

        return denoised, clean

    def test_mse_loss(self, sample_tensors):
        """Test MSE loss computation."""
        denoised, clean = sample_tensors
        loss_fn = DenoisingLoss(loss_type='mse')

        loss = loss_fn(denoised, clean)
        expected_loss = torch.nn.functional.mse_loss(denoised, clean)

        assert torch.isclose(loss, expected_loss)
        assert loss.dim() == 0  # Scalar

    def test_l1_loss(self, sample_tensors):
        """Test L1 loss computation."""
        denoised, clean = sample_tensors
        loss_fn = DenoisingLoss(loss_type='l1')

        loss = loss_fn(denoised, clean)
        expected_loss = torch.nn.functional.l1_loss(denoised, clean)

        assert torch.isclose(loss, expected_loss)
        assert loss.dim() == 0  # Scalar

    def test_combined_loss(self, sample_tensors):
        """Test combined loss computation."""
        denoised, clean = sample_tensors
        alpha = 0.84
        loss_fn = DenoisingLoss(loss_type='combined', alpha=alpha)

        loss = loss_fn(denoised, clean)

        mse = torch.nn.functional.mse_loss(denoised, clean)
        l1 = torch.nn.functional.l1_loss(denoised, clean)
        expected_loss = alpha * mse + (1 - alpha) * l1

        assert torch.isclose(loss, expected_loss)
        assert loss.dim() == 0  # Scalar

    def test_combined_loss_custom_alpha(self, sample_tensors):
        """Test combined loss with custom alpha value."""
        denoised, clean = sample_tensors
        alpha = 0.5
        loss_fn = DenoisingLoss(loss_type='combined', alpha=alpha)

        loss = loss_fn(denoised, clean)

        mse = torch.nn.functional.mse_loss(denoised, clean)
        l1 = torch.nn.functional.l1_loss(denoised, clean)
        expected_loss = alpha * mse + (1 - alpha) * l1

        assert torch.isclose(loss, expected_loss)

    def test_invalid_loss_type(self, sample_tensors):
        """Test that invalid loss type raises ValueError."""
        denoised, clean = sample_tensors
        loss_fn = DenoisingLoss(loss_type='invalid')

        with pytest.raises(ValueError, match="Unknown loss type: invalid"):
            loss_fn(denoised, clean)

    def test_default_parameters(self):
        """Test default parameter initialization."""
        loss_fn = DenoisingLoss()

        assert loss_fn.loss_type == 'mse'
        assert loss_fn.alpha == 0.84

    def test_loss_is_non_negative(self, sample_tensors):
        """Test that loss values are non-negative."""
        denoised, clean = sample_tensors

        for loss_type in ['mse', 'l1', 'combined']:
            loss_fn = DenoisingLoss(loss_type=loss_type)
            loss = loss_fn(denoised, clean)
            assert loss >= 0

    def test_zero_loss_identical_tensors(self):
        """Test that identical tensors produce zero loss."""
        tensor = torch.randn(2, 3, 32, 32)

        for loss_type in ['mse', 'l1', 'combined']:
            loss_fn = DenoisingLoss(loss_type=loss_type)
            loss = loss_fn(tensor, tensor)
            assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_gradient_flow(self, sample_tensors):
        """Test that gradients can flow through the loss."""
        denoised, clean = sample_tensors
        denoised.requires_grad = True

        loss_fn = DenoisingLoss(loss_type='combined')
        loss = loss_fn(denoised, clean)
        loss.backward()

        assert denoised.grad is not None
        assert denoised.grad.shape == denoised.shape