import pytest
import torch

from src.utils.metrics import DenoisingMetrics, _calculate_mse, _calculate_mae


class TestHelperFunctions:
    """Test helper functions for metrics calculation."""

    def test_calculate_mse_zero(self):
        """Test MSE is zero for identical tensors."""
        tensor = torch.randn(2, 3, 32, 32)
        mse = _calculate_mse(tensor, tensor)
        assert mse == 0.0

    def test_calculate_mse_known_value(self):
        """Test MSE with known values."""
        denoised = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        clean = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        mse = _calculate_mse(denoised, clean)
        assert pytest.approx(mse, abs=1e-6) == 0.25

    def test_calculate_mae_zero(self):
        """Test MAE is zero for identical tensors."""
        tensor = torch.randn(2, 3, 32, 32)
        mae = _calculate_mae(tensor, tensor)
        assert mae == 0.0

    def test_calculate_mae_known_value(self):
        """Test MAE with known values."""
        denoised = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        clean = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        mae = _calculate_mae(denoised, clean)
        assert pytest.approx(mae, abs=1e-6) == 0.5


class TestDenoisingMetrics:
    """Test DenoisingMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create a DenoisingMetrics instance."""
        return DenoisingMetrics(data_range=1.0)

    def test_initialization(self, metrics):
        """Test proper initialization."""
        assert metrics.data_range == 1.0
        assert metrics.count == 0
        assert all(v == 0.0 for v in metrics.metrics_sum.values())

    def test_reset(self, metrics):
        """Test reset functionality."""
        metrics.count = 10
        metrics.metrics_sum['psnr'] = 25.0
        metrics.reset()
        assert metrics.count == 0
        assert all(v == 0.0 for v in metrics.metrics_sum.values())

    def test_compute_empty(self, metrics):
        """Test compute returns zeros when no data."""
        result = metrics.compute()
        assert all(v == 0.0 for v in result.values())

    def test_update_and_compute(self, metrics):
        """Test update and compute with identical images."""
        denoised = torch.randn(4, 3, 32, 32)
        clean = denoised.clone()

        metrics.update(denoised, clean)
        result = metrics.compute()

        assert result['mse'] == 0.0
        assert result['mae'] == 0.0
        assert result['psnr'] == 100.0  # Perfect match
        assert 0.99 <= result['ssim'] <= 1.0  # SSIM should be close to 1

    def test_update_multiple_batches(self, metrics):
        """Test accumulation over multiple batches."""
        batch1 = torch.ones(2, 1, 16, 16)
        batch2 = torch.zeros(3, 1, 16, 16)

        metrics.update(batch1, batch1)
        metrics.update(batch2, batch2)

        assert metrics.count == 5
        result = metrics.compute()
        assert all(isinstance(v, float) for v in result.values())

    def test_psnr_calculation(self, metrics):
        """Test PSNR calculation."""
        denoised = torch.ones(1, 1, 10, 10) * 0.9
        clean = torch.ones(1, 1, 10, 10)

        psnr = metrics._calculate_psnr(denoised, clean)
        assert psnr > 0
        assert isinstance(psnr, float)

    def test_ssim_grayscale(self, metrics):
        """Test SSIM calculation for grayscale images."""
        denoised = torch.randn(2, 1, 32, 32)
        clean = denoised + 0.01 * torch.randn_like(denoised)

        ssim = metrics._calculate_ssim(denoised, clean)
        assert 0.0 <= ssim <= 1.0
        assert isinstance(ssim, float)

    def test_ssim_rgb(self, metrics):
        """Test SSIM calculation for RGB images."""
        denoised = torch.randn(2, 3, 32, 32)
        clean = denoised + 0.01 * torch.randn_like(denoised)

        ssim = metrics._calculate_ssim(denoised, clean)
        assert 0.0 <= ssim <= 1.0
        assert isinstance(ssim, float)

    def test_different_data_ranges(self):
        """Test metrics with different data ranges."""
        metrics_normalized = DenoisingMetrics(data_range=1.0)
        metrics_255 = DenoisingMetrics(data_range=255.0)

        assert metrics_normalized.data_range == 1.0
        assert metrics_255.data_range == 255.0

    def test_compute_single_batch(self):
        """Test static method for single batch computation."""
        denoised = torch.randn(4, 3, 32, 32)
        clean = denoised + 0.1 * torch.randn_like(denoised)

        result = DenoisingMetrics.compute_single_batch(
            denoised, clean, data_range=1.0
        )

        assert 'psnr' in result
        assert 'ssim' in result
        assert 'mse' in result
        assert 'mae' in result
        assert all(isinstance(v, float) for v in result.values())

    def test_metrics_with_noise(self, metrics):
        """Test metrics with realistic noisy/denoised images."""
        clean = torch.rand(2, 3, 32, 32)
        noise = torch.randn_like(clean) * 0.1

        denoised = clean + noise * 0.5  # Partial denoising

        metrics.update(denoised, clean)
        result = metrics.compute()

        # Denoised should be better than noisy
        assert result['mse'] > 0
        assert result['mae'] > 0
        assert result['psnr'] < 100.0
        assert result['ssim'] < 1.0

    def test_batch_size_consistency(self, metrics):
        """Test that batch size is correctly tracked."""
        metrics.update(torch.randn(5, 3, 16, 16), torch.randn(5, 3, 16, 16))
        assert metrics.count == 5

        metrics.update(torch.randn(3, 3, 16, 16), torch.randn(3, 3, 16, 16))
        assert metrics.count == 8