import numpy as np
from src.metrics import compute_image_metrics

def test_metrics_improve_when_denoised_equals_clean():
    clean = np.zeros((28, 28), dtype=np.float32)
    noisy = clean.copy()
    noisy[0, 0] = 1.0  ## simple noise
    denoised = clean.copy()

    m = compute_image_metrics(clean=clean, noisy=noisy, denoised=denoised, data_range=1.0)

    assert m["mse_denoised"] <= m["mse_noisy"]
    assert m["psnr_denoised"] >= m["psnr_noisy"]
    assert m["ssim_denoised"] >= m["ssim_noisy"]
