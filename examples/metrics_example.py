import numpy as np
from src.metrics import compute_image_metrics, CSVMetricLogger

def main():
    clean = np.zeros((28, 28), dtype=np.float32)
    noisy = clean.copy()
    noisy[0, 0] = 1.0
    denoised = clean.copy()

    metrics = compute_image_metrics(clean=clean, noisy=noisy, denoised=denoised, data_range=1.0)
    print(metrics)

    with CSVMetricLogger("runs/exp00/metrics.csv") as logger:
        logger.log(step=0, **metrics)

if __name__ == "__main__":
    main()