<p align="center">
    <br>
    <h3 align="center">Hybrid classical–quantum image denoising with Variational Quantum Algorithms</h3>
    <p align="center">
        Quantum Programming and Platforms final project.
        <br>
        <a href="https://github.com/martabenavente/VQE-Image-denoising/issues/new?template=bug.md">Report bug</a>
        ·
        <a href="https://github.com/martabenavente/VQE-Image-denoising/issues/new?template=feature.md&labels=feature">Request feature</a>
    </p>
</p>

> [!IMPORTANT]
> The final version of the report is available online at https://www.overleaf.com/read/kxydwfxdhqqn#8e4338

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#project-overview">Project overview</a></li>
    <li><a href="#repository-structure">Repository structure</a></li>
    <li><a href="#environment-setup">Environment setup</a></li>
    <li><a href="#how-to-run">How to run</a></li>
    <li><a href="#input-and-output-data">Input and output data</a></li>
    <li><a href="#team-members">Team Members</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>


## Project overview

This repository contains code and experiments for a hybrid classical–quantum approach to image denoising based on a variational quantum circuit embedded in a learning pipeline.

**Objective**: This project serves as a *proof of concept* to explore the feasibility and potential advantages of integrating variational quantum algorithms into image denoising tasks, comparing quantum-enhanced methods against classical baselines.

The project focuses on:
- Training behaviour and stability of the variational optimisation process.
- Denoising performance under different noise levels and model configurations.
- Comparison against a purely classical baseline.


## Repository structure

The project's file structure is organized as follows:

```
├── VQE-Image-denoising/
│   ├── examples/
│   │   ├── load_data_example.py
│   │   ├── ...
│   ├── report/
│   │   ├── Hybrid Classical-Quantum Autoencoder for Image Denoising.pdf
│   ├── src/
│   │   ├── __init__.py
│   │   ├── noise_generation.py
│   │   ├── ...
│   ├── tests/
│   │   ├── test_noise_generation.py
│   │   ├── ...
│   ├── tools/
│   │   ├── plot_wandb_metrics.py
│   │   ├── ...
│   ├── .gitingore
│   ├── README.md
│   ├── environment.yml
│   ├── ...
```

## Environment setup

### Prerequisites

- Python 3.9 or higher
- Conda or pip package manager

### Option 1: Using Conda (Recommended)

1. Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate vqe-denoising
```

### Option 2: Using pip

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Weights & Biases (W&B) Setup

If you want to use Weights & Biases for experiment tracking (optional but recommended):

1. Create a free account at https://wandb.ai/

2. Login to W&B from your terminal:

```bash
wandb login
```

3. When prompted, enter your API key (found at https://wandb.ai/authorize)

**Running without W&B:**

If you don't want to use W&B, simply set `"use_wandb": false` in your config file. The training will proceed normally without experiment tracking.

### Verify Installation

Test your installation by running:

```bash
python -c "import torch; import qiskit; print('Environment setup successful!')"
```

## How to run

### Training the Model

The project includes a training script that runs the full hybrid quantum–classical denoising pipeline using a JSON configuration file.

#### Basic Usage

From the project root, run:

```bash
python -m src.train_model --config path/to/config.json
```

If no config path is specified, it defaults to `config.json` in the project root:

```bash
python -m src.train_model
```

#### Configuration Parameters

All training parameters are controlled through a JSON configuration file. Create a file (e.g., `config.json`) with the following structure:

```json
{
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,
    "early_stopping_patience": 10,
    "save_frequency": 5,
    "num_qubits": 4,
    "num_layers": 1,
    "use_wandb": true,
    "wandb_project": "vqe-image-denoising",
    "wandb_run_name": "CLASSICAL_L1_sigma1_bs16",
    "wandb_log_images_every_n": 1,
    "sigma": 1,
    "seed": 1
}
```

**Parameter descriptions:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `batch_size` | int | Number of samples per training batch | 16 |
| `epochs` | int | Total number of training epochs | 50 |
| `learning_rate` | float | Initial learning rate for Adam optimizer | 0.001 |
| `weight_decay` | float | L2 regularization strength | 0.0001 |
| `gradient_clip` | float | Maximum gradient norm for clipping | 1.0 |
| `early_stopping_patience` | int | Epochs to wait before early stopping | 10 |
| `save_frequency` | int | Save checkpoint every N epochs | 5 |
| `num_qubits` | int | Number of qubits in quantum circuit | 4 |
| `num_layers` | int | Number of layers in quantum ansatz | 1 |
| `use_wandb` | bool | Enable Weights \& Biases logging | true |
| `wandb_project` | string | W\&B project name | "vqe-image-denoising" |
| `wandb_run_name` | string | W\&B run name (optional) | null |
| `wandb_log_images_every_n` | int | Log sample images every N epochs | 1 |
| `sigma` | float | Gaussian noise standard deviation | 1.0 |
| `seed` | int | Random seed for reproducibility | 1 |

#### Example Configurations

**Low noise training:**

```json
{
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "num_qubits": 4,
    "num_layers": 2,
    "sigma": 0.5,
    "use_wandb": false,
    "seed": 42
}
```

**High noise with more quantum layers:**

```json
{
    "batch_size": 16,
    "epochs": 150,
    "learning_rate": 0.0005,
    "num_qubits": 6,
    "num_layers": 3,
    "sigma": 2.0,
    "wandb_run_name": "high_noise_L3_q6",
    "seed": 42
}
```

#### Output

Training outputs are saved to `checkpoints/YYYYMMDD_HHMMSS/`:
- `config.json`: Copy of the configuration used
- `best_model.pt`: Best model checkpoint based on validation loss
- `training_history.pkl`: Training metrics history
- `model_epoch_N.pt`: Periodic checkpoints (based on `save_frequency`)

#### General Workflow

1. **Load/preprocess the dataset** \- MNIST images are automatically loaded
2. **Apply noise** \- Gaussian noise with specified `sigma` is added
3. **Train the model** \- Hybrid quantum\-classical denoising network
4. **Evaluate performance** \- Metrics (PSNR, SSIM) computed on validation set


## Input and output data

### Input
- Dataset: MNIST.
- Noise level: Gaussian noise with mean = 0 and standard deviation = 0.2.

### Output
- Denoised images from both models (hybrid and baseline).
- Metrics such as PSNR, SSIM, etc., comparing the original clean images with their noisy counterparts and the denoised ones obtained by each model.

## Model architecture
The hybrid quantum-classical autoencoder consists of three main components:

### 1. Classical Encoder
- **Architecture**: Convolutional neural network
- **Layers**:
  - Conv2D (1→32 filters) + BatchNorm + LeakyReLU + MaxPool
  - Conv2D (32→4 filters) + BatchNorm + LeakyReLU + MaxPool
  - Conv2D (4→4 filters) + BatchNorm + LeakyReLU + MaxPool
  - Fully connected layer → `num_qubits` features
  - Tanh activation (normalize to [-1, 1] for quantum encoding)
- **Purpose**: Compress 28×28 images into quantum-compatible feature vectors

### 2. Quantum Processing Unit (QPU) / Classical Neck
Two variants are available:

**Quantum Version:**
- **Circuit**: Variational Quantum Algorithm
- **Qubits**: Configurable (default: 4)
- **Layers**: Configurable ansatz depth (default: 1)
- **Encoding**: AngleEmbedding maps classical features to qubit rotations
- **Processing**: Variational layers with trainable parameters
- **Measurement**: Pauli-Z expectation values on all qubits
- **Output**: Quantum measurements → Linear projection → 36 features

**Classical Baseline Version:**
- **Architecture**: Fully connected neural network
- **Layers**:
  - Linear (`num_qubits` → 16) + LeakyReLU
  - Linear (16 → 16) + LeakyReLU
  - Linear (16 → 36)
- **Purpose**: Classical alternative for fair comparison with comparable parameter budget

### 3. Classical Decoder
- **Architecture**: Transposed convolutional neural network
- **Layers**:
  - ConvTranspose2D (4→4 filters) + BatchNorm + LeakyReLU
  - ConvTranspose2D (4→32 filters) + BatchNorm + LeakyReLU
  - ConvTranspose2D (32→1 filters) + Sigmoid
- **Purpose**: Reconstruct 28×28 denoised images from processed features

### Architecture Diagram
The following diagram illustrates the overall architecture of the hybrid classical-quantum autoencoder:

<p align="center">
  <img src="tools/Imgs/Quantum_autoencoder_diagram.png" width="800">
</p>

## Team Members

1. **Yeray Cordero**
   - GitHub: <https://github.com/yeray142/>
   - LinkedIn: <https://linkedin.com/in/yeray142/>

2. **Marta Benavente**
   - GitHub: <https://github.com/martabenavente>
   - LinkedIn: <https://www.linkedin.com/in/marta-benavente-vilas-5330532a7/>

## License

Distributed under the MIT License. See `LICENSE` for more information.
