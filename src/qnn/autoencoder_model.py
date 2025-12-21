import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit_machine_learning.connectors import TorchConnector

from src.circuits.circuit_base import CircuitBase
from src.qnn.qnn_builder import QNNBuilder


class Encoder(nn.Module):
    """
    Convolutional Encoder to encode input data into a quantum-compatible format.

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.
        n_filters (int): Number of filters for the convolutional layers.
    """
    def __init__(self, num_qubits: int, n_filters: int = 32):
        super(Encoder, self).__init__()
        self.num_qubits = num_qubits
        self.n_filters = n_filters

        self.conv1 = nn.Conv2d(1, self.n_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.n_filters)

        self.conv2 = nn.Conv2d(self.n_filters, 4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)

        self.conv3 = nn.Conv2d(4, 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(4)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc2 = nn.Linear(36, num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, num_qubits).
        """
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = nn.Flatten()(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x

class Decoder(nn.Module):
    """
    Convolutional Decoder to reconstruct data from quantum-processed format.

    Args:
        n_filters (int): Number of filters for the convolutional layers.
    """
    def __init__(self, n_filters: int = 32):
        super(Decoder, self).__init__()
        self.n_filters = n_filters

        self.t_conv0 = nn.ConvTranspose2d(4, 4, 3, stride=2)
        self.bn0 = nn.BatchNorm2d(4)

        self.t_conv1 = nn.ConvTranspose2d(4, self.n_filters, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(self.n_filters)

        self.t_conv2 = nn.ConvTranspose2d(self.n_filters, 1, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 4, 3, 3).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 1, height, width).
        """
        x = F.leaky_relu(self.bn0(self.t_conv0(x)))
        x = F.leaky_relu(self.bn1(self.t_conv1(x)))
        x = F.sigmoid(self.t_conv2(x))
        return x


class QuantumProcessingUnit(nn.Module):
    """
    Quantum Processing Unit using VQA (Variational Quantum Algorithm).
    Acts as the neck between encoder and decoder in the autoencoder.

    Args:
        circuit_base (CircuitBase): Circuit instance with AngleEmbedding for data encoding.
        num_output_features (int): Number of output features (e.g., 36 for decoder input).
    """

    def __init__(self, circuit_base: CircuitBase, num_output_features: int = 36):
        super(QuantumProcessingUnit, self).__init__()

        self.circuit_base = circuit_base
        self.num_qubits = circuit_base.num_qubits

        qnn_builder = QNNBuilder(circuit_base=circuit_base)
        qnn = qnn_builder.build_qnn(input_gradients=True)
        self.quantum_layer = TorchConnector(qnn)

        num_observables = len(qnn_builder.observable)
        self.output_projection = nn.Linear(num_observables, num_output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum layer.

        Args:
            x (torch.Tensor): Encoded input from encoder, shape (batch_size, num_qubits).

        Returns:
            torch.Tensor: Quantum-processed output, shape (batch_size, 4, 3, 3).
        """
        # TODO: Check if reshaping is necessary based on input dimensions
        x = x.view(-1, self.num_qubits)

        x = self.quantum_layer(x)
        x = self.output_projection(x)
        x = x.view(-1, 4, 3, 3)
        return x


class ClassicalNeck(nn.Module):
    """
    Classical Processing Unit using fully connected layers.
    Acts as the neck between encoder and decoder in the classical baseline autoencoder.

    Args:
        num_qubits (int): Number of input features (matching quantum model for fair comparison).
        num_output_features (int): Number of output features (e.g., 36 for decoder input).
        hidden_dim (int): Hidden layer dimension. Default is 64.
    """

    def __init__(self, num_qubits: int, num_output_features: int = 36, hidden_dim: int = 16):
        super(ClassicalNeck, self).__init__()
        self.num_qubits = num_qubits

        self.fc1 = nn.Linear(num_qubits, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classical layers.

        Args:
            x (torch.Tensor): Encoded input from encoder, shape (batch_size, num_qubits).

        Returns:
            torch.Tensor: Processed output, shape (batch_size, 4, 3, 3).
        """
        x = x.view(-1, self.num_qubits)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(-1, 4, 3, 3)
        return x


class ConvDenoiseNet(nn.Module):
    """
    Hybrid Convolutional Autoencoder with Quantum Layer for Denoising.

    Args:
        n_filters (int): Number of filters for convolutional layers.
        circuit (CircuitBase): Circuit instance with AngleEmbedding for data encoding.
    """
    def __init__(self, n_filters: int = 32, circuit: CircuitBase = None, quantum: bool = True):
        super(ConvDenoiseNet, self).__init__()

        # Encoder
        self.encoder = Encoder(num_qubits=circuit.num_qubits, n_filters=n_filters)

        # Quantum layer parameters
        if quantum:
            self.qpu = QuantumProcessingUnit(circuit_base=circuit, num_output_features=36)
        else:
            self.qpu = ClassicalNeck(num_qubits=circuit.num_qubits, num_output_features=36)

        # Decoder
        self.decoder = Decoder(n_filters=n_filters)

    def forward(self, x):
        """
        Forward pass through the entire autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 1, height, width).
        """
        x = self.encoder(x)
        x = self.qpu(x)
        x = self.decoder(x)
        return x
