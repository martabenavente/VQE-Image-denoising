import numpy as np
import torch

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from src.circuits.encoding_circuit import AngleEmbeddingCircuit


# Create a concrete implementation by defining the ansatz
class SimpleAngleEmbedding(AngleEmbeddingCircuit):
    """Simple implementation with a basic ansatz."""

    def ansatz_layer(self) -> QuantumCircuit:
        """Simple ansatz with RX rotations and CNOT entanglement."""
        qc = QuantumCircuit(self.num_qubits)

        # Apply trainable rotations
        for i in range(min(self.num_parameters, self.num_qubits)):
            qc.rx(self.weight_params[i], i)

        # Add entanglement (if more than 1 qubit)
        if self.num_qubits > 1:
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)

        return qc


# Example 1: Basic usage with 4 qubits and Y rotation
print("Example 1: Basic 4-qubit circuit with Y rotation")
print("=" * 60)

circuit = SimpleAngleEmbedding(
    num_qubits=4,
    num_features=4,
    num_parameters=4,
    rotation='Y',
    use_hadamard=True
)

# Build the parametrized circuit
qc = circuit.build_circuit()
qc.draw(output='mpl')
plt.title('Parametrized circuit')
plt.show()

# Bind parameters with actual values
feature_data = np.array([0.5, 1.0, 1.5, 2.0])  # Input features
weight_data = np.array([0.1, 0.2, 0.3, 0.4])   # Trainable weights

param_dict = {
    **{circuit.feature_params[i]: feature_data[i] for i in range(4)},
    **{circuit.weight_params[i]: weight_data[i] for i in range(4)}
}

bound_circuit = qc.assign_parameters(param_dict)
bound_circuit.draw('mpl')
plt.title('Bound circuit with actual values')
plt.show()

# Example 2: Using with PyTorch tensors
print("\n\nExample 2: Using PyTorch tensors")
print("=" * 60)

circuit2 = SimpleAngleEmbedding(
    num_qubits=4,
    num_features=4,
    num_parameters=4,
    rotation='X',
    use_hadamard=False  # No Hadamard gates
)

qc2 = circuit2.build_circuit()

# Convert torch tensor to numpy for binding
feature_tensor = torch.tensor([0.0, np.pi/4, np.pi/2, np.pi])
weight_tensor = torch.randn(4)

param_dict2 = {
    **{circuit2.feature_params[i]: feature_tensor[i].item() for i in range(4)},
    **{circuit2.weight_params[i]: weight_tensor[i].item() for i in range(4)}
}

bound_circuit2 = qc2.assign_parameters(param_dict2)
bound_circuit2.draw('mpl')
plt.title('Bound circuit with PyTorch tensor values')
plt.show()