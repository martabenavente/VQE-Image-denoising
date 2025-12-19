from qiskit import QuantumCircuit
from typing import Literal

from src.circuits.encoding_circuit import AngleEmbeddingCircuit


class SimpleAnsatzCircuit(AngleEmbeddingCircuit):
    """
    Simple variational ansatz on top of angle embedding.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_features (int): Number of input features to encode.
        num_parameters (int): Number of trainable parameters in the ansatz.
        rotation (str): Rotation gate type ('X', 'Y', or 'Z'). Default 'Y'.
        use_hadamard (bool): Apply Hadamard gates before encoding. Default True.
    """

    def __init__( self, num_qubits: int, num_features: int, num_parameters: int, rotation: Literal['X', 'Y', 'Z'] = 'Y', 
                 use_hadamard: bool = True):
        super().__init__(num_qubits, num_features, num_parameters, rotation, use_hadamard)

    def ansatz_layer(self, x: ParameterVector) -> QuantumCircuit:  
        """ Definition of the ansatz. """

        qc = QuantumCircuit(self.num_qubits)

        param_idx = 0

        for layer in range(self.num_layers):
            
            for i in range(self.num_qubits):
                qc.ry(x[i], i)

            # Trainable rotations.
            for i in range(self.num_qubits):
                qc.ry(self.weight_params[param_idx], i)
                param_idx += 1
                qc.rz(self.weight_params[param_idx], i)
                param_idx += 1

            # Linear entanglement.
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)

            # 4) Ring closure (optional).
            if self.num_qubits > 2:
                qc.cx(self.num_qubits - 1, 0)

        return qc
