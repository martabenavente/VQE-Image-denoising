from abc import abstractmethod
from typing import Literal
from qiskit import QuantumCircuit
from src.circuits.circuit_base import CircuitBase


class AngleEmbeddingCircuit(CircuitBase):
    """
    Angle embedding circuit for PQCs.
    Encodes features using rotation gates with Hadamard initialization.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_features (int): Number of input features to encode.
        num_parameters (int): Number of trainable parameters in the ansatz.
        rotation (str): Rotation gate type ('X', 'Y', or 'Z'). Default 'Y'.
        use_hadamard (bool): Apply Hadamard gates before encoding. Default True.
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        num_parameters: int,
        rotation: Literal['X', 'Y', 'Z'] = 'Y',
        use_hadamard: bool = True
    ):
        assert num_features <= num_qubits, \
            f"Number of features ({num_features}) cannot exceed number of qubits ({num_qubits})"

        super().__init__(num_qubits, num_features, num_parameters)
        self.rotation = rotation.upper()
        self.use_hadamard = use_hadamard

        assert self.rotation in ['X', 'Y', 'Z'], \
            f"Invalid rotation gate: {rotation}. Must be 'X', 'Y', or 'Z'"

    def encoding_layer(self) -> QuantumCircuit:
        """
        Create angle embedding encoding layer with parametrized rotations.

        Returns:
            QuantumCircuit: Encoding circuit with feature parameters.
        """
        qc = QuantumCircuit(self.num_qubits)

        # Apply Hadamard gates for superposition
        if self.use_hadamard:
            for i in range(self.num_qubits):
                qc.h(i)

        # Apply parametrized rotation gates for features
        for i in range(self.num_features):
            if self.rotation == 'X':
                qc.rx(self.feature_params[i], i)
            elif self.rotation == 'Y':
                qc.ry(self.feature_params[i], i)
            elif self.rotation == 'Z':
                qc.rz(self.feature_params[i], i)

        return qc

    @abstractmethod
    def ansatz_layer(self) -> QuantumCircuit:
        """
        Create the variational ansatz using weight_params.
        Must be implemented by subclasses.

        Returns:
            QuantumCircuit: Circuit for the ansatz.
        """
        pass