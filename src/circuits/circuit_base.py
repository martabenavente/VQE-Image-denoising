from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from abc import ABC, abstractmethod
from typing import Optional


class CircuitBase(ABC):
    """
    Base abstract class for parametrized quantum circuits for ML.
    Supports separate encoding and ansatz layers with parameters.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_features (int): Number of input features to encode.
        num_parameters (int): Number of trainable parameters in the ansatz.
    """

    def __init__(self, num_qubits: int, num_features: int, num_parameters: int):
        assert num_qubits > 0, "Number of qubits must be positive"
        assert num_features > 0, "Number of features must be positive"
        assert num_parameters >= 0, "Number of parameters must be non-negative"

        self.num_qubits = num_qubits
        self.num_features = num_features
        self.num_parameters = num_parameters

        # Create parameter vectors
        self.feature_params = ParameterVector('x', num_features)
        self.weight_params = ParameterVector('θ', num_parameters)

        self.circuit: Optional[QuantumCircuit] = None

    @abstractmethod
    def encoding_layer(self) -> QuantumCircuit:
        """
        Create the encoding layer using feature_params.

        Returns:
            QuantumCircuit: Circuit for data encoding.
        """
        pass

    @abstractmethod
    def ansatz_layer(self) -> QuantumCircuit:
        """
        Create the variational ansatz using weight_params.

        Returns:
            QuantumCircuit: Circuit for the ansatz.
        """
        pass

    def build_circuit(self) -> QuantumCircuit:
        """
        Combine encoding and ansatz into a single parametrized circuit.

        Returns:
            QuantumCircuit: The complete parametrized quantum circuit.
        """
        encoding = self.encoding_layer()
        ansatz = self.ansatz_layer()

        assert encoding.num_qubits == self.num_qubits, \
            f"Encoding layer has {encoding.num_qubits} qubits, expected {self.num_qubits}"
        assert ansatz.num_qubits == self.num_qubits, \
            f"Ansatz layer has {ansatz.num_qubits} qubits, expected {self.num_qubits}"

        self.circuit = encoding.compose(ansatz)
        return self.circuit

    def get_circuit(self) -> QuantumCircuit:
        """
        Return the parametrized quantum circuit.

        Returns:
            QuantumCircuit: The complete parametrized circuit.
        """
        if self.circuit is None:
            raise ValueError("Circuit not built yet. Call build_circuit() first.")
        return self.circuit

    def get_num_parameters(self) -> int:
        """
        Return total number of parameters (features + weights).

        Returns:
            int: Total number of parameters.
        """
        return self.num_features + self.num_parameters