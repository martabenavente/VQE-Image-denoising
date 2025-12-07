from qiskit import QuantumCircuit

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

    def ansatz_layer(self) -> QuantumCircuit:  
        ## Initial simple ansatz with less parameters, when the complete pipeline works we can update it (add layers, alternate rotations...).
        """ Definition of a simple ansatz with a trainable rotations and entanglement. """

        qc = QuantumCircuit(self.num_qubits)

        ## Layer of trainable RX rotations with trainable params.
        for i in range(min(self.num_parameters, self.num_qubits)):
            qc.rx(self.weight_params[i], i)

        ## Linear entanglement with CNOTs.
        if self.num_qubits > 1:
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)

        return qc
