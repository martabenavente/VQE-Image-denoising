from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import Optional

from src.circuits.circuit_base import CircuitBase


class QNNBuilder:
    """
    Builder class for creating EstimatorQNN with flexible configuration.
    Separates observable creation from QNN instantiation for clarity.

    Args:
        circuit_base (CircuitBase): Instance of CircuitBase (e.g., SimpleAnsatzCircuit).
        observable (SparsePauliOp): Quantum observable.
            If None, defaults to sum of Pauli-Z on all qubits.
        estimator (Estimator): Qiskit Estimator primitive.
            If None, uses default Estimator.
    """

    def __init__(
            self,
            circuit_base: CircuitBase,
            observable: Optional[SparsePauliOp] = None,
            estimator: Optional[Estimator] = None
    ):
        self.circuit_base = circuit_base
        self.circuit = circuit_base.build_circuit()

        self.observable: SparsePauliOp
        if observable is None:
            self.observable = SparsePauliOp.from_list(
                [( 'Z' * circuit_base.num_qubits, 1.0 )]
            )
        else:
            self.observable = observable

        if estimator is None:
            self.estimator = Estimator()
        else:
            self.estimator = estimator

        self.qnn: Optional[EstimatorQNN] = None

    def build_qnn(
            self,
            input_gradients: bool = True,
            **estimator_qnn_kwargs
    ) -> EstimatorQNN:
        """
        Build the EstimatorQNN with current configuration.

        Args:
            input_gradients (bool): Whether to compute gradients w.r.t. input features.
                Default True (for TorchConnector compatibility).
            **estimator_qnn_kwargs: Additional keyword arguments for EstimatorQNN.

        Returns:
            EstimatorQNN: Configured quantum neural network.
        """
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            estimator=self.estimator,
            input_params=self.circuit_base.feature_params,
            weight_params=self.circuit_base.weight_params,
            input_gradients=input_gradients,
            **estimator_qnn_kwargs
        )
        return self.qnn

    def get_qnn(self) -> EstimatorQNN:
        """
        Return the built QNN.

        Returns:
            EstimatorQNN: The quantum neural network.

        Raises:
            ValueError: If QNN hasn't been built yet.
        """
        if self.qnn is None:
            raise ValueError("QNN not built yet. Call build_qnn() first.")
        return self.qnn

    def set_observable(self, observable: SparsePauliOp):
        """
        Update the observable (requires rebuilding QNN).

        Args:
            observable (SparsePauliOp): New observable.
        """
        self.observable = observable
        self.qnn = None  # Reset QNN to require rebuild

    def set_estimator(self, estimator: Estimator):
        """
        Update the estimator primitive (requires rebuilding QNN).

        Args:
            estimator (Estimator): New estimator.
        """
        self.estimator = estimator
        self.qnn = None  # Reset QNN to require rebuild