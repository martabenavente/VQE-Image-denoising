import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

from src.circuits.circuit_base import CircuitBase
from src.qnn.qnn_builder import QNNBuilder


class MockCircuitBase(CircuitBase):
    """Mock implementation of CircuitBase for testing."""

    def __init__(self, num_qubits: int, num_features: int, num_parameters: int):
        super().__init__(num_qubits, num_features, num_parameters)

    def encoding_layer(self) -> QuantumCircuit:
        """Simple encoding with RY rotations."""
        qc = QuantumCircuit(self.num_qubits)
        for i in range(min(self.num_features, self.num_qubits)):
            qc.ry(self.feature_params[i], i)
        return qc

    def ansatz_layer(self) -> QuantumCircuit:
        """Simple ansatz with RX rotations and entanglement."""
        qc = QuantumCircuit(self.num_qubits)
        for i in range(min(self.num_parameters, self.num_qubits)):
            qc.rx(self.weight_params[i], i)
        if self.num_qubits > 1:
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        return qc


class TestQNNBuilderInitialization:
    """Tests for QNNBuilder initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default observable and estimator."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)

        assert builder.circuit_base is circuit
        assert builder.circuit is not None
        assert isinstance(builder.observable, SparsePauliOp)
        assert isinstance(builder.estimator, Estimator)
        assert builder.qnn is None

    def test_default_observable_is_pauli_z_tensor(self):
        """Test that default observable is Z⊗Z⊗...⊗Z."""
        circuit = MockCircuitBase(num_qubits=3, num_features=3, num_parameters=3)
        builder = QNNBuilder(circuit)

        # Check observable structure
        expected_pauli = 'ZZZ'
        assert str(builder.observable.paulis[0]) == expected_pauli
        assert builder.observable.coeffs[0] == 1.0

    def test_initialization_with_custom_observable(self):
        """Test initialization with custom observable."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        custom_observable = SparsePauliOp.from_list([('ZIII', 1.0)])

        builder = QNNBuilder(circuit, observable=custom_observable)

        assert builder.observable == custom_observable
        assert str(builder.observable.paulis[0]) == 'ZIII'

    def test_initialization_with_custom_estimator(self):
        """Test initialization with custom estimator."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        custom_estimator = Estimator()

        builder = QNNBuilder(circuit, estimator=custom_estimator)

        assert builder.estimator is custom_estimator

    def test_circuit_is_built_on_initialization(self):
        """Test that circuit is built during initialization."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)

        assert builder.circuit is not None
        assert builder.circuit.num_qubits == 4
        assert len(builder.circuit.parameters) == 8  # 4 features + 4 weights


class TestQNNBuilderBuildQNN:
    """Tests for build_qnn method."""

    def test_build_qnn_creates_estimator_qnn(self):
        """Test that build_qnn creates an EstimatorQNN instance."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)

        qnn = builder.build_qnn()

        assert isinstance(qnn, EstimatorQNN)
        assert builder.qnn is qnn

    def test_build_qnn_with_input_gradients_true(self):
        """Test building QNN with input_gradients=True."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)

        qnn = builder.build_qnn(input_gradients=True)

        assert qnn.input_gradients is True

    def test_build_qnn_with_input_gradients_false(self):
        """Test building QNN with input_gradients=False."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)

        qnn = builder.build_qnn(input_gradients=False)

        assert qnn.input_gradients is False

    def test_build_qnn_default_input_gradients(self):
        """Test that default input_gradients is True."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)

        qnn = builder.build_qnn()

        assert qnn.input_gradients is True

    def test_qnn_has_correct_number_of_inputs(self):
        """Test that QNN has correct number of inputs."""
        num_features = 5
        circuit = MockCircuitBase(num_qubits=4, num_features=num_features, num_parameters=4)
        builder = QNNBuilder(circuit)

        qnn = builder.build_qnn()

        assert qnn.num_inputs == num_features

    def test_qnn_has_correct_number_of_weights(self):
        """Test that QNN has correct number of weights."""
        num_parameters = 6
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=num_parameters)
        builder = QNNBuilder(circuit)

        qnn = builder.build_qnn()

        assert qnn.num_weights == num_parameters

    def test_qnn_forward_pass(self):
        """Test that QNN can perform forward pass."""
        circuit = MockCircuitBase(num_qubits=2, num_features=2, num_parameters=2)
        builder = QNNBuilder(circuit)
        qnn = builder.build_qnn()

        features = np.array([0.5, 1.0])
        weights = np.array([0.3, 0.7])

        result = qnn.forward(features, weights)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)  # Single output

    def test_qnn_backward_pass_with_gradients(self):
        """Test that QNN can compute gradients."""
        circuit = MockCircuitBase(num_qubits=2, num_features=2, num_parameters=2)
        builder = QNNBuilder(circuit)
        qnn = builder.build_qnn(input_gradients=True)

        features = np.array([0.5, 1.0])
        weights = np.array([0.3, 0.7])

        input_grad, weight_grad = qnn.backward(features, weights)

        assert input_grad is not None
        assert weight_grad is not None
        assert input_grad.shape == (1, 1, 2)  # Gradients w.r.t. inputs
        assert weight_grad.shape == (1, 1, 2)  # Gradients w.r.t. weights


class TestQNNBuilderGetQNN:
    """Tests for get_qnn method."""

    def test_get_qnn_returns_built_qnn(self):
        """Test that get_qnn returns the built QNN."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)
        qnn = builder.build_qnn()

        retrieved_qnn = builder.get_qnn()

        assert retrieved_qnn is qnn

    def test_get_qnn_raises_error_if_not_built(self):
        """Test that get_qnn raises ValueError if QNN not built."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)

        with pytest.raises(ValueError, match="QNN not built yet"):
            builder.get_qnn()


class TestQNNBuilderSetObservable:
    """Tests for set_observable method."""

    def test_set_observable_updates_observable(self):
        """Test that set_observable updates the observable."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)

        new_observable = SparsePauliOp.from_list([('XXXX', 0.5)])
        builder.set_observable(new_observable)

        assert builder.observable == new_observable

    def test_set_observable_resets_qnn(self):
        """Test that set_observable resets QNN to None."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)
        builder.build_qnn()

        assert builder.qnn is not None

        new_observable = SparsePauliOp.from_list([('ZZZZ', 1.0)])
        builder.set_observable(new_observable)

        assert builder.qnn is None

    def test_rebuild_qnn_after_set_observable(self):
        """Test that QNN can be rebuilt after changing observable."""
        circuit = MockCircuitBase(num_qubits=2, num_features=2, num_parameters=2)
        builder = QNNBuilder(circuit)

        qnn1 = builder.build_qnn()
        result1 = qnn1.forward(np.array([0.5, 1.0]), np.array([0.3, 0.7]))

        # Change observable and rebuild
        new_observable = SparsePauliOp.from_list([('ZI', 1.0)])
        builder.set_observable(new_observable)
        qnn2 = builder.build_qnn()
        result2 = qnn2.forward(np.array([0.5, 1.0]), np.array([0.3, 0.7]))

        assert qnn1 is not qnn2
        assert not np.allclose(result1, result2)


class TestQNNBuilderSetEstimator:
    """Tests for set_estimator method."""

    def test_set_estimator_updates_estimator(self):
        """Test that set_estimator updates the estimator."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)

        new_estimator = Estimator()
        builder.set_estimator(new_estimator)

        assert builder.estimator is new_estimator

    def test_set_estimator_resets_qnn(self):
        """Test that set_estimator resets QNN to None."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)
        builder.build_qnn()

        assert builder.qnn is not None

        new_estimator = Estimator()
        builder.set_estimator(new_estimator)

        assert builder.qnn is None

    def test_rebuild_qnn_after_set_estimator(self):
        """Test that QNN can be rebuilt after changing estimator."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        builder = QNNBuilder(circuit)

        qnn1 = builder.build_qnn()

        new_estimator = Estimator()
        builder.set_estimator(new_estimator)
        qnn2 = builder.build_qnn()

        assert qnn1 is not qnn2
        assert isinstance(qnn2, EstimatorQNN)


class TestQNNBuilderEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_qubit_circuit(self):
        """Test with single qubit circuit."""
        circuit = MockCircuitBase(num_qubits=1, num_features=1, num_parameters=1)
        builder = QNNBuilder(circuit)
        qnn = builder.build_qnn()

        assert qnn.num_inputs == 1
        assert qnn.num_weights == 1

    def test_large_number_of_qubits(self):
        """Test with larger number of qubits."""
        circuit = MockCircuitBase(num_qubits=10, num_features=10, num_parameters=10)
        builder = QNNBuilder(circuit)
        qnn = builder.build_qnn()

        assert qnn.num_inputs == 10
        assert qnn.num_weights == 10

    def test_zero_parameters(self):
        """Test with zero trainable parameters."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=0)
        builder = QNNBuilder(circuit)
        qnn = builder.build_qnn()

        assert qnn.num_weights == 0

    def test_multi_term_observable(self):
        """Test with multi-term observable."""
        circuit = MockCircuitBase(num_qubits=4, num_features=4, num_parameters=4)
        observable = SparsePauliOp.from_list([
            ('ZZZZ', 0.5),
            ('XXXX', 0.3),
            ('YYYY', 0.2)
        ])
        builder = QNNBuilder(circuit, observable=observable)
        qnn = builder.build_qnn()

        assert isinstance(qnn, EstimatorQNN)

        features = np.random.rand(4)
        weights = np.random.rand(4)
        result = qnn.forward(features, weights)
        assert result is not None


class TestQNNBuilderIntegration:
    """Integration tests for full workflow."""

    def test_complete_workflow(self):
        """Test complete workflow: initialize, build, use, modify, rebuild."""
        circuit = MockCircuitBase(num_qubits=3, num_features=3, num_parameters=3)
        builder = QNNBuilder(circuit)

        qnn1 = builder.build_qnn()
        features = np.array([0.1, 0.2, 0.3])
        weights = np.array([0.4, 0.5, 0.6])
        result1 = qnn1.forward(features, weights)

        new_observable = SparsePauliOp.from_list([('ZII', 1.0)])
        builder.set_observable(new_observable)

        qnn2 = builder.build_qnn()
        result2 = qnn2.forward(features, weights)

        assert result1 is not None
        assert result2 is not None
        assert not np.allclose(result1, result2)

    def test_multiple_forward_passes(self):
        """Test multiple forward passes with different inputs."""
        circuit = MockCircuitBase(num_qubits=2, num_features=2, num_parameters=2)
        builder = QNNBuilder(circuit)
        qnn = builder.build_qnn()

        weights = np.array([0.3, 0.7])

        results = []
        for i in range(5):
            features = np.random.rand(2)
            result = qnn.forward(features, weights)
            results.append(result)

        for result in results:
            assert result is not None
            assert isinstance(result, np.ndarray)