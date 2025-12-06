import pytest
import numpy as np

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from src.circuits.circuit_base import CircuitBase


class MockCircuit(CircuitBase):
    """Mock implementation for testing CircuitBase."""

    def encoding_layer(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for i in range(min(self.num_features, self.num_qubits)):
            qc.h(i)
            qc.ry(self.feature_params[i], i)
        return qc

    def ansatz_layer(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for i in range(min(self.num_parameters, self.num_qubits)):
            qc.rx(self.weight_params[i], i)
        if self.num_qubits > 1:
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        return qc


class TestCircuitBase:

    def test_initialization_valid(self):
        """Test valid initialization."""
        circuit = MockCircuit(num_qubits=4, num_features=3, num_parameters=5)
        assert circuit.num_qubits == 4
        assert circuit.num_features == 3
        assert circuit.num_parameters == 5
        assert len(circuit.feature_params) == 3
        assert len(circuit.weight_params) == 5

    def test_initialization_invalid_qubits(self):
        """Test that initialization fails with invalid qubit count."""
        with pytest.raises(AssertionError, match="Number of qubits must be positive"):
            MockCircuit(num_qubits=0, num_features=2, num_parameters=3)

        with pytest.raises(AssertionError, match="Number of qubits must be positive"):
            MockCircuit(num_qubits=-1, num_features=2, num_parameters=3)

    def test_initialization_invalid_features(self):
        """Test that initialization fails with invalid feature count."""
        with pytest.raises(AssertionError, match="Number of features must be positive"):
            MockCircuit(num_qubits=4, num_features=0, num_parameters=3)

        with pytest.raises(AssertionError, match="Number of features must be positive"):
            MockCircuit(num_qubits=4, num_features=-1, num_parameters=3)

    def test_initialization_invalid_parameters(self):
        """Test that initialization fails with invalid parameter count."""
        with pytest.raises(AssertionError, match="Number of parameters must be non-negative"):
            MockCircuit(num_qubits=4, num_features=2, num_parameters=-1)

    def test_parameter_vectors_created(self):
        """Test that parameter vectors are correctly created."""
        circuit = MockCircuit(num_qubits=4, num_features=3, num_parameters=5)
        assert isinstance(circuit.feature_params, ParameterVector)
        assert isinstance(circuit.weight_params, ParameterVector)
        assert circuit.feature_params.name == 'x'
        assert circuit.weight_params.name == 'θ'

    def test_build_circuit(self):
        """Test building the complete circuit."""
        circuit = MockCircuit(num_qubits=4, num_features=3, num_parameters=5)
        qc = circuit.build_circuit()

        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4
        assert circuit.circuit is not None

    def test_get_circuit_before_build(self):
        """Test that get_circuit raises error before building."""
        circuit = MockCircuit(num_qubits=4, num_features=3, num_parameters=5)

        with pytest.raises(ValueError, match="Circuit not built yet"):
            circuit.get_circuit()

    def test_get_circuit_after_build(self):
        """Test get_circuit after building."""
        circuit = MockCircuit(num_qubits=4, num_features=3, num_parameters=5)
        circuit.build_circuit()
        qc = circuit.get_circuit()

        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4

    def test_get_num_parameters(self):
        """Test total parameter count."""
        circuit = MockCircuit(num_qubits=4, num_features=3, num_parameters=5)
        assert circuit.get_num_parameters() == 8

    def test_circuit_parameters_bound(self):
        """Test that circuit parameters can be bound."""
        circuit = MockCircuit(num_qubits=4, num_features=3, num_parameters=4)
        qc = circuit.build_circuit()

        # Create parameter binding dictionary
        feature_values = np.random.uniform(0, 2*np.pi, 3)
        weight_values = np.random.uniform(0, 2*np.pi, 4)

        param_dict = {
            **{circuit.feature_params[i]: feature_values[i] for i in range(3)},
            **{circuit.weight_params[i]: weight_values[i] for i in range(4)}
        }

        bound_qc = qc.assign_parameters(param_dict)
        assert len(bound_qc.parameters) == 0

    def test_draw_circuit_with_8_qubits(self):
        """Test drawing a circuit with 8 qubits and random parameters."""
        np.random.seed(42)
        circuit = MockCircuit(num_qubits=8, num_features=8, num_parameters=8)
        qc = circuit.build_circuit()

        # Verify circuit structure
        assert qc.num_qubits == 8
        assert len(qc.parameters) == 16  # 8 features + 8 weights

        # Create parameter binding with random values
        feature_values = np.random.uniform(0, 2*np.pi, 8)
        weight_values = np.random.uniform(0, 2*np.pi, 8)

        param_dict = {
            **{circuit.feature_params[i]: feature_values[i] for i in range(8)},
            **{circuit.weight_params[i]: weight_values[i] for i in range(8)}
        }

        bound_qc = qc.assign_parameters(param_dict)

        # Draw the circuit (this will print to console in test output)
        print("\n" + "="*80)
        print("8-Qubit Circuit Visualization with Random Parameters:")
        print("="*80)
        print(bound_qc.draw(output='text'))
        print("="*80)

        # Verify circuit is properly bound
        assert len(bound_qc.parameters) == 0
        assert bound_qc.depth() > 0