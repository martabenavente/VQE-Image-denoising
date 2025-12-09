# import os
# import sys

# CURRENT_DIR = os.path.dirname(__file__)
# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np

from qiskit import QuantumCircuit
from src.circuits.encoding_circuit import AngleEmbeddingCircuit
from src.circuits.ansatz_circuit import SimpleAnsatzCircuit


class MockAngleEmbeddingCircuit(AngleEmbeddingCircuit):
    """Mock implementation for testing AngleEmbeddingCircuit."""

    def ansatz_layer(self) -> QuantumCircuit:
        """Simple ansatz with rotation gates and entanglement."""
        qc = QuantumCircuit(self.num_qubits)
        for i in range(min(self.num_parameters, self.num_qubits)):
            qc.rx(self.weight_params[i], i)
        if self.num_qubits > 1:
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        return qc


class TestAngleEmbeddingCircuit:
    """Test suite for AngleEmbeddingCircuit."""

    def test_initialization_valid(self):
        """Test valid initialization with default parameters."""
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=4,
            num_features=3,
            num_parameters=5
        )
        assert circuit.num_qubits == 4
        assert circuit.num_features == 3
        assert circuit.num_parameters == 5
        assert circuit.rotation == 'Y'
        assert circuit.use_hadamard is True

    def test_initialization_custom_rotation(self):
        """Test initialization with custom rotation gate."""
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=4,
            num_features=4,
            num_parameters=4,
            rotation='X',
            use_hadamard=False
        )
        assert circuit.rotation == 'X'
        assert circuit.use_hadamard is False

    def test_initialization_all_rotations(self):
        """Test initialization with all supported rotation types."""
        for rot in ['X', 'Y', 'Z', 'x', 'y', 'z']:
            circuit = MockAngleEmbeddingCircuit(
                num_qubits=4,
                num_features=4,
                num_parameters=4,
                rotation=rot
            )
            assert circuit.rotation in ['X', 'Y', 'Z']

    def test_initialization_invalid_rotation(self):
        """Test that invalid rotation gate raises error."""
        with pytest.raises(AssertionError, match="Invalid rotation gate"):
            MockAngleEmbeddingCircuit(
                num_qubits=4,
                num_features=4,
                num_parameters=4,
                rotation='W'
            )

    def test_initialization_features_exceed_qubits(self):
        """Test that features exceeding qubits raises error."""
        with pytest.raises(AssertionError, match="Number of features .* cannot exceed number of qubits"):
            MockAngleEmbeddingCircuit(
                num_qubits=4,
                num_features=5,
                num_parameters=4
            )

    def test_encoding_layer_structure(self):
        """Test encoding layer structure and gates."""
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=4,
            num_features=3,
            num_parameters=4,
            rotation='Y',
            use_hadamard=True
        )
        encoding = circuit.encoding_layer()

        assert isinstance(encoding, QuantumCircuit)
        assert encoding.num_qubits == 4

        gate_names = [instr.operation.name for instr in encoding.data]
        assert 'h' in gate_names
        assert 'ry' in gate_names

    def test_encoding_layer_without_hadamard(self):
        """Test encoding layer without Hadamard gates."""
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=4,
            num_features=4,
            num_parameters=4,
            use_hadamard=False
        )
        encoding = circuit.encoding_layer()

        gate_names = [instr.operation.name for instr in encoding.data]
        assert 'h' not in gate_names
        assert 'ry' in gate_names

    def test_encoding_layer_rx_gates(self):
        """Test encoding layer with RX rotation gates."""
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=4,
            num_features=4,
            num_parameters=4,
            rotation='X'
        )
        encoding = circuit.encoding_layer()

        gate_names = [instr.operation.name for instr in encoding.data]
        assert 'rx' in gate_names

    def test_encoding_layer_ry_gates(self):
        """Test encoding layer with RY rotation gates."""
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=4,
            num_features=4,
            num_parameters=4,
            rotation='Y'
        )
        encoding = circuit.encoding_layer()

        gate_names = [instr.operation.name for instr in encoding.data]
        assert 'ry' in gate_names

    def test_encoding_layer_rz_gates(self):
        """Test encoding layer with RZ rotation gates."""
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=4,
            num_features=4,
            num_parameters=4,
            rotation='Z'
        )
        encoding = circuit.encoding_layer()

        gate_names = [instr.operation.name for instr in encoding.data]
        assert 'rz' in gate_names

    def test_encoding_layer_parameters(self):
        """Test that encoding layer uses correct parameters."""
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=4,
            num_features=3,
            num_parameters=4
        )
        encoding = circuit.encoding_layer()

        params = encoding.parameters
        assert len(params) == 3
        for param in params:
            assert param in circuit.feature_params

    def test_build_complete_circuit(self):
        """Test building complete circuit with encoding and ansatz."""
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=4,
            num_features=3,
            num_parameters=4
        )
        qc = circuit.build_circuit()

        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4
        assert len(qc.parameters) == 7  # 3 features + 4 weights

    def test_parameter_binding(self):
        """Test binding parameters with values."""
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=4,
            num_features=4,
            num_parameters=4
        )
        qc = circuit.build_circuit()

        feature_values = np.random.uniform(0, 2*np.pi, 4)
        weight_values = np.random.uniform(0, 2*np.pi, 4)

        param_dict = {
            **{circuit.feature_params[i]: feature_values[i] for i in range(4)},
            **{circuit.weight_params[i]: weight_values[i] for i in range(4)}
        }

        bound_qc = qc.assign_parameters(param_dict)
        assert len(bound_qc.parameters) == 0

    def test_features_less_than_qubits(self):
        """Test encoding with fewer features than qubits."""
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=8,
            num_features=4,
            num_parameters=8
        )
        encoding = circuit.encoding_layer()

        # Should only apply rotations to first 4 qubits
        rotation_gates = [instr for instr in encoding.data if instr.operation.name in ['rx', 'ry', 'rz']]
        assert len(rotation_gates) == 4

    def test_draw_circuit_8_qubits_y_rotation(self):
        """Test drawing 8-qubit circuit with Y rotation and random parameters."""
        np.random.seed(42)
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=8,
            num_features=8,
            num_parameters=8,
            rotation='Y',
            use_hadamard=True
        )
        qc = circuit.build_circuit()

        assert qc.num_qubits == 8
        assert len(qc.parameters) == 16

        feature_values = np.random.uniform(0, 2*np.pi, 8)
        weight_values = np.random.uniform(0, 2*np.pi, 8)

        param_dict = {
            **{circuit.feature_params[i]: feature_values[i] for i in range(8)},
            **{circuit.weight_params[i]: weight_values[i] for i in range(8)}
        }

        bound_qc = qc.assign_parameters(param_dict)

        print("\n" + "="*80)
        print("8-Qubit AngleEmbedding Circuit (Y Rotation) with Random Parameters:")
        print("="*80)
        print(bound_qc.draw(output='text'))
        print("="*80)

        assert len(bound_qc.parameters) == 0

    def test_draw_circuit_8_qubits_x_rotation(self):
        """Test drawing 8-qubit circuit with X rotation."""
        np.random.seed(123)
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=8,
            num_features=8,
            num_parameters=8,
            rotation='X',
            use_hadamard=True
        )
        qc = circuit.build_circuit()

        feature_values = np.random.uniform(0, np.pi, 8)
        weight_values = np.random.uniform(0, np.pi, 8)

        param_dict = {
            **{circuit.feature_params[i]: feature_values[i] for i in range(8)},
            **{circuit.weight_params[i]: weight_values[i] for i in range(8)}
        }

        bound_qc = qc.assign_parameters(param_dict)

        print("\n" + "="*80)
        print("8-Qubit AngleEmbedding Circuit (X Rotation) with Random Parameters:")
        print("="*80)
        print(bound_qc.draw(output='text'))
        print("="*80)

        gate_names = [instr.operation.name for instr in bound_qc.data]
        assert 'rx' in gate_names

    def test_draw_circuit_8_qubits_no_hadamard(self):
        """Test drawing 8-qubit circuit without Hadamard gates."""
        np.random.seed(456)
        circuit = MockAngleEmbeddingCircuit(
            num_qubits=8,
            num_features=8,
            num_parameters=8,
            rotation='Z',
            use_hadamard=False
        )
        qc = circuit.build_circuit()

        feature_values = np.random.uniform(0, 2*np.pi, 8)
        weight_values = np.random.uniform(0, 2*np.pi, 8)

        param_dict = {
            **{circuit.feature_params[i]: feature_values[i] for i in range(8)},
            **{circuit.weight_params[i]: weight_values[i] for i in range(8)}
        }

        bound_qc = qc.assign_parameters(param_dict)

        print("\n" + "="*80)
        print("8-Qubit AngleEmbedding Circuit (Z Rotation, No Hadamard):")
        print("="*80)
        print(bound_qc.draw(output='text'))
        print("="*80)

        gate_names = [instr.operation.name for instr in bound_qc.data]
        assert 'h' not in gate_names
        assert 'rz' in gate_names


def test_simple_ansatz_builds_full_circuit():
    """ Test the construction of a complete circuit (encoding + ansatz) and the num. of parameters = num_features + num_parameters. """
    circuit = SimpleAnsatzCircuit(
        num_qubits=4,
        num_features=4,
        num_parameters=4,
        rotation='Y',
        use_hadamard=True,
    )

    qc = circuit.build_circuit()

    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 4

    assert len(qc.parameters) == circuit.get_num_parameters()
    assert circuit.get_num_parameters() == circuit.num_features + circuit.num_parameters


def test_simple_ansatz_uses_parameter_vectors_only():
    """ Test the parameters of the circuit (no numerical values) """
    circuit = SimpleAnsatzCircuit(
        num_qubits=3,
        num_features=3,
        num_parameters=3,
        rotation='Z',
        use_hadamard=False,
    )

    qc = circuit.build_circuit()

    for p in qc.parameters:
        assert hasattr(p, "name")


def test_simple_ansatz_can_bind_parameters():
    """ Test numerical values can be asigned to all parameters. """
    num_qubits = 4
    num_features = 4
    num_parameters = 4

    circuit = SimpleAnsatzCircuit(
        num_qubits=num_qubits,
        num_features=num_features,
        num_parameters=num_parameters,
        rotation='Y',
        use_hadamard=True,
    )

    qc = circuit.build_circuit()

    feature_values = np.linspace(0, np.pi, num_features)
    weight_values = np.linspace(0, np.pi/2, num_parameters)

    param_dict = {
        **{circuit.feature_params[i]: float(feature_values[i]) for i in range(num_features)},
        **{circuit.weight_params[i]: float(weight_values[i]) for i in range(num_parameters)},
    }

    bound_qc = qc.assign_parameters(param_dict)

    assert len(bound_qc.parameters) == 0
    assert bound_qc.depth() > 0
