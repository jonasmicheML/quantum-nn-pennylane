import QCircNet.circuits.QuantumCircuit as qcn
import pennylane as qml
import numpy as np
import inspect
import sys

def get_custom_circuits():
    """ Get the custom circuits for this file dynamically. """
    current_module = sys.modules[__name__]
    # return all classes defined in this module
    return [obj for _, obj in inspect.getmembers(current_module) 
            if inspect.isclass(obj) and obj.__module__ == __name__]


class DoubleEntanglementVLCircuit(qcn.QuantumCircuitNetwork):
    """
    Our novel approch.
    Based on senokosov2024 feature map, but custom variational layer.
    """
    def __init__(self, reps=2, **kwargs):
        self.reps = reps
        super().__init__(**kwargs)

    def _init_weight_shapes(self):
        """ Initialize the shapes of the weights for the circuit. Has to be changed if _variational_layer() is changed """
        # 3 parameters per qubit (RX, RY, RZ)
        self.weight_shapes = {"weights": (self.n_qubits, 3*self.reps)}  # 2 layers

    def circuit(self, inputs, weights):
        self._feature_map(inputs)
        for rep in range(self.reps):
            self._variational_layer(weights, layer_idx=rep)
            self._entangle_qubits(circular_connection=True)
        return self._measure() 


class Senokosov2024Circuit(qcn.QuantumCircuitNetwork):
    """
    The quantum circuit from the Senokosov et al. (2024) paper.
    https://iopscience.iop.org/article/10.1088/2632-2153/ad2aef/pdf
    """
    def __init__(self, reps=3, **kwargs):
        self.reps = reps
        super().__init__(**kwargs)

    def _init_weight_shapes(self):
        # 3 parameters per qubit (RY, RZ, RY)
        self.weight_shapes = {"weights": (self.n_qubits, 3*self.reps)} # reps layers

    def _variational_layer(self, weights, layer_idx=0):
        for i in range(self.n_qubits):
            qml.RY(weights[i, 0+layer_idx], wires=i)
            qml.RZ(weights[i, 1+layer_idx], wires=i)
            qml.RY(weights[i, 2+layer_idx], wires=i)

    def _measure(self):
        return [qml.expval(qml.PauliY(i)) for i in range(self.n_qubits)]

    def circuit(self, inputs, weights):
        self._feature_map(inputs)
        for rep in range(self.reps): 
            self._variational_layer(weights, layer_idx=rep)
            self._entangle_qubits(circular_connection=True)
        return self._measure()


class Anusha2024Circuit(qcn.QuantumCircuitNetwork):
    """
    The quantum circuit from the Anusha et al. (2024) paper.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10714726
    """
    def _init_weight_shapes(self):
        # 1 parameters per qubit (RY)
        self.weight_shapes = {"weights": (self.n_qubits, 1)} 

    def _feature_map(self, inputs):
        for i in range(self.n_qubits):
            base_idx = i * self.features_per_qubit
            for j in range(self.features_per_qubit):
                qml.RY(inputs[base_idx + 0], wires=i)

    def _variational_layer(self, weights):
        for i in range(self.n_qubits):
            qml.RY(weights[i, 0], wires=i)

    def circuit(self, inputs, weights):
        self._feature_map(inputs)
        self._variational_layer(weights)
        self._entangle_qubits(circular_connection=True)
        return self._measure()
    

class Ranga2024Circuit(qcn.QuantumCircuitNetwork):
    """
    The quantum circuit from the Ranga et al. (2024) paper.
    https://uia.brage.unit.no/uia-xmlui/handle/11250/3183181
    """
    def __init__(self, reps=2, **kwargs):
        self.reps = reps
        super().__init__(**kwargs)
  
    def _feature_map(self, inputs):
        """
        Implements a ZZ feature map similar to Qiskit's ZZFeatureMap; but with CNOT, RZ, CNOT
        """
        
        # First Hadamard layer
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
        
        # Repeated blocks
        for feat in range(self.features_per_qubit):
            offset = feat * self.n_qubits # assure correct indexing
            # single-qubit rotations
            for i in range(self.n_qubits):
                qml.RZ(2 * inputs[i+offset], wires=i)
            
            # two-qubit ZZ interactions
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    # ZZ entangling operation; CNOT, RZ, CNOT which is equivalent to ZZ rotation
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * inputs[i+offset] * inputs[j+offset], wires=j)
                    qml.CNOT(wires=[i, j])

    def _init_weight_shapes(self):
        # 3 parameters per qubit (RY)
        self.weight_shapes = {"weights": (self.n_qubits, (1+self.reps))} 
        
    def _variational_layer(self, weights, layer_idx=0):
        """ 
        Implement the RealAmplitudes like shown here:
        https://www.researchgate.net/figure/The-RealAmplitude-circuit-Ansatz-illustrated-for-the-case-of-q-4-qubits-and-L-2_fig1_368507678
        """
        offset = layer_idx
        if layer_idx == 0:
            for i in range(self.n_qubits):
                qml.RY(weights[i, 0], wires=i)
        self._entangle_qubits(circular_connection=True, circular_position="start") # entangle after each layer
        for i in range(self.n_qubits):
            qml.RY(weights[i, 1+offset], wires=i)

    def circuit(self, inputs, weights):
        self._feature_map(inputs)
        for rep in range(self.reps):
            self._variational_layer(weights, layer_idx=rep)
        return self._measure()