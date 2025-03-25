import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import QCircNet.utils as ut

class QuantumCircuitNetwork(nn.Module):
    """
    Base quantum circuit class for feature encoding and variational quantum operations.
    This class handles the common functionality while specific circuit architectures
    can be implemented in subclasses.
    """
    def __init__(self, n_qubits=4, features_per_qubit=4, seed=None):
        """ 
        Initialize the quantum circuit.

        Args:
            n_qubits (int): Number of qubits to use
            features_per_qubit (int): Number of features to encode on each qubit
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.features_per_qubit = features_per_qubit
        self.total_features = n_qubits * features_per_qubit
        self.seed = seed
        
        # init the quantum device
        self.device = qml.device("default.qubit", wires=n_qubits)
        # create the QNode with the circuit method 
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch")

        # initialize trainable weights (variational parameters)
        self._init_weight_shapes()
        self._init_weights()

    def _init_weights(self):
        """
        Set up the weights for the circuit.
        Default implementation has 2 layers of rotations for each qubit.
        """ 
        # set seed for reproducibility
        if self.seed:
            ut.set_seeds(self.seed)   

        # convert to PyTorch parameters
        weight_tensors = {
            name: torch.nn.Parameter(
                torch.Tensor(np.random.uniform(-2*np.pi, 2*np.pi, shape))
            )
            for name, shape in self.weight_shapes.items()
        }
        
        # register parameters with PyTorch
        for name, param in weight_tensors.items():
            self.register_parameter(name, param)

    def _entangle_qubits(self, circular_connection:bool=False, circular_position:str="end"):
        """
        Entangle the qubits in the circuit.
        Default implementation is a simple linear chain.
        """
        if circular_connection and circular_position == "start":
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        if circular_connection and circular_position == "end":
            qml.CNOT(wires=[self.n_qubits - 1, 0])
    
    def _feature_map(self, inputs):
        """
        Encode classical features into quantum states.
        Basic Encoding via rotation gates.
        """
        gates = [qml.RX, qml.RY, qml.RZ]
        num_gates = len(gates)
        for i in range(self.n_qubits):
            base_idx = i * self.features_per_qubit
            for j in range(self.features_per_qubit):
                index = j % num_gates
                gates[index](inputs[base_idx + 0], wires=i)

    def _init_weight_shapes(self):
        """ Initialize the shapes of the weights for the circuit. Has to be changed if _variational_layer() is changed """
        # 3 parameters per qubit (RX, RY, RZ)
        self.weight_shapes = {"weights": (self.n_qubits, 3)} 

    def _variational_layer(self, weights, layer_idx=0):
        """
        Apply a variational layer with trainable weights.
        
        Args:
            weights (torch.Tensor): The weights to use
            layer_idx (int): The starting index in the weights tensor for this layer
        """
        offset = layer_idx * 3
        for i in range(self.n_qubits):
            qml.Rot(weights[i, 0+offset], weights[i, 1+offset], weights[i, 2+offset], wires=i)

    def _measure(self):
        """ Define measurement operators."""       
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def circuit(self, inputs, weights):
        """
        The simple Quantum circuit definition.
        """
        # feature map
        self._feature_map(inputs)
        self._entangle_qubits()
        # variational layer
        self._variational_layer(weights, layer_idx=0)
        return self._measure() # measurement
    
    def forward(self, x):
        """
        Forward pass for a batch of data.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, total_features]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size]
        """
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size)

        # process each sample in the batch
        for i in range(batch_size):
            # pass through quantum circuit
            outputs[i] = torch.tensor(
                np.sum(self.qnode(x[i].detach().numpy(), self.weights.detach().numpy())),
                requires_grad=True
            )
        return outputs
    
    def draw(self, figwidth=10):
        """
        Draw the quantum circuit using PennyLane's built-in visualization.

        Args:
            figwidth (int): Width of the figure
        """
        dummy_inputs = np.random.uniform(1, 1, self.total_features) # dummy inputs for visualization
        # draw the circuit
        fig, ax = qml.draw_mpl(self.circuit)(dummy_inputs, self.weights.detach().numpy())

        fig.set_size_inches(figwidth, self.n_qubits)
        ax.set_title(f"{self.__class__.__name__} Visualization ({self.n_qubits} qubits)", fontsize=14)
        plt.close() # close the plot to avoid double plotting
        return fig
    