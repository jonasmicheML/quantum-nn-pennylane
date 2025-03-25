from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import QCircNet.circuits.QuantumCircuit as qc
import QCircNet.utils as ut

class QuantumClassifierMNIST(nn.Module):
    """
    Full quantum neural network.
    """
    def __init__(self, input_size=16, circuit:qc.QuantumCircuitNetwork=qc.QuantumCircuitNetwork, n_qubits:int=4, features_per_qubit:int=4, seed=42):
        """
        Args:
            input_size (int, optional): _description_. Defaults to 16.
            circuit (qc.QuantumCircuit, optional): _description_. Defaults to qc.QuantumCircuit.
            n_qubits (int, optional): _description_. Defaults to 4.
            features_per_qubit (int, optional): _description_. Defaults to 4.
            seed (int, optional): _description_. Defaults to 42.
        """
        super().__init__()
            
        self.n_qubits = n_qubits
        self.features_per_qubit = features_per_qubit
        
        # create quantum circuit nn
        self.quantum_circuit_nn = circuit(n_qubits=n_qubits, features_per_qubit=features_per_qubit, seed=seed)
        
        # assure that the model input fits to the qubits
        self.pre_processing = nn.Linear(input_size, n_qubits * features_per_qubit)
        # scale the quantum output the 10 classes
        self.post_processing = nn.Linear(1, 10)

        # For an existing model
        for param in self.pre_processing.parameters():
            param.requires_grad = False

        for param in self.post_processing.parameters():
            param.requires_grad = False



    def forward(self, x):
        """
        Forward pass through the entire model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, total_features]
            
        Returns:
            torch.Tensor: Predicted output values
            torch.Tensor: Logits before applying sigmoid (debugging purposes)
        """
        # pre-processing
        x = self.pre_processing(x)
        
        # quantum processing
        quantum_out = self.quantum_circuit_nn(x)
        
        # post-processing for classification
        logits = self.post_processing(quantum_out.unsqueeze(1))
        
        # apply softmax for probabilites
        probabilities = torch.softmax(logits, dim=1)
        
        return probabilities, logits
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on test data.

        Args:
            X_test (torch.Tensor): Test features
            y_test (torch.Tensor): Test targets (batchsize, 10)

        Returns:
            tuple: (cross_entropy_loss, accuracy, precision, recall, f1) metrics
        """ 
        self.eval()   
        with torch.no_grad():
            y_pred, _ = self(X_test)
            
            cross_entropy_loss = nn.CrossEntropyLoss()(y_pred, y_test).item()

            # covert one-hot to class indices
            y_pred_class = torch.argmax(y_pred, dim=1)
            y_test_class = torch.argmax(y_test, dim=1)

            # convert tensors to numpy arrays for sklearn metrics
            y_test_np = y_test_class.cpu().numpy()
            y_pred_np = y_pred_class.cpu().numpy()

            # compute classification metrics with weighted averaging
            accuracy = accuracy_score(y_test_np, y_pred_np)
            precision = precision_score(y_test_np, y_pred_np, average="weighted", zero_division=0)
            recall = recall_score(y_test_np, y_pred_np, average="weighted", zero_division=0)
            f1 = f1_score(y_test_np, y_pred_np, average="weighted", zero_division=0)

        # ensure the model is back in training mode
        self.train()

        return cross_entropy_loss, accuracy, precision, recall, f1
