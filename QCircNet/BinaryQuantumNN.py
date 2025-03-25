from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import QCircNet.circuits.QuantumCircuit as qc
import QCircNet.utils as ut

class BinQuantumNeuralNetwork(nn.Module):
    """
    Full quantum neural network.
    """
    def __init__(self, input_size=16, circuit:qc.QuantumCircuitNetwork=qc.QuantumCircuitNetwork, n_qubits:int=4, features_per_qubit:int=4, seed=42, hybrid=False):
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
        
        # preprocesing assures that the model input fits to the qubits
        self.pre_processing = nn.Linear(input_size, n_qubits * features_per_qubit)
        self.post_processing = nn.Linear(1, 1)

        for param in self.pre_processing.parameters():
            param.requires_grad = hybrid

        for param in self.post_processing.parameters():
            param.requires_grad = hybrid

    def forward(self, x):
        """
        Forward pass through the entire model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, total_features]
            
        Returns:
            torch.Tensor: Predicted output values
            torch.Tensor: Logits before applying sigmoid (debugging purposes)
        """
        # Pre-processing
        x = self.pre_processing(x)
        
        # Quantum processing
        quantum_out = self.quantum_circuit_nn(x)
        
        # Post-processing for binary classification
        logits = self.post_processing(quantum_out.unsqueeze(1))
        
        # Apply sigmoid to get probabilities between 0 and 1
        probabilities = torch.sigmoid(logits)
        
        return probabilities.squeeze(), logits.squeeze()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on test data.
        
        Args:
            model (nn.Module): Trained quantum neural network model
            X_test (torch.Tensor): Test features
            y_test (torch.Tensor): Test targets
        
        Returns:
            tuple: (bce, accuracy, precision, recall, f1) metrics
        """ 
        self.eval()   
        with torch.no_grad():
            y_pred, _ = self(X_test)
            
            bce = nn.BCELoss()(y_pred, y_test).item()

            # round predictions to get class labels
            y_pred_class = torch.round(y_pred)
            
            # convert tensors to numpy arrays for sklearn metrics
            y_test_np = y_test.cpu().numpy()
            y_pred_class_np = y_pred_class.cpu().numpy()
            
            # calculate classification metrics
            accuracy = accuracy_score(y_test_np, y_pred_class_np)
            precision = precision_score(y_test_np, y_pred_class_np, zero_division=0)
            recall = recall_score(y_test_np, y_pred_class_np, zero_division=0)
            f1 = f1_score(y_test_np, y_pred_class_np, zero_division=0)

        # assure the model is back in training mode
        self.train()
        
        return bce, accuracy, precision, recall, f1
    