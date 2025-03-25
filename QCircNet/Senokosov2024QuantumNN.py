from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import QCircNet.circuits.QuantumCircuit as qc
import QCircNet.utils as ut

class Senokosov2024QuantumClassifierMNIST(nn.Module):
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
        
        # CNN
        # 16x28x28 conv layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2, stride=1)
        # Batch normalization layer
        self.bn1 = nn.BatchNorm2d(16)
        # ReLU activation function
        self.relu = nn.ReLU()
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 32x14x14 conv layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=1)
        # Batch normalization layer
        self.bn2 = nn.BatchNorm2d(32)
        # ReLU activation function
        # Max pooling layer
        # flatten the output
        self.flatten = nn.Flatten()
        # fully connected layer with 1568 input features and 5 output features
        self.fc = nn.Linear(1568, 5)


        # assure that the model input fits to the qubits
        # self.pre_processing = nn.Linear(input_size, n_qubits * features_per_qubit)
        # scale the quantum output the 10 classes
        self.post_processing = nn.Linear(1, 10)

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
        # CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.pre_processing(x)
        
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
