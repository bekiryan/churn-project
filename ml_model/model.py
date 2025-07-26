from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from ml_model.data_loader import load_data_from_sql
from ml_model.utils import load_data, split_data
import torch
import torch.nn as nn

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42, *args, **kwargs):
        """
        Initialize the Random Forest model.
        
        Parameters:
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed for reproducibility.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, *args, **kwargs)

    def train_model(self, X_train, y_train):
        """
        Train the Random Forest model.
        
        Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict using the trained model.
        
        Parameters:
        X_test (DataFrame): Testing features.
        
        Returns:
        Series: Predicted labels.
        """
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model's performance.
        
        Parameters:
        y_true (Series): True labels.
        y_pred (Series): Predicted labels.
        
        Returns:
        dict: Evaluation metrics including accuracy and classification report.
        """
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=50, output_size=2):
        """
        Initialize the Neural Network model.
        
        Parameters:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units.
        output_size (int): Number of output classes.
        """
        super(NeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        x (Tensor): Input tensor.
        
        Returns:
        Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def train_model(self, X_train, y_train, criterion=nn.CrossEntropyLoss(), optimizer=None, epochs=100):
        """
        Train the Neural Network model.
        
        Parameters:
        X_train (Tensor): Training features.
        y_train (Tensor): Training labels.
        criterion: Loss function.
        optimizer: Optimizer for training.
        epochs (int): Number of training epochs.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        return self
    
    def predict(self, X_test):
        """
        Predict using the trained Neural Network model.
        
        Parameters:
        X_test (Tensor): Testing features.
        
        Returns:
        Tensor: Predicted labels.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X_test)
            _, predicted = torch.max(outputs.data, 1)
        return predicted
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate the Neural Network model's performance.
        
        Parameters:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predicted labels.
        
        Returns:
        dict: Evaluation metrics including accuracy and classification report.
        """
        accuracy = accuracy_score(y_true.numpy(), y_pred.numpy())
        report = classification_report(y_true.numpy(), y_pred.numpy())
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    