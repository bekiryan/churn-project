from fastapi import params
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import joblib

def load_data(file_path):
    """
    Load data from a CSV file and return features and labels.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    X (DataFrame): Features.
    y (Series): Labels.
    """
    data = pd.read_csv(file_path)
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    X (DataFrame): Features.
    y (Series): Labels.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    X_train (DataFrame): Training features.
    X_test (DataFrame): Testing features.
    y_train (Series): Training labels.
    y_test (Series): Testing labels.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_model(model, file_path):
    """
    Save the trained model to a file.
    
    Parameters:
    model: The trained model to save.
    file_path (str): Path where the model will be saved.
    """
    from ml_model.model import NeuralNetworkModel, RandomForestModel

    if isinstance(model, RandomForestModel):
        joblib.dump(model.model, file_path)

    elif isinstance(model, NeuralNetworkModel):
        torch.save(model.state_dict(), file_path)
    else:
        raise ValueError("Unsupported model type for saving.")
    
    print(f"Model saved to {file_path}")

def load_model(model_class, file_path, params=None):
    """
    Load a trained model from a file.

    Parameters:
    model_class: The class of the model to load.
    file_path (str): Path from where the model will be loaded.

    Returns:
    model: An instance of the model class with loaded weights.
    """
    from ml_model.model import NeuralNetworkModel, RandomForestModel
    if params is None:
        params = {}
    model = model_class(**params)

    if isinstance(model, RandomForestModel):
        model.model = joblib.load(file_path)

    elif isinstance(model, NeuralNetworkModel):
        model.load_state_dict(torch.load(file_path))
        model.eval()

    else:
        raise ValueError("Unsupported model type for loading.")
   
   
    print(f"Model loaded from {file_path}")
    return model


if __name__ == "__main__":
      # Load the dataset
    file_path = 'dataset.csv'  # Replace with your actual file path
    X, y = load_data(file_path)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Training set size:", X_train.shape[0])
    print("Testing set size:", X_test.shape[0])
