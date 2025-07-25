import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

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

def main():
    # Load the dataset
    file_path = 'dataset.csv'  # Replace with your actual file path
    X, y = load_data(file_path)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Training set size:", X_train.shape[0])
    print("Testing set size:", X_test.shape[0])

def save_model(model, file_path):
    """
    Save the trained model to a file.
    
    Parameters:
    model: The trained model to save.
    file_path (str): Path where the model will be saved.
    """
    torch.save(model.state_dict(), file_path)
    torch.save(model, file_path)
    
    print(f"Model saved to {file_path}")

def load_model(model_class, file_path):
    """
    Load a trained model from a file.

    Parameters:
    model_class: The class of the model to load.
    file_path (str): Path from where the model will be loaded.

    Returns:
    model: An instance of the model class with loaded weights.
    """
    model = model_class()
    model.load_state_dict(torch.load(file_path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {file_path}")
    return model


if __name__ == "__main__":
    main()