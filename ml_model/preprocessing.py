# Preprocessing data for machine learning models
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(data: pd.DataFrame) -> tuple:
    """
    Load and preprocess data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    X (DataFrame): Preprocessed features.
    y (Series): Labels.
    """    
    # Handle missing values if any
    data.fillna(method='ffill', inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    # Separate features and labels
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=X.columns), y