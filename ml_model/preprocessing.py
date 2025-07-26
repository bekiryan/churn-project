# Preprocessing data for machine learning models
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import joblib


ENCODER_PATH = "label_encoders.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
CATEGORICAL_COLUMNS_PATH = "categorical_columns.pkl"

def preprocess_data(data: pd.DataFrame):
    data.fillna(method='ffill', inplace=True)
    data.drop_duplicates(inplace=True)

    if 'customerID' in data.columns:
        data.drop(columns=['customerID'], inplace=True)

    categorical_columns = data.select_dtypes(include='object').columns.tolist()

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop('Churn', axis=1)
    y = data['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_columns = X.columns.tolist()

    # Save preprocessing components
    joblib.dump(label_encoders, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    joblib.dump(categorical_columns, CATEGORICAL_COLUMNS_PATH)


    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)

def preprocess_single_input(data_dict: dict):
    label_encoders = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    categorical_columns = joblib.load(CATEGORICAL_COLUMNS_PATH)

    df = pd.DataFrame([data_dict])
    df = df.drop(columns=['customerID', 'Churn'], errors='ignore')

    for col in categorical_columns:
        le = label_encoders.get(col)
        if le and col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                df[col] = le.transform([le.classes_[0]])
        elif col not in df.columns:
            df[col] = 0 

    df = df.reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(df)

    return torch.tensor(scaled, dtype=torch.float32)

if __name__ == "__main__":
    #process a sample dataset
    sample_data = pd.read_csv('dataset.csv')
    X, y = preprocess_data(sample_data)
    print("Preprocessed data shapes:", X.shape, y.shape)
    print("Feature columns:", joblib.load(FEATURE_COLUMNS_PATH))
    print("Categorical columns:", joblib.load(CATEGORICAL_COLUMNS_PATH))