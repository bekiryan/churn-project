from ml_model.preprocessing import preprocess_data
from ml_model.model import RandomForestModel
from ml_model.data_loader import load_data_from_sql
from ml_model.utils import split_data

def train_model(model, params=None, db_path = 'db.sqlite'):
    if params is None:
        params = {}

    # Load data from SQL database
    data = load_data_from_sql(db_path)
    
    # Preprocess the data
    X, y = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Initialize and train the Random Forest model
    model = model(**params)
    model.train_model(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model's performance
    evaluation_metrics = model.evaluate(y_test, y_pred)
    print("Model Evaluation Metrics for", model.__class__.__name__)
    print(f"Accuracy: {evaluation_metrics['accuracy']}")

    print("Classification Report:")
    print(evaluation_metrics['classification_report'])

    return model


if __name__ == "__main__":
    model = train_model(RandomForestModel, {'n_estimators': 100, 'random_state': 42})


