from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from ml_model.data_loader import load_data_from_sql
from ml_model.utils import load_data, split_data


class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42, *args, **kwargs):
        """
        Initialize the Random Forest model.
        
        Parameters:
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed for reproducibility.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, *args, **kwargs)

    def train(self, X_train, y_train):
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