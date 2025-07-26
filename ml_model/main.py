#train and save the models
from ml_model.config import RANDOM_FOREST_MODEL_PATH, NEURAL_NETWORK_MODEL_PATH
from ml_model.model import RandomForestModel, NeuralNetworkModel
from ml_model.train import train_model
from ml_model.utils import save_model

def main():
    # Train the Random Forest model
    rf_model = train_model(RandomForestModel, {'n_estimators': 100, 'random_state': 42})
    
    # Save the trained Random Forest model
    save_model(rf_model, RANDOM_FOREST_MODEL_PATH)
    
    # Train the Neural Network model (if implemented)
    nn_model = train_model(NeuralNetworkModel, {'input_size': 19, 'hidden_size': 100, 'output_size': 2})
    
    # Save the trained Neural Network model
    save_model(nn_model, NEURAL_NETWORK_MODEL_PATH)

if __name__ == "__main__":
    main()