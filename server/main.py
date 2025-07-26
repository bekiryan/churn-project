import torch
from fastapi import FastAPI, Query
from ml_model.model import NeuralNetworkModel, RandomForestModel
from ml_model.preprocessing import preprocess_single_input
from ml_model.utils import load_model
from ml_model.config import RANDOM_FOREST_MODEL_PATH, NEURAL_NETWORK_MODEL_PATH
from server.data_viewer import CustomerData
from server.utils import ModelChoice


app = FastAPI()

# Load the trained models
rf_model = load_model(RandomForestModel, RANDOM_FOREST_MODEL_PATH)
nn_model = load_model(NeuralNetworkModel, NEURAL_NETWORK_MODEL_PATH, {'input_size': 19, 'hidden_size': 100, 'output_size': 2})



@app.post("/predict")
async def predict(data: CustomerData, model: ModelChoice = Query(..., description="Choose a model: Random Forest or Neural Network")):
    input_tensor = preprocess_single_input(data.dict())

    if len(input_tensor.shape) == 1:
        input_tensor = input_tensor.unsqueeze(0)

    if model == ModelChoice.rf:
        prediction = rf_model.predict(input_tensor)
        return {
            "model": "Random Forest",
            "prediction": prediction.tolist()
        }

    elif model == ModelChoice.nn:
        with torch.no_grad():
            output = nn_model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()

        return {
            "model": "Neural Network",
            "prediction": predicted_class,
            "probabilities": probabilities
        }