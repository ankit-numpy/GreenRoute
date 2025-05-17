# emission.py

import torch
import torch.nn as nn
import numpy as np

# Load scaler parameters
scaler_mean = np.load("models/scaler_mean.npy")
scaler_scale = np.load("models/scaler_scale.npy")

# Maps
TRAFFIC_MAP = {"light": 0, "moderate": 1, "heavy": 2}
VEHICLE_TYPE_MAP = {"Small Van": [1, 0, 0], "Medium Truck": [0, 1, 0], "Large Truck": [0, 0, 1]}

# Model (11 input features)
class EmissionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load trained model
model = EmissionModel()
model.load_state_dict(torch.load("models/carbon_model.pt"))
model.eval()

# Predict function
def predict_emission(inputs):
    traffic_encoded = TRAFFIC_MAP.get(inputs["traffic_level"], 1)
    vehicle_encoded = VEHICLE_TYPE_MAP.get(inputs["vehicle_type"], [0, 0, 1])

    feature_vector = [
        inputs["distance"],
        inputs["duration"],
        inputs["temperature"],
        inputs["humidity"],
        traffic_encoded,
        inputs["traffic_delay"],
        inputs["cargo_weight"],
        inputs["mileage"],
        *vehicle_encoded
    ]

    feature_vector = np.array(feature_vector, dtype=np.float32)
    scaled = (feature_vector - scaler_mean) / scaler_scale
    input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)


    with torch.no_grad():
        emission = model(input_tensor).item()

    return emission  # kg CO2
