# train.py

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define emission factors per km (grams CO2 per km)
VEHICLE_EMISSION = {
    "Small Van": 120,
    "Medium Truck": 250,
    "Large Truck": 450
}

TRAFFIC_MAP = {
    "light": 0,
    "moderate": 1,
    "heavy": 2
}

VEHICLE_TYPE_MAP = {
    "Small Van": [1, 0, 0],
    "Medium Truck": [0, 1, 0],
    "Large Truck": [0, 0, 1]
}

# Generate synthetic dataset
def generate_data(n=200000):
    data = []
    for _ in range(n):
        distance = np.random.uniform(5, 100)  # km
        duration = distance / np.random.uniform(20, 60) * 60  # in minutes
        temperature = np.random.uniform(5, 45)
        humidity = np.random.uniform(20, 90)
        traffic_level = np.random.choice(["light", "moderate", "heavy"])
        traffic_delay = np.random.uniform(0, 30 if traffic_level == "heavy" else 15)

        cargo_weight = np.random.uniform(100, 10000)
        mileage = np.random.uniform(5, 20)
        vehicle_type = np.random.choice(["Small Van", "Medium Truck", "Large Truck"])

        # Emission base calculation (gm CO2/km * distance)
        base_emission = VEHICLE_EMISSION[vehicle_type] * distance / 1000  # convert to kg

        # Adjust for traffic, weather, cargo
        traffic_factor = 1 + (TRAFFIC_MAP[traffic_level] * 0.1)
        weather_factor = 1 + ((humidity / 100) * 0.05) + ((temperature > 35 or temperature < 10) * 0.05)
        cargo_factor = 1 + (cargo_weight / 10000) * 0.3

        # Final emission
        emission = base_emission * traffic_factor * weather_factor * cargo_factor

        features = [
            distance,
            duration,
            temperature,
            humidity,
            TRAFFIC_MAP[traffic_level],
            traffic_delay,
            cargo_weight,
            mileage,
            *VEHICLE_TYPE_MAP[vehicle_type]
        ]
        data.append((features, emission))
    return data

# Create dataset
data = generate_data()
X, y = zip(*data)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define model
class EmissionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# Training setup
model = EmissionModel(input_dim=X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Training loop
epochs = 3000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        test_loss = criterion(model(X_test_tensor), y_test_tensor)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# Save model and scaler
torch.save(model.state_dict(), "models/carbon_model.pt")
np.save("models/scaler_mean.npy", scaler.mean_)
np.save("models/scaler_scale.npy", scaler.scale_)

print("✅ Model and scaler saved successfully.")
