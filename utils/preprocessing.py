import pickle

def preprocess_input(route, weather, traffic, weight, vehicle_type, efficiency, fuel_type):
    # Manual encoding (can be replaced with sklearn pipelines)
    vehicle_map = {"diesel_truck": 0, "electric_van": 1, "hybrid": 2}
    fuel_map = {"diesel": 0, "petrol": 1, "electric": 2}

    features = [
        route["distance_km"],
        route["duration_min"],
        route["elevation_gain_m"],
        weather["temperature_C"],
        weather["wind_speed_kmph"],
        weather["precipitation_mm"],
        traffic["traffic_delay_min"],
        traffic["congestion_level"],
        weight,
        efficiency,
        vehicle_map[vehicle_type],
        fuel_map[fuel_type]
    ]

    return features[:10]  # Adjust based on model input

def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)
