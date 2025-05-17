import streamlit as st
from utils.api_clients import get_route_data, get_weather_data, get_traffic_data
from emission import predict_emission

import sqlite3

DB_PATH = "locations.db"

def load_locations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, latitude, longitude FROM locations ORDER BY name")
    rows = c.fetchall()
    conn.close()
    # Convert to dict: name -> "lat,lon" string
    return {name: f"{lat},{lon}" for name, lat, lon in rows}

# Load Delhi locations for dropdown
DELHI_LOCATIONS = load_locations()

st.title("🌿 Carbon Footprint Optimization in Supply Chain Logistics")

# Dropdowns instead of text input
origin_name = st.selectbox("Select Origin", list(DELHI_LOCATIONS.keys()), index=0)
destination_name = st.selectbox("Select Destination", list(DELHI_LOCATIONS.keys()), index=6)

origin = DELHI_LOCATIONS[origin_name]
destination = DELHI_LOCATIONS[destination_name]

cargo_weight = st.number_input("Cargo Weight (kg)", min_value=0.0, value=1000.0)
mileage = st.number_input("Vehicle Mileage (km/l)", min_value=1.0, value=15.0)
vehicle_type = st.selectbox("Vehicle Type", ["Small Van", "Medium Truck", "Large Truck"])
# 
if st.button("Calculate Route & Emissions"):
    route_data = get_route_data(origin, destination)
    weather_data = get_weather_data(origin)
    traffic_data = get_traffic_data(origin, destination)

    st.subheader("Route Info")
    st.write(f"Distance: {route_data['distance_km']:.2f} km")
    st.write(f"Duration: {route_data['duration_min']:.2f} minutes")

    st.subheader("Weather Info")
    st.write(f"Temperature: {weather_data['temperature']} C")
    st.write(f"Humidity: {weather_data['humidity']}%")

    st.subheader("Traffic Info")
    st.write(f"Traffic Level: {traffic_data['traffic_level']}")
    st.write(f"Traffic Delay: {traffic_data['traffic_delay']} minutes")

  

    inputs = {
        "distance": route_data["distance_km"],
        "duration": route_data["duration_min"],
        "temperature": weather_data["temperature"],
        "humidity": weather_data["humidity"],
        "traffic_level": traffic_data["traffic_level"],
        "traffic_delay": traffic_data["traffic_delay"],
        "cargo_weight": cargo_weight,
        "mileage": mileage,
        "vehicle_type": vehicle_type 
    }

    emission = predict_emission(inputs)
    st.success(f"Estimated Carbon Emission: {emission:.2f} kg CO2")
