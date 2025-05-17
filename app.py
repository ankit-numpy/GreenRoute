import streamlit as st
import folium
from streamlit_folium import st_folium
from utils.api_clients import get_route_data, get_weather_data, get_traffic_data
from emission import predict_emission

st.title("🌿 Carbon Footprint Optimization in Supply Chain Logistics")

# Delhi locations for dropdown
import sqlite3

DB_PATH = "locations.db"

def load_locations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, latitude, longitude FROM locations ORDER BY name")
    rows = c.fetchall()
    conn.close()
    return {name: f"{lat},{lon}" for name, lat, lon in rows}

DELHI_LOCATIONS = load_locations()

if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.header("🚛 Delivery Details")
    origin_name = st.selectbox("📍 Select Start Location", list(DELHI_LOCATIONS.keys()), key="origin_select")
    destination_name = st.selectbox("📍 Select End Location", list(DELHI_LOCATIONS.keys()), key="dest_select")
    cargo_weight = st.number_input("⚖️ Cargo Weight (kg)", min_value=0.0, max_value=10000.0, value=1000.0, step=100.0)
    vehicle_type = st.selectbox("🚚 Vehicle Type", ["Small Van", "Medium Truck", "Large Truck"])
    mileage = st.number_input("⛽ Vehicle Mileage (km per liter)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)

    if origin_name == destination_name:
        st.warning("⚠️ Start and end locations cannot be the same.")

    calc_button = st.button("📊 Calculate Optimal Route")

if calc_button and origin_name != destination_name:
    origin = DELHI_LOCATIONS[origin_name]
    destination = DELHI_LOCATIONS[destination_name]

    with st.spinner("🔄 Fetching route, weather, and traffic data..."):
        route_info = get_route_data(origin, destination)
        weather = get_weather_data(origin)
        traffic = get_traffic_data(origin, destination)

        distance_km = route_info["distance_km"]
        duration_min = route_info["duration_min"]
        geometry = route_info["geometry"]

        traffic_factor_map = {"light": 1.0, "moderate": 1.2, "heavy": 1.5}
        traffic_factor = traffic_factor_map.get(traffic["traffic_level"], 1.0)

        weather_factor = 1.0
        if weather["weather_condition"].lower() in ["rain", "snow", "storm"]:
            weather_factor = 1.1

        cargo_factor = 1 + (cargo_weight / 10000)  

        emission_factor = 2.68  

        liters_consumed = distance_km / mileage
        carbon_emission_kg = liters_consumed * emission_factor * traffic_factor * weather_factor * cargo_factor

        st.session_state.result = {
            "distance_km": distance_km,
            "duration_min": duration_min,
            "traffic_level": traffic["traffic_level"],
            "traffic_delay": traffic["traffic_delay"],
            "weather_condition": weather["weather_condition"],
            "temperature": weather["temperature"],
            "humidity": weather["humidity"],
            "carbon_emission_kg": carbon_emission_kg,
            "geometry": geometry,
            "origin_name": origin_name,
            "destination_name": destination_name,
            "origin": origin,
            "destination": destination
        }

if st.session_state.result:
    res = st.session_state.result

    st.markdown(f"### 🛣️ Route Information")
    st.markdown(f"- **From:** 🏁 {res['origin_name']}")
    st.markdown(f"- **To:** 🏁 {res['destination_name']}")
    st.markdown(f"- **Distance:** 📏 {res['distance_km']:.2f} km")
    st.markdown(f"- **Estimated Duration:** ⏱️ {res['duration_min']:.1f} minutes")
    st.markdown(f"- **Traffic Level:** 🚦 {res['traffic_level'].capitalize()} (Delay: {res['traffic_delay']:.1f} minutes)")
    st.markdown(f"- **Weather at Origin:** 🌤️ {res['weather_condition']}, {res['temperature']}°C, Humidity {res['humidity']}%")

    origin = DELHI_LOCATIONS[origin_name]
    destination = DELHI_LOCATIONS[destination_name]
    route_info = get_route_data(origin, destination)
    weather = get_weather_data(origin)
    traffic = get_traffic_data(origin, destination)
    inputs = {
        "distance": route_info["distance_km"],
        "duration": route_info["duration_min"],
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "traffic_level": res["traffic_level"],
        "traffic_delay": traffic["traffic_delay"],
        "cargo_weight": cargo_weight,
        "mileage": mileage,
        "vehicle_type": vehicle_type 
    }

    emission = predict_emission(inputs)
    st.success(f"🌍 Estimated Carbon Emission: **{emission:.2f} kg CO₂**")

    mid_lat = (float(res["origin"].split(",")[0]) + float(res["destination"].split(",")[0])) / 2
    mid_lon = (float(res["origin"].split(",")[1]) + float(res["destination"].split(",")[1])) / 2
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=11)

    folium.PolyLine(res["geometry"], color="green", weight=6, opacity=0.7, tooltip="Optimal Route").add_to(m)
    folium.Marker(location=list(map(float, res["origin"].split(","))), popup=f"Start: {res['origin_name']}", icon=folium.Icon(color="blue", icon="play")).add_to(m)
    folium.Marker(location=list(map(float, res["destination"].split(","))), popup=f"End: {res['destination_name']}", icon=folium.Icon(color="red", icon="flag")).add_to(m)

    st_folium(m, width=700, height=500)
