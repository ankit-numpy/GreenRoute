import requests
import streamlit as st

ORS_API_KEY = st.secrets["ORS_API_KEY"]
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
TOMTOM_API_KEY = st.secrets["TOMTOM_API_KEY"]

def get_route_data(origin, destination):
    """
    Fetch detailed driving route data from OpenRouteService API.
    Returns:
        dict: {
            'distance_km': float,
            'duration_min': float,
            'geometry': list of [lat, lon] coordinates
        }
    """
    import requests

    ORS_API_KEY = st.secrets["ORS_API_KEY"]
    base_url = "https://api.openrouteservice.org/v2/directions/driving-car/json"

    origin_lat, origin_lon = map(float, origin.split(","))
    dest_lat, dest_lon = map(float, destination.split(","))

    coordinates = [[origin_lon, origin_lat], [dest_lon, dest_lat]]

    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "coordinates": coordinates
    }

    response = requests.post(base_url, json=body, headers=headers)

    if response.status_code != 200:
        raise Exception(f"ORS API error: {response.status_code} {response.text}")

    data = response.json()

    try:
        route = data["routes"][0]
        summary = route["summary"]
        geometry_encoded = route["geometry"]
    except KeyError as e:
        st.error("OpenRouteService response missing expected fields.")
        st.json(data)  # Show full response in debug
        raise Exception(f"Unexpected ORS response structure: {data}")

    # Decode the encoded geometry string
    import openrouteservice
    client = openrouteservice.Client(key=ORS_API_KEY)
    decoded_geometry = openrouteservice.convert.decode_polyline(geometry_encoded)["coordinates"]

    # Convert to [lat, lon]
    route_coords = [[lat, lon] for lon, lat in decoded_geometry]

    return {
        "distance_km": summary["distance"] / 1000,
        "duration_min": summary["duration"] / 60,
        "geometry": route_coords
    }


def get_weather_data(location):
    lat, lon = location.split(",")
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "weather_condition": data["weather"][0]["main"]
    }


def get_traffic_data(origin, destination):
    """
    Fetch traffic data from TomTom API.
    Args:
        origin (str): "lat,lon"
        destination (str): "lat,lon"
    Returns:
        dict: traffic_level and delay
    """
    import time
    from urllib.parse import urlencode

    TOMTOM_API_KEY = st.secrets["TOMTOM_API_KEY"]

    orig_lat, orig_lon = map(float, origin.split(','))
    dest_lat, dest_lon = map(float, destination.split(','))

    # Midpoint of route (for traffic query)
    mid_lat = (orig_lat + dest_lat) / 2
    mid_lon = (orig_lon + dest_lon) / 2

    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "point": f"{mid_lat},{mid_lon}",
        "key": TOMTOM_API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"TOMTOM API error: {response.status_code} {response.text}")

    data = response.json()

    flow_data = data.get("flowSegmentData", {})
    speed_band = flow_data.get("trafficSpeedBand", "unknown")
    delay = flow_data.get("currentTravelTime", 0) - flow_data.get("freeFlowTravelTime", 0)

    return {
        "traffic_level": speed_band,
        "traffic_delay": round(delay / 60, 2)  # delay in minutes
    }
