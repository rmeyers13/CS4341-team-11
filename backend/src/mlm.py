import torch
import torch.nn as nn
import os
import math
from torch.serialization import add_safe_globals


# --- MODEL CLASS ---
class MultiOutputMLP(nn.Module):
    def __init__(self, input_size=3, output_size=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU()
        )
        self.head = nn.Linear(100, output_size)

    def forward(self, x):
        shared_out = self.shared(x)
        return self.head(shared_out)


# Allow this class for loading full model pickles
add_safe_globals([MultiOutputMLP])


# --- HELPER: ENCODE INPUTS ---
# Map frontend conditions to training encoding
def encode_inputs(conditions: dict):
    """Encode conditions to match training data encoding"""
    # Training encoding (from mlm_training.py):
    # LightLevel: 0=Unknown/Not reported/Other, 1=Dawn/Dusk, 2=Daylight, 3=Dark
    # Weather: 0=Unknown/Other, 1=Blowing sand/snow, 2=Severe crosswinds, 3=Clear, 4=Cloudy, 5=Fog, 6=Rain, 7=Severe crosswinds (rain), 8=Sleet, 9=Snow, 10=Severe crosswinds (rain), 11=Severe crosswinds (snow), 12=Severe crosswinds (sleet), 13=Severe crosswinds (fog)
    # RoadCondition: 0=Unknown/Other, 1=Dry, 2=Ice, 3=Sand/mud, 4=Slush, 5=Snow, 6=Water, 7=Wet

    light_map = {
        "Daylight": 2,
        "Dawn/Dusk": 1,
        "Dark - Lighted": 3,
        "Dark - Unlighted": 3,
        "Dark": 3,
    }

    weather_map = {
        "Clear": 3,
        "Cloudy": 4,
        "Fog": 5,
        "Rain": 6,
        "Snow": 9,
        "Severe Crosswinds": 2,  # Base severe crosswinds
    }

    surface_map = {
        "Dry": 1,
        "Wet": 7,
        "Snow/Ice": 2,  # Using Ice code for snow/ice
        "Sand/Mud": 3,
        "Water": 6,
    }

    light = conditions.get("light", "Daylight")
    weather = conditions.get("weather", "Clear")
    surface = conditions.get("surface", conditions.get("roadCondition", "Dry"))

    light_val = light_map.get(light, 2)  # Default to Daylight
    weather_val = weather_map.get(weather, 3)  # Default to Clear
    surface_val = surface_map.get(surface, 1)  # Default to Dry

    return torch.tensor([float(light_val), float(weather_val), float(surface_val)], dtype=torch.float32)


# --- LOAD MODEL ---
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model_save_10000")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print("📁 Loading model...")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Model structure: {model}")

    return model


# --- PREDICT LOCATION ---
def predict_location(model, conditions: dict):
    """Get predicted accident location (longitude, latitude) from model"""
    try:
        # Encode inputs
        x = encode_inputs(conditions)

        # Get model output
        with torch.no_grad():
            output = model(x)

        # Model outputs (longitude, latitude)
        lon, lat = output.tolist()

        return {
            "longitude": float(lon),
            "latitude": float(lat)
        }
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")