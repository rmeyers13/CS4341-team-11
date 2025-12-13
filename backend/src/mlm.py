import torch
import torch.nn as nn
import os
import math
from torch.serialization import add_safe_globals

# Neural network for multi-output regression
class MultiOutputMLP(nn.Module):
    def __init__(self, input_size=3, output_size=2):
        super().__init__()
        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU()
        )
        # Output layer for regression
        self.head = nn.Linear(100, output_size)

    def forward(self, x):
        # Forward pass through network
        shared_out = self.shared(x)
        return self.head(shared_out)

# Allow model class for pickle loading
add_safe_globals([MultiOutputMLP])

# Convert categorical conditions to numerical encoding
def encode_inputs(conditions: dict):
    """
    Maps categorical weather/light/road conditions to numerical values
    used during model training.
    """
    # Mapping dictionaries for each input feature
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
        "Severe Crosswinds": 2,
    }
    surface_map = {
        "Dry": 1,
        "Wet": 7,
        "Snow/Ice": 2,
        "Sand/Mud": 3,
        "Water": 6,
    }

    # Extract values with defaults
    light = conditions.get("light", "Daylight")
    weather = conditions.get("weather", "Clear")
    surface = conditions.get("surface", conditions.get("roadCondition", "Dry"))

    # Apply mappings with fallback defaults
    light_val = light_map.get(light, 2)
    weather_val = weather_map.get(weather, 3)
    surface_val = surface_map.get(surface, 1)

    return torch.tensor([float(light_val), float(weather_val), float(surface_val)], dtype=torch.float32)

# Load trained model from disk
def load_model():
    """
    Loads the pre-trained model from saved checkpoint.
    Returns: Initialized model in evaluation mode.
    """
    model_path = os.path.join(os.path.dirname(__file__), "model_save_10000")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load model weights and architecture
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()  # Set to inference mode

    return model

# Generate location predictions from input conditions
def predict_location(model, conditions: dict):
    """
    Runs inference on input conditions to predict accident location.
    Returns: Dictionary with longitude and latitude coordinates.
    """
    # Convert conditions to model input format
    x = encode_inputs(conditions)

    # Forward pass without gradient computation
    with torch.no_grad():
        output = model(x)

    # Extract coordinates from model output
    lon, lat = output.tolist()

    return {
        "longitude": float(lon),
        "latitude": float(lat)
    }