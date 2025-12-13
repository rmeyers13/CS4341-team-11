from flask import Flask, request, jsonify
from flask_cors import CORS
from mlm import load_model, predict_location, MultiOutputMLP
import torch
import torch.nn as nn
import random
import math
from datetime import datetime

# Enable PyTorch to load custom class from pickle
from torch.serialization import add_safe_globals

add_safe_globals([MultiOutputMLP])

app = Flask(__name__)
CORS(app)

# Load trained ML model at server startup
print("Loading ML model...")
try:
    model = load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

# Available environmental condition options
LIGHT_CONDITIONS = ["Daylight", "Dark - Lighted", "Dark - Unlighted", "Dawn/Dusk"]
WEATHER_CONDITIONS = ["Clear", "Rain", "Cloudy", "Snow", "Fog", "Severe Crosswinds"]
ROAD_CONDITIONS = ["Dry", "Wet", "Snow/Ice", "Sand/Mud", "Water"]


# Calculate great-circle distance between two geographic points
def haversine_distance(lat1, lon1, lat2, lon2):
    """Returns distance in miles using Earth's radius of 3958.8 miles"""
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# Convert distance to risk score with condition modifiers
def calculate_risk_from_distance(distance_miles, conditions):
    """Computes risk score from distance and environmental factors"""
    max_distance = 50.0

    if distance_miles > max_distance:
        risk_score = 0.0
    else:
        risk_score = 1.0 - (distance_miles / max_distance)

    condition_multiplier = 1.0

    # Increase risk for hazardous conditions
    if conditions["light"] in ["Dark - Unlighted", "Dark - Lighted"]:
        condition_multiplier *= 1.3
    if conditions["weather"] in ["Rain", "Snow", "Severe Crosswinds"]:
        condition_multiplier *= 1.4
    if conditions["surface"] in ["Wet", "Snow/Ice", "Water"]:
        condition_multiplier *= 1.3

    risk_score = min(1.0, risk_score * condition_multiplier)

    # Categorize risk level
    if risk_score >= 0.7:
        risk_level = "high"
    elif risk_score >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    confidence = 0.85  # Placeholder confidence value

    return risk_score, risk_level, confidence


# Generate simulated risk zones based on model predictions
def generate_risk_areas(num_areas=10):
    """Creates risk assessment zones using model predictions"""
    risk_areas = []

    # Boston coordinates as reference point
    base_lat = 42.3601
    base_lng = -71.0589

    for i in range(num_areas):
        # Random environmental conditions
        light = random.choice(LIGHT_CONDITIONS)
        weather = random.choice(WEATHER_CONDITIONS)
        road = random.choice(ROAD_CONDITIONS)

        conditions = {"light": light, "weather": weather, "surface": road}

        try:
            # Get model prediction for accident location
            location_pred = predict_location(model, conditions)
            pred_lat = location_pred["latitude"]
            pred_lon = location_pred["longitude"]

            # Calculate distance from reference point
            distance = haversine_distance(base_lat, base_lng, pred_lat, pred_lon)

            # Compute risk metrics
            risk_score, risk_level, confidence = calculate_risk_from_distance(distance, conditions)

            # Estimate incident count
            incidents = max(1, int(risk_score * 100))

        except Exception as e:
            print(f"Error generating area {i}: {e}")
            raise

        # Construct risk zone object
        area = {
            "id": i + 1,
            "name": f"Predicted Accident Zone {i + 1}",
            "riskLevel": risk_level,
            "riskScore": round(risk_score, 3),
            "description": f"Model predicts accidents near here under {light}, {weather}, {road} conditions",
            "incidents": incidents,
            "longitude": round(pred_lon, 6),
            "latitude": round(pred_lat, 6),
            "features": {
                "lightLevel": light,
                "weather": weather,
                "roadCondition": road,
                "trafficDensity": "Medium"
            },
            "modelConfidence": round(confidence, 2),
            "distanceFromCenter": round(distance, 2)
        }

        risk_areas.append(area)

    return risk_areas


# Root endpoint for server status
@app.route("/")
def root():
    return jsonify({
        "message": "RiskMap ML Backend Running",
        "model_loaded": True,
        "model_purpose": "Predicts accident locations based on conditions"
    })


# Health check endpoint
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "active",
        "timestamp": datetime.now().isoformat()
    })


# Diagnostic endpoint for model testing
@app.route("/api/debug-model", methods=["GET"])
def debug_model():
    """Tests model with predefined condition sets"""
    test_cases = [
        {"light": "Daylight", "weather": "Clear", "surface": "Dry"},
        {"light": "Dark - Unlighted", "weather": "Rain", "surface": "Wet"},
        {"light": "Dark - Lighted", "weather": "Snow", "surface": "Snow/Ice"},
    ]

    results = []
    base_lat = 42.3601
    base_lng = -71.0589

    for test in test_cases:
        try:
            location = predict_location(model, test)
            distance = haversine_distance(base_lat, base_lng, location["latitude"], location["longitude"])
            risk_score, risk_level, confidence = calculate_risk_from_distance(distance, test)

            results.append({
                "input": test,
                "predicted_location": location,
                "distance_from_boston_miles": round(distance, 2),
                "risk_score": round(risk_score, 3),
                "risk_level": risk_level,
                "confidence": round(confidence, 3)
            })
        except Exception as e:
            results.append({"input": test, "error": str(e)})

    return jsonify({"test_results": results})


# Main prediction endpoint for frontend requests
@app.route("/predict-risk", methods=["POST"])
def predict_risk():
    """Calculates accident risk based on user location and conditions"""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        # Extract user location and conditions
        user_lat = float(data.get("latitude", 42.3601))
        user_lon = float(data.get("longitude", -71.0589))
        conditions = {
            "light": data.get("lightLevel", "Daylight"),
            "weather": data.get("weather", "Clear"),
            "surface": data.get("roadCondition", "Dry")
        }

        # Get model prediction
        predicted_location = predict_location(model, conditions)

        # Calculate distance to predicted accident location
        distance = haversine_distance(
            user_lat, user_lon,
            predicted_location["latitude"], predicted_location["longitude"]
        )

        # Compute risk metrics
        risk_score, risk_level, confidence = calculate_risk_from_distance(distance, conditions)

        # Construct response
        response = {
            "success": True,
            "riskScore": round(risk_score, 3),
            "riskLevel": risk_level,
            "confidence": round(confidence, 3),
            "modelUsed": "MultiOutputMLP",
            "distanceToPredictedAccident": round(distance, 2),
            "predictedAccidentLocation": predicted_location,
            "inputConditions": conditions
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Model prediction failed: {str(e)}"
        }), 500


# Endpoint for simulated risk zone generation
@app.route("/risk-predictions", methods=["GET"])
def risk_predictions():
    """Returns collection of model-predicted risk zones"""
    try:
        risk_areas = generate_risk_areas(num_areas=10)

        return jsonify({
            "riskAreas": risk_areas,
            "modelStatus": "active",
            "areasGenerated": len(risk_areas),
            "generatedAt": datetime.now().isoformat(),
            "dataSource": "Model-predicted accident locations"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to generate risk areas: {str(e)}"
        }), 500


# Simple weather simulation endpoint
@app.route("/weather-data", methods=["GET"])
def weather_data():
    """Returns simulated weather data for frontend display"""
    conditions = ["Clear", "Cloudy", "Rain"]
    condition = random.choice(conditions)

    if condition == "Clear":
        temp = 75
        precip = 0
    elif condition == "Cloudy":
        temp = 65
        precip = 20
    else:
        temp = 60
        precip = 80

    return jsonify({
        "temperature": temp,
        "conditions": condition,
        "precipitation": precip,
        "visibility": "Good" if condition == "Clear" else "Moderate",
        "updated": datetime.now().isoformat()
    })


# Server startup
if __name__ == "__main__":
    print("RiskMap ML Backend - Accident Location Predictor")
    print("Available Endpoints:")
    print("   POST /predict-risk      - Predict risk based on distance to accident location")
    print("   GET  /risk-predictions  - Get model-predicted accident locations")
    print("   GET  /api/debug-model   - Debug model predictions")
    print("   GET  /weather-data      - Get basic weather")
    print("Server running on http://0.0.0.0:5000")

    app.run(host="0.0.0.0", port=5000, debug=True)