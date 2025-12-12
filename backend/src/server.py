# backend/src/server.py - FIXED VERSION
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import sys

app = Flask(__name__)
CORS(app)

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to load model, but handle missing files gracefully
try:
    model_path = os.path.join(os.path.dirname(__file__), 'risk_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    encoder_path = os.path.join(os.path.dirname(__file__), 'encoder.pkl')

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(" Model loaded successfully")
    else:
        model = None
        print("⚠ Model file not found, using fallback predictions")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None

    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
    else:
        encoder = {}

except Exception as e:
    print(f" Error loading model: {e}")
    model = None
    scaler = None
    encoder = {}

# Light condition mapping (must match training)
light_mapping = {
    'Daylight': 0,
    'Dawn/Dusk': 1,
    'Dark - Lighted': 2,
    'Dark - Unlighted': 3
}

weather_mapping = {
    'Clear': 0,
    'Cloudy': 1,
    'Rain': 2,
    'Snow': 3,
    'Fog': 4,
    'Severe Crosswinds': 5
}

road_mapping = {
    'Dry': 0,
    'Wet': 1,
    'Snow/Ice': 2,
    'Sand/Mud': 3,
    'Water': 4
}


@app.route('/api/risk-predictions', methods=['GET'])
def get_risk_predictions():
    """Get risk predictions for predefined areas"""

    # Sample areas with their features
    sample_areas = [
        {
            "id": 1,
            "name": "Downtown Intersection",
            "longitude": -71.0589,
            "latitude": 42.3601,
            "position": {"left": '35%', "top": '25%', "width": '15%', "height": '12%'},
            "lightLevel": "Daylight",
            "weather": "Clear",
            "roadCondition": "Dry",
            "description": "Busy intersection with high traffic volume during rush hours",
            "incidents": 47
        },
        {
            "id": 2,
            "name": "Highway 101 Exit",
            "longitude": -71.0689,
            "latitude": 42.3701,
            "position": {"left": '60%', "top": '15%', "width": '20%', "height": '10%'},
            "lightLevel": "Daylight",
            "weather": "Cloudy",
            "roadCondition": "Wet",
            "description": "Cars merge quickly here and sometimes bump into each other",
            "incidents": 39
        },
        {
            "id": 3,
            "name": "Main St & 5th Ave",
            "longitude": -71.0489,
            "latitude": 42.3501,
            "position": {"left": '20%', "top": '45%', "width": '18%', "height": '15%'},
            "lightLevel": "Dawn/Dusk",
            "weather": "Clear",
            "roadCondition": "Dry",
            "description": "Moderate traffic with occasional congestion",
            "incidents": 24
        },
        {
            "id": 4,
            "name": "School Zone Area",
            "longitude": -71.0789,
            "latitude": 42.3801,
            "position": {"left": '65%', "top": '60%', "width": '16%', "height": '12%'},
            "lightLevel": "Daylight",
            "weather": "Clear",
            "roadCondition": "Dry",
            "description": "Lots of kids crossing the street when school starts and ends",
            "incidents": 18
        },
        {
            "id": 5,
            "name": "Residential District",
            "longitude": -71.0889,
            "latitude": 42.3901,
            "position": {"left": '75%', "top": '40%', "width": '15%', "height": '18%'},
            "lightLevel": "Dark - Lighted",
            "weather": "Clear",
            "roadCondition": "Dry",
            "description": "Quiet neighborhood with wide, well-lit streets",
            "incidents": 8
        },
        {
            "id": 6,
            "name": "Park Boulevard",
            "longitude": -71.0389,
            "latitude": 42.3401,
            "position": {"left": '45%', "top": '75%', "width": '12%', "height": '10%'},
            "lightLevel": "Daylight",
            "weather": "Clear",
            "roadCondition": "Dry",
            "description": "Low traffic area with good visibility",
            "incidents": 5
        }
    ]

    risk_areas = []

    for area in sample_areas:
        # Try to get real prediction if model exists
        risk_level = "medium"  # default
        risk_score = 0.5

        if model is not None and scaler is not None:
            try:
                # Prepare features for prediction
                light_encoded = light_mapping.get(area['lightLevel'], 0)
                weather_encoded = weather_mapping.get(area['weather'], 0)
                road_encoded = road_mapping.get(area['roadCondition'], 0)

                # Create feature array
                features = np.array([[
                    light_encoded,
                    weather_encoded,
                    road_encoded,
                    area['longitude'],
                    area['latitude']
                ]])

                # Scale features if scaler exists
                if scaler:
                    features = scaler.transform(features)

                # Make prediction
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]

                # Map prediction to risk level
                risk_map = {0: 'low', 1: 'medium', 2: 'high'}
                risk_level = risk_map.get(prediction, 'medium')
                risk_score = float(proba.max())

            except Exception as e:
                print(f"Prediction error for area {area['id']}: {e}")
                # Use incident-based fallback
                if area['incidents'] > 30:
                    risk_level = 'high'
                    risk_score = 0.8
                elif area['incidents'] > 15:
                    risk_level = 'medium'
                    risk_score = 0.6
                else:
                    risk_level = 'low'
                    risk_score = 0.3
        else:
            # Model not available, use incident-based fallback
            if area['incidents'] > 30:
                risk_level = 'high'
                risk_score = 0.8 + (np.random.random() * 0.1)
            elif area['incidents'] > 15:
                risk_level = 'medium'
                risk_score = 0.5 + (np.random.random() * 0.2)
            else:
                risk_level = 'low'
                risk_score = 0.2 + (np.random.random() * 0.2)

        risk_areas.append({
            "id": area["id"],
            "name": area["name"],
            "riskLevel": risk_level,
            "riskScore": risk_score,
            "description": area["description"],
            "incidents": area["incidents"],
            "longitude": area['longitude'],
            "latitude": area['latitude'],
            "position": area.get('position', {}),
            "features": {
                "lightLevel": area['lightLevel'],
                "weather": area['weather'],
                "roadCondition": area['roadCondition']
            },
            "confidence": 0.85 if model else 0.5,
            "lastUpdated": datetime.now().isoformat()
        })

    return jsonify({
        "riskAreas": risk_areas,
        "timestamp": datetime.now().isoformat(),
        "modelStatus": "loaded" if model else "fallback"
    })


@app.route('/api/predict-risk', methods=['POST'])
def predict_risk():
    """Predict risk for custom input"""
    try:
        data = request.json

        if model is None or scaler is None:
            return jsonify({
                "error": "Model not loaded",
                "riskLevel": "medium",
                "riskScore": 0.5
            })

        # Encode features
        light_encoded = light_mapping.get(data.get('lightLevel', 'Daylight'), 0)
        weather_encoded = weather_mapping.get(data.get('weather', 'Clear'), 0)
        road_encoded = road_mapping.get(data.get('roadCondition', 'Dry'), 0)
        longitude = data.get('longitude', -71.0589)
        latitude = data.get('latitude', 42.3601)

        # Create feature array
        features = np.array([[
            light_encoded,
            weather_encoded,
            road_encoded,
            longitude,
            latitude
        ]])

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Map to risk level
        risk_map = {0: 'low', 1: 'medium', 2: 'high'}
        risk_level = risk_map.get(prediction, 'medium')

        return jsonify({
            "riskLevel": risk_level,
            "riskScore": float(probability.max()),
            "confidence": float(probability.max() * 0.9),
            "predictionDetails": {
                "classProbabilities": {
                    "low": float(probability[0]),
                    "medium": float(probability[1]),
                    "high": float(probability[2])
                }
            }
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "riskLevel": "medium",
            "riskScore": 0.5
        })


@app.route('/api/weather-data', methods=['GET'])
def get_weather_data():
    """Get current weather conditions"""
    import random
    conditions = ['Clear', 'Cloudy', 'Rain', 'Snow', 'Fog']

    return jsonify({
        "temperature": random.randint(60, 85),
        "conditions": random.choice(conditions),
        "precipitation": random.randint(0, 100),
        "visibility": random.choice(['Good', 'Fair', 'Poor']),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "modelLoaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })


if __name__ == '__main__':
    print(" Starting RiskMap Backend Server...")
    print(f" Current directory: {os.getcwd()}")
    print(f" Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    app.run(debug=True, port=5000, host='0.0.0.0')