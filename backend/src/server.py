# backend/src/server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Load pre-trained model and preprocessing objects
model = joblib.load('risk_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Real-time data simulation
weather_conditions = {
    'Clear': 1, 'Cloudy': 2, 'Rain': 3, 'Snow': 4,
    'Fog': 5, 'Severe Crosswinds': 6
}

light_conditions = {
    'Daylight': 1, 'Dawn/Dusk': 2,
    'Dark - Lighted': 3, 'Dark - Unlighted': 4
}

road_conditions = {
    'Dry': 1, 'Wet': 2, 'Snow/Ice': 3,
    'Sand/Mud': 4, 'Water': 5
}


@app.route('/api/risk-predictions', methods=['GET'])
def get_risk_predictions():
    """Get risk predictions for predefined areas"""
    # For now, return enhanced sample data
    # TODO: Integrate with actual ML model
    risk_areas = [
        {
            "id": 1,
            "name": "Downtown Intersection",
            "riskLevel": "high",
            "riskScore": 0.87,
            "description": "Busy intersection with high traffic volume during rush hours",
            "incidents": 47,
            "longitude": -71.0589,
            "latitude": 42.3601,
            "features": {
                "lightLevel": "Daylight",
                "weather": "Clear",
                "roadCondition": "Dry",
                "trafficDensity": "High",
                "timeOfDay": "Peak Hours"
            },
            "confidence": 0.92,
            "lastUpdated": datetime.now().isoformat()
        },
        # ... other areas
    ]
    return jsonify({"riskAreas": risk_areas, "timestamp": datetime.now().isoformat()})


@app.route('/api/predict-risk', methods=['POST'])
def predict_risk():
    """Predict risk for a specific location and conditions"""
    data = request.json

    # Extract features
    features = np.array([[
        light_conditions.get(data['lightLevel'], 0),
        weather_conditions.get(data['weather'], 0),
        road_conditions.get(data['roadCondition'], 0),
        data['longitude'],
        data['latitude'],
        data.get('hourOfDay', 12),  # Time factor
        data.get('dayOfWeek', 3)  # Day factor
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0].max()

    # Convert to risk level
    risk_levels = ["low", "medium", "high"]
    risk_level = risk_levels[prediction]

    return jsonify({
        "riskLevel": risk_level,
        "riskScore": float(probability),
        "confidence": float(probability * 0.9),  # Adjust based on model confidence
        "factors": {
            "primary": "Traffic density",
            "secondary": "Weather conditions",
            "tertiary": "Road visibility"
        }
    })


@app.route('/api/weather-data', methods=['GET'])
def get_weather_data():
    """Get current weather conditions"""
    # Simulate weather API response
    return jsonify({
        "temperature": 72,
        "conditions": "Clear",
        "precipitation": 0,
        "visibility": "Good",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/historical-data', methods=['GET'])
def get_historical_data():
    """Get historical accident data for analysis"""
    # Load from CSV or database
    df = pd.read_csv('oddYears(2005-2019).csv')
    summary = {
        "totalIncidents": len(df),
        "bySeverity": df['Severity'].value_counts().to_dict(),
        "byWeather": df.groupby('Weather').size().to_dict(),
        "byTime": df.groupby('Hour').size().to_dict()
    }
    return jsonify(summary)


if __name__ == '__main__':
    app.run(debug=True, port=5000)