# backend/src/server.py - FINAL FIXED VERSION
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from datetime import datetime
import os
import sys
import random

app = Flask(__name__)
# Fix CORS - allow all origins for development
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize model variables
model = None
scaler = None
encoder = None

# Try to load model, but handle missing files gracefully
try:
    model_path = os.path.join(os.path.dirname(__file__), 'risk_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    encoder_path = os.path.join(os.path.dirname(__file__), 'encoder.pkl')

    print(f"📁 Looking for model at: {model_path}")
    print(f"📁 Looking for scaler at: {scaler_path}")
    print(f"📁 Looking for encoder at: {encoder_path}")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
    else:
        model = None
        print("⚠️  Model file not found, using fallback predictions")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded successfully")
    else:
        scaler = None
        print("⚠️  Scaler file not found")

    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
        print("✅ Encoder loaded successfully")
    else:
        encoder = {}
        print("⚠️  Encoder file not found")

except Exception as e:
    print(f"❌ Error loading model: {e}")
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

# Reverse mapping for display
risk_map = {0: 'low', 1: 'medium', 2: 'high'}


def scale_features_for_prediction(longitude, latitude):
    """Scale only longitude and latitude using the scaler (which was trained on 2 features)"""
    if scaler is None:
        return longitude, latitude

    try:
        # Create array with only the 2 features the scaler expects
        features_to_scale = np.array([[longitude, latitude]])
        scaled_features = scaler.transform(features_to_scale)
        return float(scaled_features[0][0]), float(scaled_features[0][1])
    except Exception as e:
        print(f"⚠️  Scaling error: {e}")
        return longitude, latitude


def predict_with_model(light_level, weather, road_condition, longitude, latitude):
    """Make prediction using the loaded model"""
    if model is None:
        return None, None, None

    try:
        # Encode categorical features
        light_encoded = light_mapping.get(light_level, 0)
        weather_encoded = weather_mapping.get(weather, 0)
        road_encoded = road_mapping.get(road_condition, 0)

        # Scale numerical features
        scaled_longitude, scaled_latitude = scale_features_for_prediction(longitude, latitude)

        # Create feature array for model (5 features total)
        features = np.array([[
            light_encoded,
            weather_encoded,
            road_encoded,
            scaled_longitude,
            scaled_latitude
        ]])

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Map prediction to risk level
        risk_level = risk_map.get(prediction, 'medium')
        risk_score = float(probability.max())

        return risk_level, risk_score, probability
    except Exception as e:
        print(f"⚠️  Model prediction error: {e}")
        return None, None, None


@app.route('/api/risk-predictions', methods=['GET'])
def get_risk_predictions():
    """Get risk predictions for predefined areas"""
    print("📡 Received request for /api/risk-predictions")

    # Sample areas with their features
    sample_areas = [
        {
            "id": 1,
            "name": "Downtown Intersection",
            "longitude": -71.0589,
            "latitude": 42.3601,
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

        if model is not None:
            try:
                # Use the model to predict
                pred_risk_level, pred_risk_score, probabilities = predict_with_model(
                    area['lightLevel'],
                    area['weather'],
                    area['roadCondition'],
                    area['longitude'],
                    area['latitude']
                )

                if pred_risk_level is not None:
                    risk_level = pred_risk_level
                    risk_score = pred_risk_score
                else:
                    # Model prediction failed, use incident-based fallback
                    raise Exception("Model prediction returned None")

            except Exception as e:
                print(f"⚠️ Prediction error for area {area['id']}: {e}")
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
                risk_score = 0.8 + (random.random() * 0.1)
            elif area['incidents'] > 15:
                risk_level = 'medium'
                risk_score = 0.5 + (random.random() * 0.2)
            else:
                risk_level = 'low'
                risk_score = 0.2 + (random.random() * 0.2)

        risk_areas.append({
            "id": area["id"],
            "name": area["name"],
            "riskLevel": risk_level,
            "riskScore": float(risk_score),
            "description": area["description"],
            "incidents": area["incidents"],
            "longitude": area['longitude'],
            "latitude": area['latitude'],
            "features": {
                "lightLevel": area['lightLevel'],
                "weather": area['weather'],
                "roadCondition": area['roadCondition']
            },
            "confidence": 0.85 if model else 0.5,
            "lastUpdated": datetime.now().isoformat()
        })

    response = jsonify({
        "riskAreas": risk_areas,
        "timestamp": datetime.now().isoformat(),
        "modelStatus": "loaded" if model else "fallback",
        "totalAreas": len(risk_areas)
    })

    print(f"✅ Returning {len(risk_areas)} risk areas")
    return response


@app.route('/api/predict-risk', methods=['POST'])
def predict_risk():
    """Predict risk for custom input"""
    print("📡 Received prediction request")

    try:
        data = request.json
        print(f"📝 Prediction data: {data}")

        # Get input values with defaults
        light_level = data.get('lightLevel', 'Daylight')
        weather = data.get('weather', 'Clear')
        road_condition = data.get('roadCondition', 'Dry')
        longitude = float(data.get('longitude', -71.0589))
        latitude = float(data.get('latitude', 42.3601))
        traffic_density = data.get('trafficDensity', 'Medium')
        time_of_day = data.get('timeOfDay', '12:00')

        if model is not None:
            # Try to get real prediction if model exists
            risk_level, risk_score, probabilities = predict_with_model(
                light_level, weather, road_condition, longitude, latitude
            )

            if risk_level is not None:
                # Successfully got model prediction
                confidence = float(probabilities.max() * 0.9) if probabilities is not None else 0.7

                response_data = {
                    "riskLevel": risk_level,
                    "riskScore": float(risk_score),
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }

                if probabilities is not None and len(probabilities) >= 3:
                    response_data["predictionDetails"] = {
                        "classProbabilities": {
                            "low": float(probabilities[0]),
                            "medium": float(probabilities[1]),
                            "high": float(probabilities[2])
                        }
                    }

                print(f"✅ Model prediction: {risk_level} (score: {risk_score:.2f})")
                return jsonify(response_data)
            else:
                # Model prediction failed, fall through to fallback
                print("⚠️  Model prediction failed, using fallback")

        # Fallback prediction (when model is None or prediction failed)
        print("⚠️  Using fallback prediction")

        # Simple risk calculation based on conditions
        risk_score = 0.5

        # Adjust based on light level
        if light_level in ['Dark - Lighted', 'Dark - Unlighted']:
            risk_score += 0.2
        elif light_level == 'Dawn/Dusk':
            risk_score += 0.1

        # Adjust based on weather
        if weather in ['Rain', 'Snow']:
            risk_score += 0.25
        elif weather in ['Fog', 'Severe Crosswinds']:
            risk_score += 0.15
        elif weather == 'Cloudy':
            risk_score += 0.05

        # Adjust based on road condition
        if road_condition in ['Snow/Ice', 'Water']:
            risk_score += 0.3
        elif road_condition == 'Wet':
            risk_score += 0.15
        elif road_condition == 'Sand/Mud':
            risk_score += 0.1

        # Adjust based on traffic density
        if traffic_density in ['High', 'Congested']:
            risk_score += 0.2
        elif traffic_density == 'Medium':
            risk_score += 0.1

        # Adjust based on time of day
        try:
            hour = int(time_of_day.split(':')[0])
            if hour < 6 or hour >= 20:  # Night time
                risk_score += 0.15
            elif 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hour
                risk_score += 0.1
        except:
            pass

        # Ensure risk score is between 0.1 and 0.95
        risk_score = min(max(risk_score, 0.1), 0.95)

        # Determine risk level
        if risk_score > 0.7:
            risk_level = 'high'
        elif risk_score > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return jsonify({
            "riskLevel": risk_level,
            "riskScore": float(risk_score),
            "confidence": 0.7,
            "message": "Fallback prediction" + ("" if model is None else " (ML model prediction failed)"),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": str(e),
            "riskLevel": "medium",
            "riskScore": 0.5,
            "message": "Prediction failed - using fallback",
            "timestamp": datetime.now().isoformat()
        })


@app.route('/api/weather-data', methods=['GET'])
def get_weather_data():
    """Get current weather conditions"""
    conditions = ['Clear', 'Cloudy', 'Rain', 'Snow', 'Fog']

    response = jsonify({
        "temperature": random.randint(60, 85),
        "conditions": random.choice(conditions),
        "precipitation": random.randint(0, 100),
        "visibility": random.choice(['Good', 'Fair', 'Poor']),
        "timestamp": datetime.now().isoformat()
    })

    return response


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "RiskMap Backend",
        "modelLoaded": model is not None,
        "scalerLoaded": scaler is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "endpoints": {
            "GET /api/health": "Health check",
            "GET /api/test": "Test endpoint",
            "GET /api/risk-predictions": "Get risk predictions",
            "POST /api/predict-risk": "Predict risk for location",
            "GET /api/weather-data": "Get weather data"
        }
    })


@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint"""
    return jsonify({
        "message": "Backend is working!",
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    })


@app.route('/api/risk-predictions/simple', methods=['GET'])
def get_simple_risk_predictions():
    """Simple version without model - for testing"""
    sample_areas = [
        {
            "id": 1,
            "name": "Test Area 1",
            "riskLevel": "high",
            "riskScore": 0.85,
            "description": "Test area 1",
            "incidents": 30,
            "longitude": -71.0589,
            "latitude": 42.3601
        },
        {
            "id": 2,
            "name": "Test Area 2",
            "riskLevel": "medium",
            "riskScore": 0.55,
            "description": "Test area 2",
            "incidents": 15,
            "longitude": -71.0689,
            "latitude": 42.3701
        }
    ]

    return jsonify({
        "riskAreas": sample_areas,
        "timestamp": datetime.now().isoformat(),
        "modelStatus": "simple_test"
    })


# Add a root endpoint
@app.route('/')
def index():
    return jsonify({
        "message": "RiskMap Backend API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs": "Visit /api/health for API information"
    })


# Add a catch-all for /api/ routes that don't exist
@app.route('/api/<path:path>')
def api_catch_all(path):
    return jsonify({
        "error": f"Endpoint /api/{path} not found",
        "available_endpoints": [
            "/api/health",
            "/api/test",
            "/api/risk-predictions",
            "/api/predict-risk",
            "/api/weather-data",
            "/api/risk-predictions/simple"
        ]
    }), 404


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 Starting RiskMap Backend Server...")
    print("=" * 50)
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("\n📋 Available endpoints:")
    print("   GET  /                     - Root endpoint")
    print("   GET  /api/health          - Health check")
    print("   GET  /api/test            - Test endpoint")
    print("   GET  /api/risk-predictions - Get risk predictions")
    print("   POST /api/predict-risk    - Predict risk for location")
    print("   GET  /api/weather-data    - Get weather data")
    print("   GET  /api/risk-predictions/simple - Simple test data")
    print("\n🔗 Server will be available at:")
    print("   http://localhost:5000")
    print("   http://127.0.0.1:5000")
    print("=" * 50 + "\n")

    # Fix: Use threaded=True for better performance
    app.run(debug=True, port=5000, host='0.0.0.0', threaded=True)