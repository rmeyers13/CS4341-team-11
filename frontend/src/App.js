import React, { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import { GoogleMap, Marker, InfoWindow, useLoadScript } from '@react-google-maps/api';

function App() {
  const [activeTab, setActiveTab] = useState('map');
  const [riskAreas, setRiskAreas] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [weatherData, setWeatherData] = useState({
    temperature: 72,
    conditions: 'Clear',
    precipitation: 0,
    visibility: 'Good'
  });
  const [selectedArea, setSelectedArea] = useState(null);
  const [userLocation, setUserLocation] = useState({ lat: 42.3601, lng: -71.0589 });
  const [predictionResult, setPredictionResult] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [mapReady, setMapReady] = useState(false);
  const [predictionForm, setPredictionForm] = useState({
    lightLevel: 'Daylight',
    weather: 'Clear',
    roadCondition: 'Dry',
    longitude: -71.0589,
    latitude: 42.3601,
    timeOfDay: '12:00',
    trafficDensity: 'Medium'
  });

  const mapRef = useRef(null);

  const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';
  const GOOGLE_MAPS_API_KEY = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;

  const { isLoaded, loadError } = useLoadScript({
    googleMapsApiKey: GOOGLE_MAPS_API_KEY,
    libraries: ['places']
  });

  const lightLevelOptions = [
    { value: 'Daylight', label: 'Daylight' },
    { value: 'Dawn/Dusk', label: 'Dawn/Dusk' },
    { value: 'Dark - Lighted', label: 'Dark - Lighted' },
    { value: 'Dark - Unlighted', label: 'Dark - Unlighted' }
  ];

  const weatherOptions = [
    { value: 'Clear', label: 'Clear' },
    { value: 'Cloudy', label: 'Cloudy' },
    { value: 'Rain', label: 'Rain' },
    { value: 'Snow', label: 'Snow' },
    { value: 'Fog', label: 'Fog' },
    { value: 'Severe Crosswinds', label: 'Severe Crosswinds' }
  ];

  const roadConditionOptions = [
    { value: 'Dry', label: 'Dry' },
    { value: 'Wet', label: 'Wet' },
    { value: 'Snow/Ice', label: 'Snow/Ice' },
    { value: 'Sand/Mud', label: 'Sand/Mud' },
    { value: 'Water', label: 'Water (Standing)' }
  ];

  const trafficOptions = [
    { value: 'Low', label: 'Low Traffic' },
    { value: 'Medium', label: 'Medium Traffic' },
    { value: 'High', label: 'High Traffic' },
    { value: 'Congested', label: 'Congested' }
  ];

  const mapContainerStyle = {
    width: '100%',
    height: '500px'
  };

  useEffect(() => {
    fetchData();

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setUserLocation({ lat: latitude, lng: longitude });
          setPredictionForm(prev => ({
            ...prev,
            latitude: latitude,
            longitude: longitude
          }));
        },
        (error) => {
          console.log('Geolocation error:', error);
        }
      );
    }
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const riskResponse = await fetch(`${API_BASE_URL}/risk-predictions`);
      if (!riskResponse.ok) {
        throw new Error(`HTTP error! status: ${riskResponse.status}`);
      }
      const riskData = await riskResponse.json();
      setRiskAreas(riskData.riskAreas || []);

      const weatherResponse = await fetch(`${API_BASE_URL}/weather-data`);
      if (weatherResponse.ok) {
        const weatherData = await weatherResponse.json();
        setWeatherData(weatherData);
      }

    } catch (error) {
      console.error('Error fetching data:', error);
      setError(`Failed to connect to backend: ${error.message}`);

      const sampleRiskAreas = getSampleRiskAreas();
      setRiskAreas(sampleRiskAreas);
    } finally {
      setLoading(false);
    }
  };

  const getSampleRiskAreas = () => {
    return [
      {
        id: 1,
        name: "Downtown Intersection",
        riskLevel: "high",
        riskScore: 0.87,
        description: "Busy intersection with high traffic volume during rush hours",
        incidents: 47,
        longitude: -71.0589,
        latitude: 42.3601,
        features: {
          lightLevel: "Daylight",
          weather: "Clear",
          roadCondition: "Dry",
          trafficDensity: "High"
        }
      },
      {
        id: 2,
        name: "Highway 101 Exit",
        riskLevel: "high",
        riskScore: 0.82,
        description: "Cars merge quickly here and sometimes bump into each other",
        incidents: 39,
        longitude: -71.0689,
        latitude: 42.3701,
        features: {
          lightLevel: "Daylight",
          weather: "Cloudy",
          roadCondition: "Wet"
        }
      },
      {
        id: 3,
        name: "Main St & 5th Ave",
        riskLevel: "medium",
        riskScore: 0.64,
        description: "Moderate traffic with occasional congestion",
        incidents: 24,
        longitude: -71.0489,
        latitude: 42.3501,
        features: {
          lightLevel: "Dawn/Dusk",
          weather: "Clear",
          roadCondition: "Dry"
        }
      },
      {
        id: 4,
        name: "School Zone Area",
        riskLevel: "medium",
        riskScore: 0.58,
        description: "Lots of kids crossing the street when school starts and ends",
        incidents: 18,
        longitude: -71.0789,
        latitude: 42.3801,
        features: {
          lightLevel: "Daylight",
          weather: "Clear",
          roadCondition: "Dry"
        }
      },
      {
        id: 5,
        name: "Residential District",
        riskLevel: "low",
        riskScore: 0.32,
        description: "Quiet neighborhood with wide, well-lit streets",
        incidents: 8,
        longitude: -71.0889,
        latitude: 42.3901,
        features: {
          lightLevel: "Dark - Lighted",
          weather: "Clear",
          roadCondition: "Dry"
        }
      }
    ];
  };

  const getRiskColor = (riskLevel) => {
    switch(riskLevel) {
      case 'high': return '#ef4444';
      case 'medium': return '#f59e0b';
      case 'low': return '#10b981';
      default: return '#6b7280';
    }
  };

  const getMarkerIcon = (riskLevel) => {
  const color = getRiskColor(riskLevel);

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40">
      <circle cx="20" cy="20" r="18" fill="${color}" stroke="white" stroke-width="3"/>
      <circle cx="20" cy="20" r="10" fill="white" fill-opacity="0.7"/>
    </svg>
  `;

  return {
    url: `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`,
    scaledSize: { width: 40, height: 40 },
    anchor: { x: 20, y: 20 }
  };
};

  const onMapClick = useCallback((event) => {
  if (!event || !event.latLng) return;

  const lat = event.latLng.lat();
  const lng = event.latLng.lng();

  setPredictionForm(prev => ({
    ...prev,
    latitude: lat,
    longitude: lng
  }));

    if (mapRef.current) {
        mapRef.current.panTo({ lat, lng });
        mapRef.current.setZoom(15);
      }
    }, []);

  const handlePredictRisk = async (e) => {
    e.preventDefault();
    setIsPredicting(true);
    setPredictionResult(null);

    try {
      const url = `${API_BASE_URL}/predict-risk`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(predictionForm)
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictionResult(data);

      setUserLocation({
        lat: predictionForm.latitude,
        lng: predictionForm.longitude
      });

    } catch (error) {
      console.error('Prediction error:', error);
      setPredictionResult({
        error: error.message,
        riskLevel: "medium",
        riskScore: 0.5
      });
    } finally {
      setIsPredicting(false);
    }
  };

  const handleInputChange = (e) => {
  const { name, value } = e.target;

  if (name === 'latitude' || name === 'longitude') {
    setPredictionForm(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0
    }));
  } else {
    setPredictionForm(prev => ({
      ...prev,
      [name]: value
    }));
  }
};

  const testBackendConnection = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/risk-predictions`);
      const data = await response.json();
      alert(`Backend connected successfully!\n\nAreas loaded: ${data.riskAreas?.length || 0}\nStatus: ${data.modelStatus || 'N/A'}`);
    } catch (error) {
      alert(`Backend connection failed:\n\n${error.message}\n\nMake sure the Flask server is running on port 5000.`);
    }
  };

  if (!GOOGLE_MAPS_API_KEY) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 flex items-center justify-center p-4">
        <div className="text-center p-8 bg-white rounded-lg shadow-lg max-w-md">
          <div className="text-red-500 text-5xl mb-4">🔑</div>
          <h1 className="text-2xl font-bold text-gray-800 mb-2">Google Maps API Key Required</h1>
          <p className="text-gray-600 mb-4">
            Please add your Google Maps API key to the environment variables.
          </p>
          <div className="bg-gray-100 p-4 rounded text-left text-sm font-mono mb-4">
            REACT_APP_GOOGLE_MAPS_API_KEY=your_api_key_here
          </div>
          <p className="text-sm text-gray-500">
            Add this to a .env file in your frontend directory and restart the server.
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          <p className="mt-4 text-gray-600">Loading risk data from backend...</p>
          <button
            onClick={testBackendConnection}
            className="mt-4 px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm"
          >
            Test Backend Connection
          </button>
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 flex items-center justify-center p-4">
        <div className="text-center p-8 bg-white rounded-lg shadow-lg max-w-md">
          <div className="text-red-500 text-5xl mb-4">⚠️</div>
          <h1 className="text-2xl font-bold text-gray-800 mb-2">Failed to Load Google Maps</h1>
          <p className="text-gray-600 mb-4">
            {loadError.message || 'Unable to load Google Maps. Please check your API key and internet connection.'}
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      <div className="container mx-auto p-4 md:p-6 max-w-6xl">
        <header className="text-center mb-6 md:mb-8">
          <div className="flex items-center justify-center gap-3 mb-3 md:mb-4">
            <div className="bg-blue-500 p-2 md:p-3 rounded-full">
              <svg className="w-5 h-5 md:w-6 md:h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/>
              </svg>
            </div>
            <div>
              <h1 className="text-2xl md:text-4xl font-bold text-gray-800">RiskMap AI</h1>
              <div className="flex items-center justify-center gap-2 mt-1">
                <div className={`w-2 h-2 rounded-full ${error ? 'bg-red-500' : 'bg-green-500 animate-pulse'}`}></div>
                <span className="text-xs md:text-sm text-gray-600">
                  {error ? 'Using fallback data' : 'Connected to AI backend'}
                </span>
              </div>
            </div>
          </div>
          <p className="text-gray-600 text-sm md:text-lg">AI-Powered Traffic Risk Prediction System</p>
        </header>

        {error && (
          <div className="mb-4 md:mb-6 p-3 md:p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-yellow-600">⚠️</span>
                <span className="text-yellow-800 text-sm">{error}</span>
              </div>
              <button
                onClick={() => fetchData()}
                className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded text-sm hover:bg-yellow-200"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        <div className="mb-6 md:mb-8 flex justify-center">
          <div className="inline-flex h-10 items-center justify-center rounded-lg bg-gray-100 p-1 text-gray-500">
            <button
              onClick={() => setActiveTab('map')}
              className={`inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 md:px-4 py-1.5 text-sm font-medium transition-all ${activeTab === 'map' ? 'bg-white text-gray-900 shadow-sm' : ''}`}
            >
              🗺️ Live Map
            </button>
            <button
              onClick={() => setActiveTab('predict')}
              className={`inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 md:px-4 py-1.5 text-sm font-medium transition-all ${activeTab === 'predict' ? 'bg-white text-gray-900 shadow-sm' : ''}`}
            >
              🔮 AI Predictor
            </button>
            <button
              onClick={() => setActiveTab('list')}
              className={`inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 md:px-4 py-1.5 text-sm font-medium transition-all ${activeTab === 'list' ? 'bg-white text-gray-900 shadow-sm' : ''}`}
            >
              📋 Risk List
            </button>
          </div>
        </div>

        {activeTab === 'map' && (
          <div className="bg-white border border-gray-200 shadow-md rounded-lg overflow-hidden">
            <div className="p-4 md:p-6">
              <div className="flex flex-col md:flex-row md:justify-between md:items-center mb-4 gap-2">
                <h2 className="text-gray-800 text-lg md:text-xl font-semibold">Live Risk Map</h2>
                <div className="text-sm text-gray-600">
                  Weather: <span className="font-medium">{weatherData.conditions}</span> | {weatherData.temperature}°F
                </div>
              </div>

              <div className="flex flex-wrap gap-3 md:gap-4 mb-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 md:w-4 md:h-4 bg-red-500 rounded-full"></div>
                  <span className="text-xs md:text-sm text-gray-600">High Risk</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 md:w-4 md:h-4 bg-yellow-500 rounded-full"></div>
                  <span className="text-xs md:text-sm text-gray-600">Medium Risk</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 md:w-4 md:h-4 bg-green-500 rounded-full"></div>
                  <span className="text-xs md:text-sm text-gray-600">Low Risk</span>
                </div>
              </div>

              {!isLoaded ? (
                <div className="h-80 md:h-96 bg-gray-100 rounded-lg flex items-center justify-center border-2 border-dashed border-gray-300">
                  <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
                    <p className="mt-2 text-gray-600">Loading Google Maps...</p>
                  </div>
                </div>
              ) : (
                <GoogleMap
                  mapContainerStyle={mapContainerStyle}
                  zoom={12}
                  center={userLocation}
                  options={{
                    disableDefaultUI: false,
                    zoomControl: true,
                    streetViewControl: true,
                    mapTypeControl: true,
                    fullscreenControl: true
                  }}
                  onLoad={(map) => {
                    mapRef.current = map;
                    setTimeout(() => {
                      console.log('Map is now ready for markers');
                      setMapReady(true);
                    }, 100);
                  }}
                  onClick={onMapClick}
                >
                  {mapReady && riskAreas.map(area => (
                    <Marker
                      key={area.id}
                      position={{ lat: area.latitude, lng: area.longitude }}
                      icon={getMarkerIcon(area.riskLevel)}
                      onClick={() => setSelectedArea(area)}
                    />
                  ))}

                  {selectedArea && (
                    <InfoWindow
                      position={{ lat: selectedArea.latitude, lng: selectedArea.longitude }}
                      onCloseClick={() => setSelectedArea(null)}
                    >
                      <div className="p-2 max-w-xs">
                        <h3 className="font-bold text-lg">{selectedArea.name}</h3>
                        <div className={`px-2 py-1 rounded inline-block mt-1 ${
                          selectedArea.riskLevel === 'high' ? 'bg-red-100 text-red-800' :
                          selectedArea.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {selectedArea.riskLevel.toUpperCase()} RISK
                        </div>
                        <p className="mt-2 text-gray-600">{selectedArea.description}</p>
                        <p className="mt-1"><strong>Risk Score:</strong> {selectedArea.riskScore?.toFixed(2) || 'N/A'}</p>
                        <p><strong>Incidents:</strong> {selectedArea.incidents}</p>
                        <div className="mt-2 text-sm">
                          <p><strong>Conditions:</strong></p>
                          <p>Light: {selectedArea.features?.lightLevel || 'N/A'}</p>
                          <p>Weather: {selectedArea.features?.weather || 'N/A'}</p>
                          <p>Road: {selectedArea.features?.roadCondition || 'N/A'}</p>
                        </div>
                      </div>
                    </InfoWindow>
                  )}
                </GoogleMap>
              )}

              <div className="mt-3 md:mt-4 text-center text-xs md:text-sm text-gray-500">
                {isLoaded ? (
                  <>Click on markers for details • {riskAreas.length} areas monitored</>
                ) : (
                  <>Loading map markers...</>
                )}
                {error && <span className="text-yellow-600 ml-1"> (Using fallback data)</span>}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'predict' && (
          <div className="bg-white border border-gray-200 shadow-md rounded-lg overflow-hidden">
            <div className="p-4 md:p-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h2 className="text-gray-800 text-lg md:text-xl font-semibold mb-3 md:mb-4">AI Risk Predictor</h2>
                  <p className="text-gray-600 mb-4 md:mb-6 text-sm md:text-base">Enter conditions to get AI-powered risk prediction for any location</p>

                  <form onSubmit={handlePredictRisk} className="space-y-3 md:space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Light Condition
                      </label>
                      <select
                        value={predictionForm.lightLevel}
                        onChange={(e) => setPredictionForm(prev => ({...prev, lightLevel: e.target.value}))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm md:text-base"
                      >
                        {lightLevelOptions.map(option => (
                          <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Weather Condition
                      </label>
                      <select
                        value={predictionForm.weather}
                        onChange={(e) => setPredictionForm(prev => ({...prev, weather: e.target.value}))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm md:text-base"
                      >
                        {weatherOptions.map(option => (
                          <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Road Condition
                      </label>
                      <select
                        value={predictionForm.roadCondition}
                        onChange={(e) => setPredictionForm(prev => ({...prev, roadCondition: e.target.value}))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm md:text-base"
                      >
                        {roadConditionOptions.map(option => (
                          <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Traffic Density
                      </label>
                      <select
                        value={predictionForm.trafficDensity}
                        onChange={(e) => setPredictionForm(prev => ({...prev, trafficDensity: e.target.value}))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm md:text-base"
                      >
                        {trafficOptions.map(option => (
                          <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                      </select>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 md:gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Latitude
                        </label>
                        <input
                          type="number"
                          step="0.000001"
                          name="latitude"
                          value={predictionForm.latitude}
                          onChange={handleInputChange}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm md:text-base"
                          required
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Longitude
                        </label>
                        <input
                          type="number"
                          step="0.000001"
                          name="longitude"
                          value={predictionForm.longitude}
                          onChange={handleInputChange}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm md:text-base"
                          required
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Time of Day
                      </label>
                      <input
                        type="time"
                        name="timeOfDay"
                        value={predictionForm.timeOfDay}
                        onChange={handleInputChange}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm md:text-base"
                      />
                    </div>

                    <div className="pt-3 md:pt-4">
                      <button
                        type="submit"
                        disabled={isPredicting}
                        className={`w-full px-4 py-3 font-medium rounded-lg text-white text-sm md:text-base ${isPredicting ? 'bg-blue-400' : 'bg-blue-600 hover:bg-blue-700'} transition-colors`}
                      >
                        {isPredicting ? (
                          <>
                            <span className="inline-block animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white mr-2"></span>
                            Predicting...
                          </>
                        ) : '🔮 Get AI Risk Prediction'}
                      </button>
                    </div>
                  </form>

                  <div className="mt-4 md:mt-6">
                    <p className="text-sm text-gray-600 mb-2">Quick locations:</p>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          setPredictionForm(prev => ({
                            ...prev,
                            latitude: 42.3601,
                            longitude: -71.0589
                          }));
                        }}
                        className="px-3 py-1 text-xs md:text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                      >
                        Boston
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          if (navigator.geolocation) {
                            navigator.geolocation.getCurrentPosition((position) => {
                              setPredictionForm(prev => ({
                                ...prev,
                                latitude: position.coords.latitude,
                                longitude: position.coords.longitude
                              }));
                            });
                          }
                        }}
                        className="px-3 py-1 text-xs md:text-sm bg-green-100 text-green-700 rounded hover:bg-green-200"
                      >
                        My Location
                      </button>
                      <button
                        type="button"
                        onClick={() => setActiveTab('map')}
                        className="px-3 py-1 text-xs md:text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                      >
                        Pick on Map
                      </button>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-gray-800 text-lg font-semibold mb-3 md:mb-4">Prediction Results</h3>

                  {predictionResult ? (
                    <div className={`p-4 md:p-6 rounded-lg border-2 ${
                      predictionResult.error ? 'border-red-200 bg-red-50' :
                      predictionResult.riskLevel === 'high' ? 'border-red-200 bg-red-50' :
                      predictionResult.riskLevel === 'medium' ? 'border-yellow-200 bg-yellow-50' :
                      'border-green-200 bg-green-50'
                    }`}>
                      {predictionResult.error ? (
                        <div className="text-center">
                          <div className="text-red-500 text-4xl mb-2">❌</div>
                          <h4 className="text-lg font-medium text-red-800 mb-2">Prediction Failed</h4>
                          <p className="text-red-600">{predictionResult.error}</p>
                          <p className="text-gray-600 mt-2 text-sm">Make sure the backend server is running.</p>
                        </div>
                      ) : (
                        <>
                          <div className="flex items-center justify-between mb-4">
                            <div>
                              <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                                predictionResult.riskLevel === 'high' ? 'bg-red-100 text-red-800' :
                                predictionResult.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                                'bg-green-100 text-green-800'
                              }`}>
                                {predictionResult.riskLevel === 'high' ? '⚠️ HIGH RISK' :
                                 predictionResult.riskLevel === 'medium' ? '🔶 MEDIUM RISK' :
                                 '✅ LOW RISK'}
                              </div>
                            </div>
                            <div className="text-2xl md:text-3xl font-bold">
                              {(predictionResult.riskScore * 100).toFixed(0)}%
                            </div>
                          </div>

                          <div className="space-y-4">
                            <div>
                              <div className="flex justify-between text-sm mb-1">
                                <span>Risk Score</span>
                                <span>{(predictionResult.riskScore * 100).toFixed(1)}%</span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full ${
                                    predictionResult.riskLevel === 'high' ? 'bg-red-500' :
                                    predictionResult.riskLevel === 'medium' ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(predictionResult.riskScore * 100, 100)}%` }}
                                ></div>
                              </div>
                            </div>

                            {predictionResult.confidence && (
                              <div>
                                <div className="flex justify-between text-sm mb-1">
                                  <span>AI Confidence</span>
                                  <span>{(predictionResult.confidence * 100).toFixed(1)}%</span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                  <div
                                    className="h-2 rounded-full bg-blue-500"
                                    style={{ width: `${Math.min(predictionResult.confidence * 100, 100)}%` }}
                                  ></div>
                                </div>
                              </div>
                            )}

                            <div className="pt-3 md:pt-4 border-t border-gray-200">
                              <h4 className="font-medium text-gray-700 mb-2">Location Details</h4>
                              <div className="grid grid-cols-2 gap-2 text-sm">
                                  <div className="bg-gray-50 p-2 rounded">
                                    <div className="text-gray-500">Latitude</div>
                                    <div>{Number(predictionForm.latitude).toFixed(6)}</div>
                                  </div>
                                  <div className="bg-gray-50 p-2 rounded">
                                    <div className="text-gray-500">Longitude</div>
                                    <div>{Number(predictionForm.longitude).toFixed(6)}</div>
                                  </div>
                                </div>
                            </div>

                            <div className="pt-3 md:pt-4 border-t border-gray-200">
                              <h4 className="font-medium text-gray-700 mb-2">Input Conditions</h4>
                              <div className="flex flex-wrap gap-2">
                                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                                  {predictionForm.lightLevel}
                                </span>
                                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                                  {predictionForm.weather}
                                </span>
                                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                                  {predictionForm.roadCondition}
                                </span>
                                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                                  {predictionForm.trafficDensity}
                                </span>
                              </div>
                            </div>
                          </div>

                          <button
                            onClick={() => setActiveTab('map')}
                            className="w-full mt-4 md:mt-6 px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-900 text-sm md:text-base"
                          >
                            View on Map
                          </button>
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="p-6 md:p-8 text-center border-2 border-dashed border-gray-300 rounded-lg">
                      <div className="text-gray-400 mb-4">
                        <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                        </svg>
                      </div>
                      <h4 className="text-lg font-medium text-gray-600 mb-2">No Prediction Yet</h4>
                      <p className="text-gray-500 text-sm">Fill out the form and click "Get AI Risk Prediction" to see results</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'list' && (
          <div className="bg-white border border-gray-200 shadow-md rounded-lg">
            <div className="p-4 md:p-6">
              <h2 className="text-gray-800 text-lg md:text-xl font-semibold mb-2">All Risk Areas</h2>
              <p className="text-gray-600 mb-4 md:mb-6 text-sm md:text-base">Areas listed from highest to lowest risk</p>

              <div className="space-y-3 md:space-y-4">
                {riskAreas
                  .sort((a, b) => b.riskScore - a.riskScore)
                  .map(area => (
                    <div key={area.id} className="bg-gray-50 border border-gray-200 hover:shadow-md transition-shadow duration-200 rounded-lg p-3 md:p-4">
                      <div className="space-y-2 md:space-y-3">
                        <div className="flex items-start justify-between gap-3 md:gap-4">
                          <h3 className="text-base md:text-lg font-semibold text-gray-800">{area.name}</h3>
                          <span className={`px-2 py-1 rounded-full text-xs md:text-sm font-medium ${
                            area.riskLevel === 'high' ? 'bg-red-100 text-red-800' :
                            area.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                            {area.riskLevel === 'high' ? '⚠️' : area.riskLevel === 'medium' ? '🔶' : '✅'} {area.riskLevel.toUpperCase()} Risk
                          </span>
                        </div>
                        <p className="text-gray-600 text-sm md:text-base">{area.description}</p>
                        <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-2">
                          <div className="text-xs md:text-sm text-gray-500">
                            <strong>{area.incidents}</strong> incidents in the last 30 days
                          </div>
                          <div className="text-xs md:text-sm text-gray-500">
                            Risk score: <strong>{area.riskScore?.toFixed(2) || 'N/A'}</strong>
                          </div>
                        </div>
                        <div className="text-xs text-gray-500">
                          Coordinates: {area.latitude?.toFixed(4)}, {area.longitude?.toFixed(4)}
                        </div>
                      </div>
                    </div>
                  ))
                }
              </div>

              <div className="mt-4 md:mt-6 p-3 md:p-4 bg-blue-50 rounded-lg border border-blue-200">
                <div className="text-center text-gray-700">
                  <p className="font-medium text-sm md:text-base">Summary</p>
                  <p className="text-xs md:text-sm mt-1">
                    Monitoring <strong>{riskAreas.length}</strong> areas •
                    <span className="text-red-600 font-medium"> {riskAreas.filter(a => a.riskLevel === 'high').length} high risk</span> •
                    <span className="text-yellow-600 font-medium"> {riskAreas.filter(a => a.riskLevel === 'medium').length} medium risk</span> •
                    <span className="text-green-600 font-medium"> {riskAreas.filter(a => a.riskLevel === 'low').length} low risk</span>
                  </p>
                  <p className="text-xs text-gray-500 mt-2">
                    Data source: {error ? 'Fallback sample data' : 'AI Backend API'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="mt-6 md:mt-8">
          <div className="bg-white rounded-lg border border-gray-200 p-4 md:p-6 shadow-sm">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 md:gap-0">
              <div>
                <h3 className="font-semibold text-gray-800 text-sm md:text-base">AI Backend Status</h3>
                <p className="text-xs md:text-sm text-gray-600 mt-1">
                  {error ? 'Failed to connect to backend server' : 'Successfully connected to AI model'}
                </p>
              </div>
              <div className="flex gap-2 md:gap-3">
                <button
                  onClick={testBackendConnection}
                  className="px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 text-xs md:text-sm"
                >
                  Test Connection
                </button>
                <button
                  onClick={() => setActiveTab('predict')}
                  className="px-3 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 text-xs md:text-sm"
                >
                  Try AI Predictor
                </button>
              </div>
            </div>
            <div className="mt-3 md:mt-4 grid grid-cols-2 md:grid-cols-4 gap-2 md:gap-4">
              <div className="p-2 md:p-3 bg-gray-50 rounded">
                <p className="text-xs md:text-sm text-gray-600">AI Model</p>
                <p className="text-sm md:text-lg font-semibold">Random Forest</p>
              </div>
              <div className="p-2 md:p-3 bg-gray-50 rounded">
                <p className="text-xs md:text-sm text-gray-600">Areas Monitored</p>
                <p className="text-sm md:text-lg font-semibold">{riskAreas.length}</p>
              </div>
              <div className="p-2 md:p-3 bg-gray-50 rounded">
                <p className="text-xs md:text-sm text-gray-600">Connection Status</p>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${error ? 'bg-red-500' : 'bg-green-500'}`}></div>
                  <span className={`text-xs md:text-sm ${error ? 'text-red-600' : 'text-green-600'}`}>
                    {error ? 'Offline' : 'Online'}
                  </span>
                </div>
              </div>
              <div className="p-2 md:p-3 bg-gray-50 rounded">
                <p className="text-xs md:text-sm text-gray-600">Current Weather</p>
                <p className="text-sm md:text-lg font-semibold">{weatherData.conditions}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;