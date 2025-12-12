import React, { useState, useEffect } from 'react';
import './App.css';

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

  // Backend API URL
  const API_BASE_URL = 'http://localhost:5000/api';

  // Fetch risk areas from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);

        // Fetch risk predictions
        const riskResponse = await fetch(`${API_BASE_URL}/risk-predictions`);
        if (!riskResponse.ok) {
          throw new Error(`HTTP error! status: ${riskResponse.status}`);
        }
        const riskData = await riskResponse.json();
        setRiskAreas(riskData.riskAreas || []);

        // Fetch weather data
        const weatherResponse = await fetch(`${API_BASE_URL}/weather-data`);
        if (weatherResponse.ok) {
          const weatherData = await weatherResponse.json();
          setWeatherData(weatherData);
        }

        setError(null);
      } catch (error) {
        console.error('Error fetching data:', error);
        setError(`Failed to connect to backend: ${error.message}`);

        // Fallback to sample data if backend is down
        const sampleRiskAreas = getSampleRiskAreas();
        setRiskAreas(sampleRiskAreas);
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    // Update weather every 30 seconds
    const weatherInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/weather-data`);
        if (response.ok) {
          const data = await response.json();
          setWeatherData(data);
        }
      } catch (error) {
        console.log('Weather update failed, using fallback');
        // Fallback: simulate weather change
        setWeatherData(prev => ({
          ...prev,
          conditions: ['Clear', 'Cloudy', 'Rain', 'Snow'][Math.floor(Math.random() * 4)],
          temperature: 65 + Math.floor(Math.random() * 20)
        }));
      }
    }, 30000);

    return () => clearInterval(weatherInterval);
  }, []);

  // Sample data for fallback when backend is not available
  const getSampleRiskAreas = () => {
    return [
      {
        id: 1,
        name: "Downtown Intersection",
        riskLevel: "high",
        riskScore: 0.87,
        description: "Busy intersection with lots of traffic during rush hour",
        incidents: 47,
        longitude: -71.0589,
        latitude: 42.3601,
        position: { left: '35%', top: '25%', width: '15%', height: '12%' },
        features: {
          lightLevel: "Daylight",
          weather: "Clear",
          roadCondition: "Dry",
          trafficDensity: "High",
          timeOfDay: "Peak Hours"
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
        position: { left: '60%', top: '15%', width: '20%', height: '10%' },
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
        position: { left: '20%', top: '45%', width: '18%', height: '15%' },
        features: {
          lightLevel: "Dusk",
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
        position: { left: '65%', top: '60%', width: '16%', height: '12%' },
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
        position: { left: '75%', top: '40%', width: '15%', height: '18%' },
        features: {
          lightLevel: "Dark - lighted roadway",
          weather: "Clear",
          roadCondition: "Dry"
        }
      },
      {
        id: 6,
        name: "Park Boulevard",
        riskLevel: "low",
        riskScore: 0.28,
        description: "Low traffic area with good visibility",
        incidents: 5,
        longitude: -71.0389,
        latitude: 42.3401,
        position: { left: '45%', top: '75%', width: '12%', height: '10%' },
        features: {
          lightLevel: "Daylight",
          weather: "Clear",
          roadCondition: "Dry"
        }
      }
    ];
  };

  const showAreaDetails = (area) => {
    alert(`${area.name}\n\nRisk Level: ${area.riskLevel.toUpperCase()}\nRisk Score: ${area.riskScore}\nIncidents: ${area.incidents}\n\n${area.description}\n\nConditions:\n- Light: ${area.features?.lightLevel || 'N/A'}\n- Weather: ${area.features?.weather || 'N/A'}\n- Road: ${area.features?.roadCondition || 'N/A'}`);
  };

  const getRiskBadge = (riskLevel) => {
    switch(riskLevel) {
      case 'high': return 'bg-red-100 text-red-800 border-red-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getRiskEmoji = (riskLevel) => {
    switch(riskLevel) {
      case 'high': return '⚠️';
      case 'medium': return '🔶';
      case 'low': return '✅';
      default: return '❓';
    }
  };

  // Test backend connection
  const testBackendConnection = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/risk-predictions`);
      const data = await response.json();
      alert(`Backend connected successfully!\n\nAreas loaded: ${data.riskAreas?.length || 0}\nTimestamp: ${data.timestamp || 'N/A'}`);
    } catch (error) {
      alert(`Backend connection failed:\n\n${error.message}\n\nMake sure the Flask server is running on port 5000.`);
    }
  };

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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      <div className="container mx-auto p-6 max-w-6xl">
        {/* Header */}
        <header className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="bg-blue-500 p-3 rounded-full">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/>
              </svg>
            </div>
            <div>
              <h1 className="text-4xl font-bold text-gray-800">RiskMap</h1>
              <div className="flex items-center justify-center gap-2 mt-1">
                <div className={`w-2 h-2 rounded-full ${error ? 'bg-red-500' : 'bg-green-500 animate-pulse'}`}></div>
                <span className="text-sm text-gray-600">
                  {error ? 'Using fallback data' : 'Connected to backend'}
                </span>
              </div>
            </div>
          </div>
          <p className="text-gray-600 text-lg">See where traffic accidents might happen</p>
        </header>

        {/* Error message if backend is down */}
        {error && (
          <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-yellow-600">⚠️</span>
                <span className="text-yellow-800">{error}</span>
              </div>
              <button
                onClick={() => window.location.reload()}
                className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded text-sm hover:bg-yellow-200"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="mb-8 flex justify-center">
          <div className="inline-flex h-10 items-center justify-center rounded-lg bg-gray-100 p-1 text-gray-500">
            <button
              onClick={() => setActiveTab('map')}
              className={`inline-flex items-center justify-center whitespace-nowrap rounded-md px-4 py-1.5 text-sm font-medium transition-all ${activeTab === 'map' ? 'bg-white text-gray-900 shadow-sm' : ''}`}
            >
              🗺️ Map View
            </button>
            <button
              onClick={() => setActiveTab('list')}
              className={`inline-flex items-center justify-center whitespace-nowrap rounded-md px-4 py-1.5 text-sm font-medium transition-all ${activeTab === 'list' ? 'bg-white text-gray-900 shadow-sm' : ''}`}
            >
              📋 List View
            </button>
          </div>
        </div>

        {/* Map View */}
        {activeTab === 'map' && (
          <div className="bg-white border border-gray-200 shadow-md rounded-lg">
            <div className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-gray-800 text-xl font-semibold">Traffic Risk Areas</h2>
                <div className="text-sm text-gray-600">
                  Weather: <span className="font-medium">{weatherData.conditions}</span> | {weatherData.temperature}°F
                </div>
              </div>

              {/* Legend */}
              <div className="flex gap-4 mb-6">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-red-400 rounded"></div>
                  <span className="text-gray-600">High Risk</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-yellow-400 rounded"></div>
                  <span className="text-gray-600">Medium Risk</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-green-400 rounded"></div>
                  <span className="text-gray-600">Low Risk</span>
                </div>
              </div>

              {/* Map Container */}
              <div className="relative h-96 bg-gray-50 border-2 border-gray-200 rounded-lg">
                {/* Grid Background */}
                <div
                  className="absolute inset-0 opacity-20 pointer-events-none"
                  style={{
                    backgroundImage: 'repeating-linear-gradient(0deg, #ccc, #ccc 1px, transparent 1px, transparent 20px), repeating-linear-gradient(90deg, #ccc, #ccc 1px, transparent 1px, transparent 20px)',
                    backgroundSize: '20px 20px'
                  }}
                ></div>

                {/* Risk Zones */}
                {riskAreas.map(area => {
                  let bgColor, borderColor;
                  if (area.riskLevel === 'high') {
                    bgColor = 'bg-red-400';
                    borderColor = 'border-red-500';
                  } else if (area.riskLevel === 'medium') {
                    bgColor = 'bg-yellow-400';
                    borderColor = 'border-yellow-500';
                  } else {
                    bgColor = 'bg-green-400';
                    borderColor = 'border-green-500';
                  }

                  return (
                    <div
                      key={area.id}
                      className={`absolute ${bgColor} ${borderColor} border-2 rounded-md hover-grow cursor-pointer flex items-center justify-center p-2`}
                      style={area.position}
                      title={`${area.name} - ${area.riskLevel} risk`}
                      onClick={() => showAreaDetails(area)}
                    >
                      <span className="text-white text-sm font-semibold text-center leading-tight drop-shadow">
                        {area.name}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="mt-4 text-center text-gray-500 text-sm">
                Click on areas to see more details • {riskAreas.length} areas monitored
                {error && <span className="text-yellow-600 ml-2"> (Using fallback data)</span>}
              </div>
            </div>
          </div>
        )}

        {/* List View */}
        {activeTab === 'list' && (
          <div className="bg-white border border-gray-200 shadow-md rounded-lg">
            <div className="p-6">
              <h2 className="text-gray-800 text-xl font-semibold mb-2">All Risk Areas</h2>
              <p className="text-gray-600 mb-6">Areas listed from highest to lowest risk</p>

              <div className="space-y-4">
                {riskAreas
                  .sort((a, b) => {
                    const order = { high: 3, medium: 2, low: 1 };
                    return order[b.riskLevel] - order[a.riskLevel];
                  })
                  .map(area => (
                    <div key={area.id} className="bg-gray-50 border border-gray-200 hover:shadow-md transition-shadow duration-200 rounded-lg p-4">
                      <div className="space-y-3">
                        <div className="flex items-start justify-between gap-4">
                          <h3 className="text-lg font-semibold text-gray-800">{area.name}</h3>
                          <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getRiskBadge(area.riskLevel)}`}>
                            {getRiskEmoji(area.riskLevel)} {area.riskLevel.charAt(0).toUpperCase() + area.riskLevel.slice(1)} Risk
                          </span>
                        </div>
                        <p className="text-gray-600">{area.description}</p>
                        <div className="flex justify-between items-center">
                          <div className="text-sm text-gray-500">
                            <strong>{area.incidents}</strong> incidents in the last 30 days
                          </div>
                          <div className="text-sm text-gray-500">
                            Risk score: <strong>{area.riskScore?.toFixed(2) || 'N/A'}</strong>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                }
              </div>

              {/* Summary */}
              <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <div className="text-center text-gray-700">
                  <p className="font-medium">Summary</p>
                  <p className="text-sm mt-1">
                    Monitoring <strong>{riskAreas.length}</strong> areas •
                    <span className="text-red-600 font-medium"> {riskAreas.filter(a => a.riskLevel === 'high').length} high risk</span> •
                    <span className="text-yellow-600 font-medium"> {riskAreas.filter(a => a.riskLevel === 'medium').length} medium risk</span> •
                    <span className="text-green-600 font-medium"> {riskAreas.filter(a => a.riskLevel === 'low').length} low risk</span>
                  </p>
                  <p className="text-xs text-gray-500 mt-2">
                    Data source: {error ? 'Fallback sample data' : 'Backend API'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Backend Status Card */}
        <div className="mt-8">
          <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-gray-800">Backend Connection</h3>
                <p className="text-sm text-gray-600 mt-1">
                  {error ? 'Failed to connect to backend server' : 'Successfully connected to backend'}
                </p>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={testBackendConnection}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 text-sm"
                >
                  Test Connection
                </button>
                <button
                  onClick={() => window.open('http://localhost:5000/api/risk-predictions', '_blank')}
                  className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 text-sm"
                >
                  View API Response
                </button>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-600">API Endpoint</p>
                <p className="font-mono text-sm truncate">{API_BASE_URL}/risk-predictions</p>
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-600">Areas Loaded</p>
                <p className="text-lg font-semibold">{riskAreas.length}</p>
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-600">Connection Status</p>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${error ? 'bg-red-500' : 'bg-green-500'}`}></div>
                  <span className={error ? 'text-red-600' : 'text-green-600'}>
                    {error ? 'Offline' : 'Online'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;