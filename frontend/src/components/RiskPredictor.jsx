// frontend/src/components/RiskPredictor.jsx
import React, { useState } from 'react';

const RiskPredictor = ({ onPredict }) => {
  const [formData, setFormData] = useState({
    lightLevel: 'Daylight',
    weather: 'Clear',
    roadCondition: 'Dry',
    longitude: '-71.0589',
    latitude: '42.3601',
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (onPredict) {
      const result = await onPredict({
        ...formData,
        longitude: parseFloat(formData.longitude),
        latitude: parseFloat(formData.latitude),
      });
      alert(`Predicted Risk: ${result?.riskLevel || 'Unknown'}\nScore: ${result?.riskScore || 'N/A'}`);
    }
  };

  return (
    <div className="bg-white border border-gray-200 shadow-md rounded-lg">
      <div className="p-6">
        <h2 className="text-gray-800 text-xl font-semibold mb-4">Predict Risk for Location</h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Light Level</label>
              <select
                value={formData.lightLevel}
                onChange={(e) => setFormData({...formData, lightLevel: e.target.value})}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              >
                <option value="Daylight">Daylight</option>
                <option value="Dusk">Dusk</option>
                <option value="Dark - lighted roadway">Dark (Lighted)</option>
                <option value="Dark - roadway not lighted">Dark (Unlighted)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Weather</label>
              <select
                value={formData.weather}
                onChange={(e) => setFormData({...formData, weather: e.target.value})}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              >
                <option value="Clear">Clear</option>
                <option value="Cloudy">Cloudy</option>
                <option value="Rain">Rain</option>
                <option value="Snow">Snow</option>
                <option value="Fog">Fog</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Road Condition</label>
            <select
              value={formData.roadCondition}
              onChange={(e) => setFormData({...formData, roadCondition: e.target.value})}
              className="w-full border border-gray-300 rounded-lg px-3 py-2"
            >
              <option value="Dry">Dry</option>
              <option value="Wet">Wet</option>
              <option value="Ice">Ice</option>
              <option value="Snow">Snow</option>
              <option value="Sand/mud/dirt">Sand/Mud</option>
            </select>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Longitude</label>
              <input
                type="number"
                step="0.000001"
                value={formData.longitude}
                onChange={(e) => setFormData({...formData, longitude: e.target.value})}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Latitude</label>
              <input
                type="number"
                step="0.000001"
                value={formData.latitude}
                onChange={(e) => setFormData({...formData, latitude: e.target.value})}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              />
            </div>
          </div>

          <button
            type="submit"
            className="w-full bg-blue-500 text-white font-medium py-3 rounded-lg hover:bg-blue-600 transition"
          >
            Predict Risk Level
          </button>
        </form>
      </div>
    </div>
  );
};

export default RiskPredictor;