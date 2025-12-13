import React from 'react';

const WeatherPanel = ({ data }) => {
  return (
    <div className="flex items-center space-x-2">
      <div className="text-center">
        <span className="text-lg">☀️</span>
        <p className="text-xs text-gray-600">{data?.temperature || '72'}°F</p>
      </div>
      <div>
        <p className="text-sm font-medium">{data?.conditions || 'Clear'}</p>
        <p className="text-xs text-gray-500">Updated just now</p>
      </div>
    </div>
  );
};

export default WeatherPanel;