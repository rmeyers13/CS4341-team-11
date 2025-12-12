// frontend/src/components/RiskMap.jsx
import React from 'react';

const RiskMap = ({ riskAreas = [] }) => {
  return (
    <div className="bg-white border border-gray-200 shadow-md rounded-lg">
      <div className="p-6">
        <h2 className="text-gray-800 text-xl font-semibold mb-4">Traffic Risk Areas</h2>

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

        <div className="relative h-96 bg-gray-50 border-2 border-gray-200 rounded-lg">
          {/* Map content will go here */}
          <div className="text-center text-gray-500 mt-40">
            Map visualization loading...
            <p className="text-sm mt-2">Risk areas: {riskAreas.length}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskMap;