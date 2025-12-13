import React from 'react';

const RiskList = ({ riskAreas = [] }) => {
  const getRiskBadge = (level) => {
    switch(level) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="bg-white border border-gray-200 shadow-md rounded-lg">
      <div className="p-6">
        <h2 className="text-gray-800 text-xl font-semibold mb-2">All Risk Areas</h2>
        <p className="text-gray-600 mb-6">Areas listed from highest to lowest risk</p>

        <div className="space-y-4">
          {riskAreas.length > 0 ? (
            riskAreas.map(area => (
              <div key={area.id} className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <div className="flex items-start justify-between gap-4">
                  <h3 className="text-lg font-semibold text-gray-800">{area.name}</h3>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskBadge(area.riskLevel)}`}>
                    {area.riskLevel === 'high' && '⚠️ High Risk'}
                    {area.riskLevel === 'medium' && '🔶 Medium Risk'}
                    {area.riskLevel === 'low' && '✅ Low Risk'}
                  </span>
                </div>
                <p className="text-gray-600 mt-2">{area.description}</p>
                <div className="text-sm text-gray-500 mt-2">
                  <strong>{area.incidents}</strong> incidents in the last 30 days
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8 text-gray-500">
              No risk areas data available
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RiskList;