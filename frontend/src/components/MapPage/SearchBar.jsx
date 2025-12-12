import React, { useState } from 'react';
import axios from 'axios';
import './mapStyles/SearchBar.css';

const SearchBar = ({ onLocationSelect }) => {
    const [query, setQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        setError('');

        try {
            // Call your backend API
            const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/api/coordinates`, {
                params: { location: query }
            });

            const { lat, lng } = response.data;
            onLocationSelect({ lat, lng });
        } catch (err) {
            setError(err.response?.data?.message || 'Failed to fetch coordinates');
            console.error('Error fetching coordinates:', err);
        } finally {
            setLoading(false);
        }
    };

    // Fallback to Google Maps Geocoding API directly (if backend is not ready)
    const handleDirectSearch = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        setError('');

        try {
            const response = await axios.get(
                `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(query)}&key=${process.env.REACT_APP_GOOGLE_MAPS_API_KEY}`
            );

            if (response.data.results.length > 0) {
                const location = response.data.results[0].geometry.location;
                onLocationSelect({ lat: location.lat, lng: location.lng });
            } else {
                setError('Location not found');
            }
        } catch (err) {
            setError('Failed to fetch location');
            console.error('Error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="search-bar">
            <form onSubmit={handleDirectSearch} className="search-form">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter a location (e.g., Paris, France)"
                    className="search-input"
                    disabled={loading}
                />
                <button
                    type="submit"
                    className="search-button"
                    disabled={loading}
                >
                    {loading ? 'Searching...' : 'Search'}
                </button>
            </form>

            {error && <div className="error-message">{error}</div>}

            <div className="examples">
                <p>Try: Tokyo, Japan • Sydney Opera House • Mount Everest</p>
            </div>
        </div>
    );
};

export default SearchBar;