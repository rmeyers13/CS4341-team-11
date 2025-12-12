import React, { useState } from 'react';
import MapComponent from './components/MapComponent';
import SearchBar from './components/SearchBar';
import './App.css';

function App() {
    const [center, setCenter] = useState({ lat: 40.7128, lng: -74.0060 }); // Default: NYC
    const [zoom, setZoom] = useState(10);

    const handleLocationSelect = (coordinates) => {
        setCenter(coordinates);
        // Zoom in when a location is selected
        setZoom(14);
    };

    return (
        <div className="app">
            <header className="app-header">
                <h1>Location Map Finder</h1>
                <p>Search for any location and see it on the map</p>
            </header>

            <main className="app-main">
                <div className="search-container">
                    <SearchBar onLocationSelect={handleLocationSelect} />
                </div>

                <div className="map-container">
                    <MapComponent center={center} zoom={zoom} />
                </div>
            </main>

            <footer className="app-footer">
                <p>Powered by Google Maps API</p>
            </footer>
        </div>
    );
}

export default App;