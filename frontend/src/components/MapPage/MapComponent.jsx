import React from 'react';
import { GoogleMap, LoadScript, Marker } from '@react-google-maps/api';
import './mapStyles/MapComponent.css';

const containerStyle = {
    width: '100%',
    height: '600px'
};

const MapComponent = ({ center, zoom }) => {
    const mapOptions = {
        disableDefaultUI: false,
        zoomControl: true,
        streetViewControl: true,
        mapTypeControl: true,
        fullscreenControl: true,
    };

    return (
        <div className="map-wrapper">
            <LoadScript
                googleMapsApiKey={process.env.REACT_APP_GOOGLE_MAPS_API_KEY}
                libraries={['places']}
            >
                <GoogleMap
                    mapContainerStyle={containerStyle}
                    center={center}
                    zoom={zoom}
                    options={mapOptions}
                >
                    <Marker position={center} />
                </GoogleMap>
            </LoadScript>

            <div className="map-info">
                <p>Coordinates: {center.lat.toFixed(6)}, {center.lng.toFixed(6)}</p>
            </div>
        </div>
    );
};

export default MapComponent;