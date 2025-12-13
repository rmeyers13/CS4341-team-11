// const express = require('express');
// const cors = require('cors');
// const axios = require('axios');
// require('dotenv').config();
//
// const app = express();
const PORT = process.env.PORT || 5000;
console.log(process.env.PORT);

// // Middleware
// app.use(cors());
// app.use(express.json());
//
// console.log('API Key type:', typeof process.env.REACT_APP_GOOGLE_MAPS_API_KEY);
// console.log('API Key value:', process.env.REACT_APP_GOOGLE_MAPS_API_KEY);
// console.log(process.env.PORT);
//
// console.log('=== ENV TEST ===');
// console.log('NODE_ENV:', process.env.NODE_ENV);
// console.log('All env vars:', process.env);
// console.log('REACT_APP_TEST:', process.env.REACT_APP_TEST);
//
// // Routes
// app.get('/api/coordinates', async (req, res) => {
//     try {
//         const { location } = req.query;
//
//         if (!location) {
//             return res.status(400).json({
//                 error: 'Location parameter is required',
//                 example: '/api/coordinates?location=Paris,France'
//             });
//         }
//
//         const response = await axios.get(
//             'https://maps.googleapis.com/maps/api/geocode/json',
//             {
//                 params: {
//                     address: location,
//                     key: process.env.GOOGLE_MAPS_API_KEY
//                 }
//             }
//         );
//
//         if (response.data.status === 'ZERO_RESULTS') {
//             return res.status(404).json({
//                 error: 'Location not found',
//                 message: `No results found for "${location}"`
//             });
//         }
//
//         if (response.data.status === 'REQUEST_DENIED') {
//             return res.status(403).json({
//                 error: 'API key invalid or missing',
//                 message: 'Please check your Google Maps API key'
//             });
//         }
//
//         if (response.data.status !== 'OK') {
//             return res.status(500).json({
//                 error: 'Geocoding service error',
//                 status: response.data.status
//             });
//         }
//
//         const result = response.data.results[0];
//         const { lat, lng } = result.geometry.location;
//
//         res.json({
//             lat,
//             lng,
//             address: result.formatted_address,
//             place_id: result.place_id,
//             types: result.types
//         });
//
//     } catch (error) {
//         console.error('Error fetching coordinates:', error);
//
//         if (error.code === 'ENOTFOUND') {
//             return res.status(503).json({
//                 error: 'Service unavailable',
//                 message: 'Could not connect to Google Maps API'
//             });
//         }
//
//         res.status(500).json({
//             error: 'Internal server error',
//             message: 'Failed to fetch coordinates'
//         });
//     }
// });
//
// // Health check endpoint
// app.get('/api/health', (req, res) => {
//     res.json({
//         status: 'OK',
//         timestamp: new Date().toISOString(),
//         service: 'Maps Locator API'
//     });
// });
//
// // API info endpoint
// app.get('/api', (req, res) => {
//     res.json({
//         name: 'Maps Locator API',
//         version: '1.0.0',
//         endpoints: {
//             geocode: {
//                 method: 'GET',
//                 path: '/api/coordinates',
//                 parameters: {
//                     location: 'string (required)'
//                 },
//                 example: '/api/coordinates?location=Paris,France'
//             },
//             health: {
//                 method: 'GET',
//                 path: '/api/health'
//             }
//         }
//     });
// });
//
// // 404 handler
// app.use('*', (req, res) => {
//     res.status(404).json({
//         error: 'Endpoint not found',
//         availableEndpoints: ['/api/coordinates', '/api/health', '/api']
//     });
// });
//
// // Error handling middleware
// app.use((err, req, res, next) => {
//     console.error(err.stack);
//     res.status(500).json({ error: 'Something went wrong!' });
// });
//
// app.listen(PORT, () => {
//     console.log(`
//   Maps Locator Backend Server
//   Local: http://localhost:${PORT}
//   API: http://localhost:${PORT}/api
//   Health: http://localhost:${PORT}/api/health
//   Mode: ${process.env.NODE_ENV || 'development'}
//   `);
// });