/**
 * Frontend Configuration
 * Centralized configuration for API endpoints and constants
 */

import configService from './services/DynamicConfigService';

// Dynamic API URL - will be updated when config service discovers backend
let API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
let WS_BASE_URL = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');

// Update URLs when config service is ready
configService.on('config-ready', (config) => {
  API_BASE_URL = config.API_BASE_URL || API_BASE_URL;
  WS_BASE_URL = config.WS_BASE_URL || WS_BASE_URL;

  // Update exported values
  module.exports.API_BASE_URL = API_BASE_URL;
  module.exports.WS_BASE_URL = WS_BASE_URL;
});

// Also listen for config updates
configService.on('config-updated', (config) => {
  API_BASE_URL = config.API_BASE_URL || API_BASE_URL;
  WS_BASE_URL = config.WS_BASE_URL || WS_BASE_URL;

  // Update exported values
  module.exports.API_BASE_URL = API_BASE_URL;
  module.exports.WS_BASE_URL = WS_BASE_URL;
});

// Export dynamic URLs
export { API_BASE_URL, WS_BASE_URL };

// Other configuration constants
export const CONFIG = {
  // Speech recognition settings
  SPEECH_RECOGNITION_TIMEOUT: 15000,
  SPEECH_SYNTHESIS_RATE: 1.0,
  SPEECH_SYNTHESIS_PITCH: 0.95,
  SPEECH_SYNTHESIS_VOLUME: 1.0,

  // Audio settings
  AUDIO_SAMPLE_RATE: 44100,
  AUDIO_CHANNELS: 1,

  // Vision system settings
  VISION_UPDATE_INTERVAL: 2000,
  VISION_RECONNECT_ATTEMPTS: 5,
  VISION_RECONNECT_DELAY: 2000,

  // WebSocket settings
  WS_RECONNECT_DELAY: 3000,
  WS_MAX_RECONNECT_ATTEMPTS: 10,

  // ML Audio settings
  ML_AUDIO_ENABLED: true,
  ML_AUTO_RECOVERY: true,
  ML_MAX_RETRIES: 5,
  ML_RETRY_DELAYS: [100, 500, 1000, 2000, 5000],
  ML_ANOMALY_THRESHOLD: 0.8,
  ML_PREDICTION_THRESHOLD: 0.7
};

// Helper function to get current API URL
export const getApiUrl = (endpoint = '') => {
  const baseUrl = configService.getApiUrl() || API_BASE_URL;
  return endpoint ? `${baseUrl}/${endpoint.replace(/^\//, '')}` : baseUrl;
};

// Helper function to get current WebSocket URL
export const getWebSocketUrl = (endpoint = '') => {
  const baseUrl = configService.getWebSocketUrl() || WS_BASE_URL;
  return endpoint ? `${baseUrl}/${endpoint.replace(/^\//, '')}` : baseUrl;
};