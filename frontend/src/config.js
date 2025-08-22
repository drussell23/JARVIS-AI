/**
 * Frontend Configuration
 * Centralized configuration for API endpoints and constants
 */

// Get API URL from environment variable or use default
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// WebSocket URL derived from API base URL
export const WS_BASE_URL = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');

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