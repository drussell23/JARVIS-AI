/**
 * Config-Aware Startup Handler
 * ===========================
 * Ensures all components wait for dynamic config before making API calls
 */

import configService from '../services/DynamicConfigService';

class ConfigAwareStartup {
  constructor() {
    this.initialized = false;
    this.configReady = false;
    this.pendingCallbacks = [];
    this.config = null;
    
    // Initialize immediately
    this.init();
  }
  
  async init() {
    console.log('[ConfigAwareStartup] Initializing...');
    
    // Wait for config with timeout
    try {
      this.config = await configService.waitForConfig(10000); // 10 second timeout
      this.configReady = true;
      console.log('[ConfigAwareStartup] Config ready:', this.config);
      
      // Execute pending callbacks
      this.pendingCallbacks.forEach(cb => cb(this.config));
      this.pendingCallbacks = [];
      
      // Listen for updates
      configService.on('config-updated', (newConfig) => {
        console.log('[ConfigAwareStartup] Config updated:', newConfig);
        this.config = newConfig;
      });
      
    } catch (error) {
      console.error('[ConfigAwareStartup] Config discovery timeout:', error);
      // Use fallback config
      this.config = {
        API_BASE_URL: 'http://localhost:8010',
        WS_BASE_URL: 'ws://localhost:8010'
      };
      this.configReady = true;
      console.log('[ConfigAwareStartup] Using fallback config:', this.config);
    }
    
    this.initialized = true;
  }
  
  /**
   * Execute callback when config is ready
   */
  whenReady(callback) {
    if (this.configReady) {
      callback(this.config);
    } else {
      this.pendingCallbacks.push(callback);
    }
  }
  
  /**
   * Get API URL (waits for config if needed)
   */
  async getApiUrl(endpoint = '') {
    if (!this.configReady) {
      await this.waitForReady();
    }
    
    const baseUrl = this.config.API_BASE_URL || 'http://localhost:8010';
    return endpoint ? `${baseUrl}/${endpoint.replace(/^\//, '')}` : baseUrl;
  }
  
  /**
   * Get WebSocket URL (waits for config if needed)
   */
  async getWebSocketUrl(endpoint = '') {
    if (!this.configReady) {
      await this.waitForReady();
    }
    
    const baseUrl = this.config.WS_BASE_URL || 'ws://localhost:8010';
    return endpoint ? `${baseUrl}/${endpoint.replace(/^\//, '')}` : baseUrl;
  }
  
  /**
   * Wait for config to be ready
   */
  async waitForReady(timeout = 10000) {
    if (this.configReady) {
      return this.config;
    }
    
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error('Config ready timeout'));
      }, timeout);
      
      const checkReady = () => {
        if (this.configReady) {
          clearTimeout(timeoutId);
          resolve(this.config);
        } else {
          setTimeout(checkReady, 100);
        }
      };
      
      checkReady();
    });
  }
  
  /**
   * Make API call with auto-discovery
   */
  async fetch(endpoint, options = {}) {
    const url = await this.getApiUrl(endpoint);
    console.log(`[ConfigAwareStartup] Fetching: ${url}`);
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        }
      });
      
      // If we get a 404 or connection error, try rediscovery
      if (response.status === 404 || !response.ok) {
        console.warn(`[ConfigAwareStartup] Request failed (${response.status}), triggering rediscovery`);
        configService.discover();
      }
      
      return response;
    } catch (error) {
      console.error('[ConfigAwareStartup] Fetch error:', error);
      // Trigger rediscovery on network errors
      configService.discover();
      throw error;
    }
  }
}

// Create singleton instance
const configAwareStartup = new ConfigAwareStartup();

// Export convenience functions
export const whenConfigReady = (callback) => configAwareStartup.whenReady(callback);
export const getApiUrl = async (endpoint) => configAwareStartup.getApiUrl(endpoint);
export const getWebSocketUrl = async (endpoint) => configAwareStartup.getWebSocketUrl(endpoint);
export const configFetch = async (endpoint, options) => configAwareStartup.fetch(endpoint, options);

export default configAwareStartup;