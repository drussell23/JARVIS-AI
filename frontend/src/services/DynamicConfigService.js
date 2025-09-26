/**
 * Dynamic Configuration Service
 * =============================
 * Automatically discovers backend services and configures endpoints
 * with zero hardcoding. Features self-healing and intelligent retry.
 */

import logger from '../utils/DebugLogger';

class DynamicConfigService {
  constructor() {
    this.config = {
      API_BASE_URL: null,
      WS_BASE_URL: null,
      ENDPOINTS: {},
      SERVICES: {},
      discovered: false
    };

    // Discovery configuration
    // Prioritize known working ports and exclude problematic ones
    this.commonPorts = [8010, 8000, 3001, 8080, 8888];
    // Skip ports with known CORS issues: 5000 (Control Center), 8001 (node service)
    this.excludedPorts = [5000, 8001, 5001];
    this.discoveryTimeout = 500; // ms per port
    this.maxRetries = 3;

    // Service patterns for identification
    this.servicePatterns = {
      backend: {
        endpoints: ['/health', '/api/health', '/api', '/docs'],
        identifiers: ['jarvis', 'api', 'backend', 'fastapi']
      },
      websocket: {
        endpoints: ['/ws', '/websocket', '/socket.io'],
        identifiers: ['websocket', 'ws', 'realtime']
      }
    };

    // Health monitoring
    this.healthCheckInterval = 30000; // 30 seconds
    this.healthScores = new Map();
    this.lastHealthCheck = new Map();

    // Event listeners
    this.listeners = new Map();

    // Auto-discovery on instantiation
    this.discover();
  }

  async discover() {
    logger.config('🔍 Starting automatic service discovery...');

    // Try to load from localStorage first
    const cached = this.loadCachedConfig();
    if (cached) {
      logger.info('Found cached config, validating...');
      const isValid = await this.validateConfig(cached);
      if (isValid) {
        logger.success('✅ Using cached configuration', cached);
        this.config = cached;
        this.emit('config-ready', this.config);

        // Still run discovery in background to update
        this.backgroundDiscovery();
        return this.config;
      } else {
        logger.warning('❌ Cached config validation failed, will rediscover');
      }
    }

    logger.config('No valid cached config, starting fresh discovery...');

    // Full discovery
    const services = await this.discoverServices();

    if (services.backend) {
      this.config.API_BASE_URL = services.backend.url;
      this.config.WS_BASE_URL = services.backend.url.replace('http://', 'ws://');
      this.config.ENDPOINTS = services.backend.endpoints;
      this.config.discovered = true;

      // Save to cache
      this.saveConfig();

      // Start health monitoring
      this.startHealthMonitoring();

      logger.success('✅ Service discovery complete:', this.config);
      this.emit('config-ready', this.config);
    } else {
      logger.error('❌ No backend service found');
      this.emit('discovery-failed', { reason: 'No backend found' });

      // Retry discovery
      setTimeout(() => this.discover(), 5000);
    }

    return this.config;
  }

  async discoverServices() {
    const discovered = {};

    // Parallel port scanning for efficiency
    const scanPromises = this.commonPorts.map(port =>
      this.scanPort(port).catch(() => null)
    );

    const results = await Promise.all(scanPromises);

    // Process results
    for (const service of results) {
      if (service) {
        discovered[service.type] = service;
        this.healthScores.set(service.url, 1.0);
      }
    }

    // If no backend found on common ports, do a deeper scan
    if (!discovered.backend) {
      console.log('🔍 Deep scanning for backend service...');
      discovered.backend = await this.deepScan();
    }

    this.config.SERVICES = discovered;
    return discovered;
  }

  async scanPort(port) {
    // Skip excluded ports
    if (this.excludedPorts.includes(port)) {
      logger.debug(`Skipping excluded port ${port}`);
      return null;
    }
    
    const baseUrl = `http://localhost:${port}`;
    // Only log in debug mode to reduce console noise
    // logger.debug(`Scanning port ${port}...`);

    // Try to identify service type
    for (const [serviceType, patterns] of Object.entries(this.servicePatterns)) {
      for (const endpoint of patterns.endpoints) {
        try {
          const url = `${baseUrl}${endpoint}`;
          const response = await this.fetchWithTimeout(url, this.discoveryTimeout, true);

          if (response.ok) {
            // Analyze response to confirm service type
            let data;
            try {
              data = await response.json();
            } catch {
              data = await response.text();
            }

            const isMatch = this.identifyService(data, patterns.identifiers);

            if (isMatch || endpoint === '/health') {
              logger.success(`✅ Found ${serviceType} service on port ${port}`);
              logger.api(`Response from ${url}:`, typeof data === 'object' ? JSON.stringify(data).substring(0, 200) : data.substring(0, 200));

              // Discover all endpoints
              const endpoints = await this.discoverEndpoints(baseUrl);

              return {
                type: serviceType,
                port,
                url: baseUrl,
                endpoints,
                health: 1.0,
                lastSeen: new Date()
              };
            }
          }
        } catch (error) {
          // Continue scanning silently - these errors are expected
        }
      }
    }

    return null;
  }

  async discoverEndpoints(baseUrl) {
    const endpoints = {};
    
    logger.debug(`🔍 Discovering endpoints at ${baseUrl}...`);

    // Common API endpoints to check
    const endpointsToCheck = [
      { path: '/health', name: 'health' },
      { path: '/api/health', name: 'health' },
      { path: '/audio/ml/config', name: 'ml_audio_config' },
      { path: '/audio/ml/stream', name: 'ml_audio_stream' },
      { path: '/voice/jarvis/status', name: 'jarvis_status' },
      { path: '/voice/jarvis/activate', name: 'jarvis_activate', method: 'POST' },
      { path: '/api/wake-word/status', name: 'wake_word_status' },
      { path: '/vision/ws/vision', name: 'vision_websocket' },
      { path: '/ws', name: 'websocket' },
      { path: '/openapi.json', name: 'openapi' },
      { path: '/docs', name: 'docs' }
    ];

    // Check endpoints in parallel
    const checks = endpointsToCheck.map(async ({ path, name, method = 'HEAD' }) => {
      try {
        // Use appropriate method for discovery
        let discoveryMethod = 'GET';
        if (method === 'POST' || name === 'jarvis_activate') {
          // For POST endpoints, try GET first for discovery
          discoveryMethod = 'GET';
        }
        
        const options = {
          method: discoveryMethod,
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          }
        };
        
        const response = await this.fetchWithTimeout(`${baseUrl}${path}`, 500, true, options);
        
        // Accept 200, 204, 401, 405 (method not allowed means endpoint exists)
        if (response.ok || response.status === 401 || response.status === 405) {
          endpoints[name] = { path, method: endpointsToCheck.find(e => e.name === name)?.method || 'GET' };
          return { name, path, status: response.status, method };
        }
      } catch {
        // Endpoint doesn't exist
      }
      return null;
    });

    const results = await Promise.all(checks);

    // Log discovered endpoints
    const found = results.filter(r => r !== null);
    
    // Check if this is minimal mode based on discovered endpoints
    const hasAdvancedEndpoints = found.some(e => 
      e.name === 'ml_audio_config' || 
      e.name === 'wake_word_status' || 
      e.name === 'vision_websocket'
    );
    
    if (found.length > 0 && !hasAdvancedEndpoints) {
      logger.info(`⚡ Backend running in MINIMAL MODE at ${baseUrl}`);
      logger.info(`  ✅ Found ${found.length} basic endpoints`);
      logger.info(`  ⏳ Advanced features will be available when full mode starts`);
      logger.info(`  📌 Available endpoints:`, found.map(e => e.name).join(', '));
    } else if (found.length > 0) {
      logger.api(`📍 Discovered ${found.length} endpoints at ${baseUrl} (FULL MODE):`, found);
    } else {
      logger.warning(`⚠️  No endpoints found at ${baseUrl}`);
    }

    return endpoints;
  }

  identifyService(data, identifiers) {
    const dataStr = typeof data === 'string' ? data.toLowerCase() : JSON.stringify(data).toLowerCase();
    return identifiers.some(id => dataStr.includes(id));
  }

  async fetchWithTimeout(url, timeout, silent = false, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        signal: controller.signal,
        mode: 'cors',
        credentials: 'omit',
        ...options
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      // Only throw if not in silent mode (used during discovery)
      if (!silent) {
        throw error;
      }
      // Return a fake failed response for silent mode
      return { ok: false, status: 0 };
    }
  }

  async deepScan() {
    console.log('🔍 Performing deep scan (ports 3000-9999)...');

    // Scan in chunks to avoid overwhelming the system
    const chunkSize = 100;
    const startPort = 3000;
    const endPort = 9999;

    for (let port = startPort; port <= endPort; port += chunkSize) {
      const chunk = [];
      for (let p = port; p < Math.min(port + chunkSize, endPort + 1); p++) {
        // Skip excluded ports in deep scan
        if (!this.excludedPorts.includes(p)) {
          chunk.push(this.quickPortCheck(p));
        }
      }

      const results = await Promise.all(chunk);
      const openPorts = results.filter(p => p !== null);

      // Check each open port
      for (const openPort of openPorts) {
        const service = await this.scanPort(openPort);
        if (service && service.type === 'backend') {
          return service;
        }
      }
    }

    return null;
  }

  async quickPortCheck(port) {
    try {
      const response = await this.fetchWithTimeout(`http://localhost:${port}/health`, 200, true);
      return response.ok ? port : null;
    } catch {
      return null;
    }
  }

  // Configuration API
  getApiUrl(endpoint = '') {
    if (!this.config.API_BASE_URL) {
      logger.warning('API URL not yet discovered, config:', this.config);
      return null;
    }

    // Check if endpoint is a known name
    if (endpoint && this.config.ENDPOINTS[endpoint]) {
      const endpointConfig = this.config.ENDPOINTS[endpoint];
      // Handle both string (legacy) and object (new) endpoint formats
      const path = typeof endpointConfig === 'string' ? endpointConfig : endpointConfig.path;
      return `${this.config.API_BASE_URL}${path}`;
    }

    // Return base URL + endpoint
    return endpoint ? `${this.config.API_BASE_URL}/${endpoint.replace(/^\//, '')}` : this.config.API_BASE_URL;
  }

  // Get the HTTP method for a known endpoint
  getEndpointMethod(endpoint) {
    if (!endpoint || !this.config.ENDPOINTS[endpoint]) {
      return 'GET'; // Default to GET
    }

    const endpointConfig = this.config.ENDPOINTS[endpoint];
    if (typeof endpointConfig === 'string') {
      return 'GET'; // Legacy format, default to GET
    }

    return endpointConfig.method || 'GET';
  }

  getWebSocketUrl(endpoint = '') {
    if (!this.config.WS_BASE_URL) {
      logger.warning('WebSocket URL not yet discovered, config:', this.config);
      return null;
    }

    // Check for known websocket endpoints
    if (endpoint && this.config.ENDPOINTS[`${endpoint}_websocket`]) {
      return `${this.config.WS_BASE_URL}${this.config.ENDPOINTS[`${endpoint}_websocket`]}`;
    }

    return endpoint ? `${this.config.WS_BASE_URL}/${endpoint.replace(/^\//, '')}` : this.config.WS_BASE_URL;
  }

  async waitForConfig(timeout = 30000) {
    if (this.config.discovered) {
      return this.config;
    }

    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.off('config-ready', onReady);
        reject(new Error('Configuration discovery timeout'));
      }, timeout);

      const onReady = (config) => {
        clearTimeout(timeoutId);
        resolve(config);
      };

      this.once('config-ready', onReady);
    });
  }

  // Health monitoring and self-healing
  async startHealthMonitoring() {
    setInterval(async () => {
      for (const [url, score] of this.healthScores) {
        try {
          const response = await this.fetchWithTimeout(url + '/health', 1000);

          if (response.ok) {
            // Increase health score
            this.healthScores.set(url, Math.min(1.0, score * 1.1));
          } else {
            // Decrease health score
            this.healthScores.set(url, score * 0.9);

            if (score < 0.5) {
              console.warn(`⚠️ Service unhealthy: ${url}`);
              this.attemptHealing(url);
            }
          }
        } catch {
          this.healthScores.set(url, score * 0.8);
        }

        this.lastHealthCheck.set(url, new Date());
      }
    }, this.healthCheckInterval);
  }

  async attemptHealing(url) {
    console.log(`🔧 Attempting to heal service: ${url}`);

    // Strategy 1: Re-discover on different port
    const service = Object.values(this.config.SERVICES).find(s => s.url === url);
    if (service) {
      // Check nearby ports
      const originalPort = service.port;
      const portsToCheck = [
        originalPort + 1, originalPort - 1,
        originalPort + 10, originalPort - 10,
        originalPort + 1000, originalPort - 1000
      ];

      for (const port of portsToCheck) {
        const newService = await this.scanPort(port);
        if (newService && newService.type === service.type) {
          console.log(`✅ Found service on new port: ${port}`);

          // Update configuration
          service.port = port;
          service.url = newService.url;

          if (service.type === 'backend') {
            this.config.API_BASE_URL = newService.url;
            this.config.WS_BASE_URL = newService.url.replace('http://', 'ws://');
          }

          this.emit('service-relocated', { service, oldPort: originalPort, newPort: port });
          this.saveConfig();
          return;
        }
      }
    }

    // Strategy 2: Full re-discovery
    console.log('🔍 Full re-discovery triggered');
    this.discover();
  }

  // Event system
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  once(event, callback) {
    const wrapper = (...args) => {
      callback(...args);
      this.off(event, wrapper);
    };
    this.on(event, wrapper);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  emit(event, data) {
    logger.debug(`Emitting event: ${event}`, data);
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          logger.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }

  // Persistence
  saveConfig() {
    try {
      localStorage.setItem('jarvis_dynamic_config', JSON.stringify({
        ...this.config,
        savedAt: new Date().toISOString()
      }));
    } catch (error) {
      console.error('Failed to save config:', error);
    }
  }

  loadCachedConfig() {
    try {
      const cached = localStorage.getItem('jarvis_dynamic_config');
      if (cached) {
        const config = JSON.parse(cached);

        // Check if cache is fresh (less than 5 minutes old)
        const savedAt = new Date(config.savedAt);
        const age = Date.now() - savedAt.getTime();

        if (age < 5 * 60 * 1000) {  // 5 minutes
          logger.debug('Loading cached config, age:', Math.floor(age / 1000), 'seconds');
          return config;
        } else {
          logger.info('Cached config is stale, clearing...');
          this.clearCache();
        }
      }
    } catch (error) {
      console.error('Failed to load cached config:', error);
      this.clearCache();
    }
    return null;
  }

  async validateConfig(config) {
    try {
      // Quick health check on the cached URL
      const response = await this.fetchWithTimeout(config.API_BASE_URL + '/health', 1000);
      if (!response.ok) {
        logger.warning(`Cached config validation failed with status ${response.status}`);
        // Clear invalid cache
        this.clearCache();
        return false;
      }
      return true;
    } catch (error) {
      logger.error('Cached config validation error:', error);
      // Clear invalid cache
      this.clearCache();
      return false;
    }
  }

  // Clear cache method
  clearCache() {
    try {
      localStorage.removeItem('jarvis_dynamic_config');
      logger.info('Cleared stale config cache');
    } catch (error) {
      logger.error('Failed to clear cache:', error);
    }
  }

  // Background discovery
  async backgroundDiscovery() {
    // Run discovery in background to catch service changes
    setInterval(async () => {
      const services = await this.discoverServices();

      // Check for changes
      let hasChanges = false;

      for (const [type, service] of Object.entries(services)) {
        const existing = this.config.SERVICES[type];
        // Check if service is valid (not null)
        if (service) {
          if (!existing || existing.port !== service.port) {
            hasChanges = true;
            console.log(`🔄 Service ${type} changed: port ${existing?.port} -> ${service.port}`);
          }
        } else if (existing) {
          // Service was available but is now gone
          hasChanges = true;
          console.log(`⚠️ Service ${type} is no longer available`);
        }
      }

      if (hasChanges) {
        this.config.SERVICES = services;
        if (services.backend) {
          this.config.API_BASE_URL = services.backend.url;
          this.config.WS_BASE_URL = services.backend.url.replace('http://', 'ws://');
          this.config.ENDPOINTS = services.backend.endpoints;
        }

        this.saveConfig();
        this.emit('config-updated', this.config);
      }
    }, 300000);  // Every 5 minutes (reduce frequency to minimize console noise)
  }
}

// Create singleton instance
const configService = new DynamicConfigService();

// Export for use in other modules
export default configService;