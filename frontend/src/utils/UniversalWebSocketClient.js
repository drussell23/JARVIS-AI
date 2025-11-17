/**
 * Universal WebSocket Client for JARVIS
 *
 * Supports dynamic configuration, health checks, capability negotiation, and state synchronization
 * Works with any backend WebSocket server following the JARVIS protocol v2.0
 *
 * Features:
 * - Zero hardcoded URLs (uses /api/config endpoint)
 * - Health check before WebSocket connection
 * - Automatic reconnection with exponential backoff
 * - Capability negotiation
 * - State synchronization
 * - Message buffering and replay
 *
 * @example
 * const client = new UniversalWebSocketClient({
 *   clientType: 'web',
 *   capabilities: ['voice', 'vision'],
 *   baseURL: 'http://localhost:8010' // Optional, will auto-detect
 * });
 *
 * client.on('message', (data) => console.log('Received:', data));
 * client.on('stateChange', (state) => console.log('State:', state));
 *
 * await client.connect();
 */

class UniversalWebSocketClient {
  constructor(options = {}) {
    // Client configuration
    this.clientId = options.clientId || this.generateClientId();
    this.clientType = options.clientType || 'web';
    this.clientCapabilities = options.capabilities || ['voice'];
    this.baseURL = options.baseURL || this.detectBaseURL();

    // Connection state
    this.connectionState = 'disconnected';
    this.websocket = null;
    this.configuration = null;
    this.reconnectAttempt = 0;
    this.reconnectTimer = null;

    // Server info
    this.serverVersion = 'unknown';
    this.serverCapabilities = [];

    // Event handlers
    this.eventHandlers = {
      message: [],
      stateChange: [],
      error: [],
      connected: [],
      disconnected: []
    };

    // Message queue for when disconnected
    this.messageQueue = [];

    console.log('🚀 UniversalWebSocketClient initialized', {
      clientId: this.clientId,
      clientType: this.clientType,
      capabilities: this.clientCapabilities,
      baseURL: this.baseURL
    });
  }

  // MARK: - Public API

  /**
   * Connect to the WebSocket server with full protocol flow
   */
  async connect() {
    try {
      await this.performConnectionFlow();
    } catch (error) {
      console.error('❌ Connection flow failed:', error);
      this.emit('error', error.message);
      await this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.websocket) {
      this.websocket.close(1000, 'Client disconnect');
      this.websocket = null;
    }

    this.setState('disconnected');
  }

  /**
   * Send a message to the server
   */
  send(message) {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      console.warn('⚠️  WebSocket not connected, queueing message');
      this.messageQueue.push(message);
      return false;
    }

    try {
      this.websocket.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('❌ Failed to send message:', error);
      this.emit('error', `Send failed: ${error.message}`);
      return false;
    }
  }

  /**
   * Add event listener
   */
  on(event, handler) {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event].push(handler);
    }
  }

  /**
   * Remove event listener
   */
  off(event, handler) {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event] = this.eventHandlers[event].filter(h => h !== handler);
    }
  }

  // MARK: - Connection Flow

  async performConnectionFlow() {
    console.log('🚀 Starting Universal WebSocket Connection Flow...');

    // Step 1: Fetch configuration
    this.setState('fetchingConfig');
    await this.fetchConfiguration();

    if (!this.configuration) {
      throw new Error('Failed to fetch configuration');
    }

    // Step 2: Health check (if required)
    if (this.configuration.websocket.health_check_required) {
      this.setState('healthCheck');
      const healthy = await this.performHealthCheck();
      if (!healthy) {
        throw new Error('Health check failed - backend not ready');
      }
    }

    // Step 3: Connect to WebSocket
    this.setState('connecting');
    await this.connectToWebSocket();
  }

  async fetchConfiguration() {
    console.log('📡 Fetching configuration from backend...');

    try {
      const response = await fetch(`${this.baseURL}/api/config`);

      if (!response.ok) {
        throw new Error(`Config endpoint returned ${response.status}`);
      }

      const config = await response.json();
      this.configuration = config;

      console.log('✅ Configuration fetched successfully');
      console.log('   WebSocket URL:', config.websocket.url);
      console.log('   Server capabilities:', config.capabilities.join(', '));

    } catch (error) {
      console.warn('⚠️  Failed to fetch config:', error);
      console.log('   Using fallback configuration...');
      this.useFallbackConfiguration();
    }
  }

  useFallbackConfiguration() {
    const wsProtocol = this.baseURL.startsWith('https') ? 'wss:' : 'ws:';
    const wsURL = this.baseURL.replace(/^https?:/, wsProtocol) + '/ws';

    this.configuration = {
      version: '2.0.0',
      websocket: {
        url: wsURL,
        ready: true,
        health_check_required: true,
        reconnect: {
          enabled: true,
          max_attempts: 999,
          base_delay_ms: 500,
          max_delay_ms: 15000,
          jitter: true
        }
      },
      http: {
        base_url: this.baseURL,
        health_endpoint: '/health',
        config_endpoint: '/api/config'
      },
      capabilities: ['voice', 'vision', 'commands']
    };

    console.log('✅ Fallback configuration loaded');
  }

  async performHealthCheck() {
    console.log('🏥 Performing health check...');

    try {
      const healthURL = `${this.configuration.http.base_url}/health`;
      const response = await fetch(healthURL);

      if (!response.ok) {
        throw new Error(`Health check returned ${response.status}`);
      }

      const health = await response.json();

      console.log('✅ Health check passed');
      console.log('   Status:', health.status);
      console.log('   WebSocket Ready:', health.websocket_ready);
      console.log('   Version:', health.version);
      console.log('   Uptime:', Math.floor(health.uptime), 's');

      this.serverVersion = health.version;
      this.serverCapabilities = health.capabilities || [];

      return health.websocket_ready &&
             (health.status === 'healthy' || health.status === 'degraded');

    } catch (error) {
      console.error('❌ Health check error:', error);
      return false;
    }
  }

  async connectToWebSocket() {
    return new Promise((resolve, reject) => {
      const wsURL = this.configuration.websocket.url;
      console.log('🔌 Connecting to WebSocket:', wsURL);

      try {
        this.websocket = new WebSocket(wsURL);

        this.websocket.onopen = () => {
          console.log('✅ WebSocket connected successfully');
          this.setState('connected');
          this.reconnectAttempt = 0;

          // Send capability negotiation handshake
          this.sendHandshake();

          // Send queued messages
          this.flushMessageQueue();

          this.emit('connected');
          resolve();
        };

        this.websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('❌ Failed to parse WebSocket message:', error);
          }
        };

        this.websocket.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          this.emit('error', 'WebSocket connection error');
        };

        this.websocket.onclose = (event) => {
          console.log('🔌 WebSocket disconnected', event.code, event.reason);
          this.websocket = null;
          this.setState('disconnected');
          this.emit('disconnected');

          // Auto-reconnect unless it was a normal closure
          if (event.code !== 1000) {
            this.scheduleReconnect();
          }
        };

      } catch (error) {
        console.error('❌ Failed to create WebSocket:', error);
        reject(error);
      }
    });
  }

  sendHandshake() {
    const handshake = {
      type: 'client_connect',
      client_id: this.clientId,
      client_type: this.clientType,
      version: '2.0.0',
      capabilities: this.clientCapabilities,
      request_state: true
    };

    this.send(handshake);
    console.log('📤 Sent capability negotiation handshake');
  }

  handleMessage(data) {
    // Detect welcome message
    if (data.type === 'welcome') {
      console.log('📨 Received welcome message');
      console.log('   Server version:', data.server_version);
      console.log('   Protocol version:', data.protocol_version);
      console.log('   Buffered messages:', data.buffered_messages_count);

      this.serverVersion = data.server_version;
      this.serverCapabilities = data.capabilities || [];
    }

    // Forward to event handlers
    this.emit('message', data);
  }

  async scheduleReconnect() {
    if (!this.configuration || !this.configuration.websocket.reconnect.enabled) {
      console.log('❌ Reconnection disabled');
      return;
    }

    this.reconnectAttempt++;

    if (this.reconnectAttempt > this.configuration.websocket.reconnect.max_attempts) {
      console.error('❌ Max reconnection attempts reached');
      this.emit('error', 'Max reconnection attempts reached');
      return;
    }

    this.setState('reconnecting', { attempt: this.reconnectAttempt });

    // Calculate backoff delay
    const baseDelay = this.configuration.websocket.reconnect.base_delay_ms;
    const maxDelay = this.configuration.websocket.reconnect.max_delay_ms;

    let delay = Math.min(
      baseDelay * Math.pow(2, this.reconnectAttempt - 1),
      maxDelay
    );

    // Add jitter
    if (this.configuration.websocket.reconnect.jitter) {
      const jitter = (Math.random() - 0.5) * 0.2 * delay;
      delay += jitter;
    }

    console.log(`🔄 Reconnecting in ${(delay / 1000).toFixed(1)}s (attempt ${this.reconnectAttempt}/${this.configuration.websocket.reconnect.max_attempts})`);

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  // MARK: - Helper Methods

  generateClientId() {
    return 'web-' + Math.random().toString(36).substring(2, 15);
  }

  detectBaseURL() {
    // Try to detect from current page
    if (typeof window !== 'undefined') {
      const protocol = window.location.protocol;
      const host = window.location.hostname;
      const port = '8010'; // Default backend port

      return `${protocol}//${host}:${port}`;
    }

    return 'http://localhost:8010';
  }

  setState(state, data = {}) {
    this.connectionState = state;
    this.emit('stateChange', { state, ...data });
  }

  emit(event, data) {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event].forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`❌ Event handler error for ${event}:`, error);
        }
      });
    }
  }

  flushMessageQueue() {
    if (this.messageQueue.length > 0) {
      console.log(`📤 Sending ${this.messageQueue.length} queued messages`);
      this.messageQueue.forEach(message => this.send(message));
      this.messageQueue = [];
    }
  }
}

export default UniversalWebSocketClient;
