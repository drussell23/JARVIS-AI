/**
 * Dynamic WebSocket Client - JavaScript Implementation
 * Zero-hardcoding WebSocket client that works with the TypeScript version
 */

class DynamicWebSocketClient {
  constructor(config = {}) {
    this.config = {
      autoDiscover: true,
      reconnectStrategy: 'exponential',
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      dynamicRouting: true,
      messageValidation: true,
      ...config
    };
    
    this.connections = new Map();
    this.messageHandlers = new Map();
    this.endpoints = [];
    this.reconnectTimers = new Map();
    this.messageTypes = new Map();
    this.connectionMetrics = new Map();
    
    if (this.config.autoDiscover) {
      this.discoverEndpoints();
    }
  }
  
  async discoverEndpoints() {
    console.log('🔍 Discovering WebSocket endpoints...');
    
    try {
      const discovered = await Promise.allSettled([
        this.discoverViaAPI(),
        this.discoverViaDOM(),
        this.discoverViaNetworkScan()
      ]);
      
      const allEndpoints = discovered
        .filter(result => result.status === 'fulfilled')
        .flatMap(result => result.value)
        .filter((endpoint, index, self) => 
          index === self.findIndex(e => e.path === endpoint.path)
        );
      
      this.endpoints = this.prioritizeEndpoints(allEndpoints);
      console.log(`✅ Discovered ${this.endpoints.length} WebSocket endpoints:`, this.endpoints);
      
    } catch (error) {
      console.error('❌ Endpoint discovery failed:', error);
      this.endpoints = this.getFallbackEndpoints();
    }
  }
  
  async discoverViaAPI() {
    try {
      const response = await fetch('/api/websocket/endpoints');
      if (response.ok) {
        const data = await response.json();
        return data.endpoints || [];
      }
    } catch (error) {
      console.debug('API discovery failed:', error);
    }
    return [];
  }
  
  async discoverViaDOM() {
    const endpoints = [];
    
    // Look for data attributes
    document.querySelectorAll('[data-websocket]').forEach(element => {
      const path = element.getAttribute('data-websocket');
      if (path) {
        endpoints.push({
          path,
          capabilities: (element.getAttribute('data-capabilities') || '').split(','),
          priority: parseInt(element.getAttribute('data-priority') || '5')
        });
      }
    });
    
    return endpoints;
  }
  
  async discoverViaNetworkScan() {
    const baseUrl = window.location.origin.replace('http', 'ws');
    const commonPaths = [
      '/ws', '/websocket', '/socket',
      '/vision/ws/vision', '/voice/ws', '/automation/ws',
      '/api/ws', '/stream'
    ];
    
    const discovered = [];
    const testPromises = commonPaths.map(path => 
      this.testWebSocketPath(`${baseUrl}${path}`)
        .then(result => {
          if (result.success) {
            discovered.push({
              path: result.path,
              capabilities: this.inferCapabilities(path),
              priority: 5,
              latency: result.latency
            });
          }
        })
    );
    
    await Promise.allSettled(testPromises);
    return discovered;
  }
  
  async testWebSocketPath(path) {
    return new Promise(resolve => {
      const startTime = Date.now();
      const testWs = new WebSocket(path);
      const timeout = setTimeout(() => {
        testWs.close();
        resolve({ success: false, path });
      }, 2000);
      
      testWs.onopen = () => {
        const latency = Date.now() - startTime;
        clearTimeout(timeout);
        testWs.close();
        resolve({ success: true, path, latency });
      };
      
      testWs.onerror = () => {
        clearTimeout(timeout);
        resolve({ success: false, path });
      };
    });
  }
  
  inferCapabilities(path) {
    const capabilities = [];
    
    if (path.includes('vision')) capabilities.push('vision', 'monitoring', 'analysis');
    if (path.includes('voice')) capabilities.push('voice', 'speech', 'audio');
    if (path.includes('automation')) capabilities.push('automation', 'tasks', 'scheduling');
    if (capabilities.length === 0) capabilities.push('general');
    
    return capabilities;
  }
  
  prioritizeEndpoints(endpoints) {
    return endpoints.sort((a, b) => {
      if (a.priority !== b.priority) return b.priority - a.priority;
      if (a.reliability && b.reliability) return b.reliability - a.reliability;
      if (a.latency && b.latency) return a.latency - b.latency;
      return 0;
    });
  }
  
  getFallbackEndpoints() {
    const baseUrl = window.location.origin.replace('http', 'ws');
    return [
      {
        path: `${baseUrl}/vision/ws/vision`,
        capabilities: ['vision', 'monitoring', 'analysis'],
        priority: 10
      },
      {
        path: `${baseUrl}/ws`,
        capabilities: ['general'],
        priority: 5
      }
    ];
  }
  
  async connect(endpointOrCapability) {
    let endpoint;
    
    if (!endpointOrCapability) {
      endpoint = this.endpoints[0]?.path;
    } else if (endpointOrCapability.startsWith('ws')) {
      endpoint = endpointOrCapability;
    } else {
      const capable = this.endpoints.find(ep => 
        ep.capabilities.includes(endpointOrCapability)
      );
      endpoint = capable?.path || this.endpoints[0]?.path;
    }
    
    if (!endpoint) {
      throw new Error('No WebSocket endpoints available');
    }
    
    // Check existing connection
    const existing = this.connections.get(endpoint);
    if (existing?.readyState === WebSocket.OPEN) {
      return existing;
    }
    
    // Create new connection
    console.log(`🔌 Connecting to ${endpoint}...`);
    const ws = new WebSocket(endpoint);
    this.setupConnection(ws, endpoint);
    
    return new Promise((resolve, reject) => {
      ws.onopen = () => {
        console.log(`✅ Connected to ${endpoint}`);
        this.connections.set(endpoint, ws);
        this.startHeartbeat(endpoint);
        resolve(ws);
      };
      
      ws.onerror = (error) => {
        console.error(`❌ Connection failed to ${endpoint}:`, error);
        reject(error);
      };
    });
  }
  
  setupConnection(ws, endpoint) {
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log(`📨 Message from ${endpoint}:`, data);
        
        // Learn message type
        if (!this.messageTypes.has(data.type)) {
          this.learnMessageType(data);
        }
        
        // Route to handlers
        this.routeMessage(data, endpoint);
        
        // Update metrics
        this.updateMetrics(endpoint, 'message');
      } catch (error) {
        console.error('Message parsing error:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error(`WebSocket error on ${endpoint}:`, error);
      this.updateMetrics(endpoint, 'error');
    };
    
    ws.onclose = () => {
      console.log(`🔌 Disconnected from ${endpoint}`);
      this.connections.delete(endpoint);
      this.handleReconnection(endpoint);
    };
  }
  
  learnMessageType(data) {
    const type = data.type;
    if (!type) return;
    
    const schema = {};
    for (const [key, value] of Object.entries(data)) {
      if (value === null) schema[key] = 'null';
      else if (Array.isArray(value)) schema[key] = 'array';
      else schema[key] = typeof value;
    }
    
    this.messageTypes.set(type, { type, schema });
    console.log(`📚 Learned message type: ${type}`, schema);
  }
  
  routeMessage(data, endpoint) {
    const handlers = this.messageHandlers.get(data.type) || [];
    const globalHandlers = this.messageHandlers.get('*') || [];
    
    [...handlers, ...globalHandlers].forEach(handler => {
      try {
        handler(data, endpoint);
      } catch (error) {
        console.error('Handler error:', error);
      }
    });
  }
  
  handleReconnection(endpoint, attempt = 0) {
    if (attempt >= this.config.maxReconnectAttempts) {
      console.error(`Max reconnection attempts reached for ${endpoint}`);
      return;
    }
    
    const delay = this.calculateReconnectDelay(attempt);
    console.log(`🔄 Reconnecting to ${endpoint} in ${delay}ms (attempt ${attempt + 1})`);
    
    const timer = setTimeout(async () => {
      try {
        await this.connect(endpoint);
        this.reconnectTimers.delete(endpoint);
      } catch (error) {
        this.handleReconnection(endpoint, attempt + 1);
      }
    }, delay);
    
    this.reconnectTimers.set(endpoint, timer);
  }
  
  calculateReconnectDelay(attempt) {
    const baseDelay = 1000;
    
    switch (this.config.reconnectStrategy) {
      case 'linear':
        return baseDelay * (attempt + 1);
      case 'exponential':
        return baseDelay * Math.pow(2, attempt);
      default:
        return baseDelay;
    }
  }
  
  startHeartbeat(endpoint) {
    const ws = this.connections.get(endpoint);
    if (!ws) return;
    
    const interval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ 
          type: 'ping', 
          timestamp: Date.now() 
        }));
      } else {
        clearInterval(interval);
      }
    }, this.config.heartbeatInterval);
  }
  
  updateMetrics(endpoint, event) {
    if (!this.connectionMetrics.has(endpoint)) {
      this.connectionMetrics.set(endpoint, {
        messages: 0,
        errors: 0,
        lastActivity: Date.now()
      });
    }
    
    const metrics = this.connectionMetrics.get(endpoint);
    if (event === 'message') metrics.messages++;
    if (event === 'error') metrics.errors++;
    metrics.lastActivity = Date.now();
    
    // Update endpoint reliability
    const epIndex = this.endpoints.findIndex(ep => ep.path === endpoint);
    if (epIndex !== -1) {
      const errorRate = metrics.errors / (metrics.messages + metrics.errors || 1);
      this.endpoints[epIndex].reliability = 1 - errorRate;
    }
  }
  
  on(messageType, handler) {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, []);
    }
    this.messageHandlers.get(messageType).push(handler);
  }
  
  async send(message, capability) {
    let targetWs;
    
    if (capability) {
      for (const [endpoint, ws] of this.connections) {
        const epConfig = this.endpoints.find(ep => ep.path === endpoint);
        if (epConfig?.capabilities.includes(capability)) {
          targetWs = ws;
          break;
        }
      }
    }
    
    if (!targetWs) {
      targetWs = Array.from(this.connections.values())[0];
    }
    
    if (!targetWs || targetWs.readyState !== WebSocket.OPEN) {
      // Try to connect first
      await this.connect(capability);
      targetWs = Array.from(this.connections.values())[0];
    }
    
    targetWs.send(JSON.stringify(message));
  }
  
  getStats() {
    return {
      connections: Array.from(this.connections.entries()).map(([endpoint, ws]) => ({
        endpoint,
        state: ws.readyState,
        stateText: ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'][ws.readyState],
        metrics: this.connectionMetrics.get(endpoint)
      })),
      discoveredEndpoints: this.endpoints,
      learnedMessageTypes: Array.from(this.messageTypes.keys()),
      totalMessages: Array.from(this.connectionMetrics.values())
        .reduce((sum, m) => sum + m.messages, 0)
    };
  }
  
  destroy() {
    this.reconnectTimers.forEach(timer => clearTimeout(timer));
    this.reconnectTimers.clear();
    this.connections.forEach(ws => ws.close());
    this.connections.clear();
    this.messageHandlers.clear();
  }
}

// Create global instance for easy access
window.DynamicWebSocketClient = DynamicWebSocketClient;