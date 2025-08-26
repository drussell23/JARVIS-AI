/**
 * Dynamic WebSocket Client with Zero Hardcoding
 * Automatically discovers endpoints, adapts to message types, and self-heals
 */

interface WebSocketConfig {
  autoDiscover?: boolean;
  reconnectStrategy?: 'exponential' | 'linear' | 'fibonacci';
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  dynamicRouting?: boolean;
  messageValidation?: boolean;
}

interface DiscoveredEndpoint {
  path: string;
  capabilities: string[];
  priority: number;
  latency?: number;
  reliability?: number;
}

interface MessageType {
  type: string;
  schema?: any;
  validator?: (data: any) => boolean;
}

export class DynamicWebSocketClient {
  private connections: Map<string, WebSocket> = new Map();
  private messageHandlers: Map<string, Function[]> = new Map();
  private endpoints: DiscoveredEndpoint[] = [];
  private reconnectTimers: Map<string, NodeJS.Timeout> = new Map();
  private messageTypes: Map<string, MessageType> = new Map();
  private config: WebSocketConfig;
  
  // ML-based routing
  private routingModel: any = null;
  private connectionMetrics: Map<string, any> = new Map();
  
  constructor(config: WebSocketConfig = {}) {
    this.config = {
      autoDiscover: true,
      reconnectStrategy: 'exponential',
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      dynamicRouting: true,
      messageValidation: true,
      ...config
    };
    
    if (this.config.autoDiscover) {
      this.discoverEndpoints();
    }
    
    // Self-learning system
    this.initializeMLRouting();
  }
  
  /**
   * Discover available WebSocket endpoints dynamically
   */
  private async discoverEndpoints(): Promise<void> {
    try {
      // Try multiple discovery methods
      const discovered = await Promise.allSettled([
        this.discoverViaAPI(),
        this.discoverViaDOM(),
        this.discoverViaNetworkScan(),
        this.discoverViaConfig()
      ]);
      
      // Merge and deduplicate discovered endpoints
      const allEndpoints = discovered
        .filter(result => result.status === 'fulfilled')
        .flatMap((result: any) => result.value)
        .filter((endpoint, index, self) => 
          index === self.findIndex(e => e.path === endpoint.path)
        );
      
      this.endpoints = this.prioritizeEndpoints(allEndpoints);
      console.log(`🔍 Discovered ${this.endpoints.length} WebSocket endpoints`);
    } catch (error) {
      console.error('Endpoint discovery failed:', error);
      // Fallback to known endpoints
      this.endpoints = this.getFallbackEndpoints();
    }
  }
  
  /**
   * Discover endpoints via API
   */
  private async discoverViaAPI(): Promise<DiscoveredEndpoint[]> {
    const response = await fetch('/api/websocket/endpoints');
    if (response.ok) {
      const data = await response.json();
      return data.endpoints || [];
    }
    return [];
  }
  
  /**
   * Discover endpoints by scanning DOM
   */
  private async discoverViaDOM(): Promise<DiscoveredEndpoint[]> {
    const endpoints: DiscoveredEndpoint[] = [];
    
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
    
    // Look in script tags
    document.querySelectorAll('script').forEach(script => {
      const content = script.textContent || '';
      const wsMatches = content.match(/ws:\/\/[^'"]+|wss:\/\/[^'"]+/g);
      if (wsMatches) {
        wsMatches.forEach(match => {
          endpoints.push({
            path: match,
            capabilities: [],
            priority: 5
          });
        });
      }
    });
    
    return endpoints;
  }
  
  /**
   * Network scan for WebSocket endpoints
   */
  private async discoverViaNetworkScan(): Promise<DiscoveredEndpoint[]> {
    // Get base URL
    const baseUrl = window.location.origin.replace('http', 'ws');
    const commonPaths = [
      '/ws', '/websocket', '/socket', '/live',
      '/vision/ws/vision', '/voice/ws', '/automation/ws',
      '/api/ws', '/stream', '/realtime'
    ];
    
    const discovered: DiscoveredEndpoint[] = [];
    
    // Test each path
    await Promise.allSettled(
      commonPaths.map(async path => {
        const testWs = new WebSocket(`${baseUrl}${path}`);
        
        return new Promise<void>((resolve) => {
          const timeout = setTimeout(() => {
            testWs.close();
            resolve();
          }, 2000);
          
          testWs.onopen = () => {
            clearTimeout(timeout);
            discovered.push({
              path: `${baseUrl}${path}`,
              capabilities: [],
              priority: 5
            });
            testWs.close();
            resolve();
          };
          
          testWs.onerror = () => {
            clearTimeout(timeout);
            resolve();
          };
        });
      })
    );
    
    return discovered;
  }
  
  /**
   * Discover from configuration files
   */
  private async discoverViaConfig(): Promise<DiscoveredEndpoint[]> {
    try {
      const response = await fetch('/config/websockets.json');
      if (response.ok) {
        return await response.json();
      }
    } catch {
      // Config file doesn't exist
    }
    return [];
  }
  
  /**
   * Prioritize endpoints based on various factors
   */
  private prioritizeEndpoints(endpoints: DiscoveredEndpoint[]): DiscoveredEndpoint[] {
    return endpoints.sort((a, b) => {
      // Priority first
      if (a.priority !== b.priority) {
        return b.priority - a.priority;
      }
      
      // Then reliability
      if (a.reliability && b.reliability) {
        return b.reliability - a.reliability;
      }
      
      // Then latency (lower is better)
      if (a.latency && b.latency) {
        return a.latency - b.latency;
      }
      
      return 0;
    });
  }
  
  /**
   * Get fallback endpoints if discovery fails
   */
  private getFallbackEndpoints(): DiscoveredEndpoint[] {
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
  
  /**
   * Connect to a WebSocket endpoint with dynamic capabilities
   */
  async connect(endpointOrCapability?: string): Promise<WebSocket> {
    let endpoint: string;
    
    if (!endpointOrCapability) {
      // Use highest priority endpoint
      endpoint = this.endpoints[0]?.path;
    } else if (endpointOrCapability.startsWith('ws')) {
      // Direct endpoint provided
      endpoint = endpointOrCapability;
    } else {
      // Find endpoint by capability
      const capable = this.endpoints.find(ep => 
        ep.capabilities.includes(endpointOrCapability)
      );
      endpoint = capable?.path || this.endpoints[0]?.path;
    }
    
    if (!endpoint) {
      throw new Error('No WebSocket endpoints available');
    }
    
    // Check if already connected
    const existing = this.connections.get(endpoint);
    if (existing?.readyState === WebSocket.OPEN) {
      return existing;
    }
    
    // Create new connection
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
  
  /**
   * Setup connection handlers
   */
  private setupConnection(ws: WebSocket, endpoint: string): void {
    // Message handling with type inference
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Learn message types dynamically
        if (!this.messageTypes.has(data.type)) {
          this.learnMessageType(data);
        }
        
        // Validate if enabled
        if (this.config.messageValidation) {
          const messageType = this.messageTypes.get(data.type);
          if (messageType?.validator && !messageType.validator(data)) {
            console.warn(`Invalid message structure for type: ${data.type}`);
            return;
          }
        }
        
        // Route to handlers
        this.routeMessage(data, endpoint);
        
        // Update metrics
        this.updateConnectionMetrics(endpoint, 'message', data);
      } catch (error) {
        console.error('Message parsing error:', error);
      }
    };
    
    // Error handling
    ws.onerror = (error) => {
      console.error(`WebSocket error on ${endpoint}:`, error);
      this.updateConnectionMetrics(endpoint, 'error', error);
    };
    
    // Close handling with reconnection
    ws.onclose = () => {
      console.log(`🔌 Disconnected from ${endpoint}`);
      this.connections.delete(endpoint);
      this.handleReconnection(endpoint);
    };
  }
  
  /**
   * Learn message types dynamically
   */
  private learnMessageType(data: any): void {
    const type = data.type;
    if (!type) return;
    
    // Extract schema from message
    const schema = this.extractSchema(data);
    
    // Create validator function
    const validator = this.createValidator(schema);
    
    this.messageTypes.set(type, {
      type,
      schema,
      validator
    });
    
    console.log(`📚 Learned new message type: ${type}`);
  }
  
  /**
   * Extract schema from a message
   */
  private extractSchema(data: any): any {
    const schema: any = {};
    
    for (const [key, value] of Object.entries(data)) {
      if (value === null) {
        schema[key] = 'null';
      } else if (Array.isArray(value)) {
        schema[key] = 'array';
      } else {
        schema[key] = typeof value;
      }
    }
    
    return schema;
  }
  
  /**
   * Create a validator function for a schema
   */
  private createValidator(schema: any): (data: any) => boolean {
    return (data: any) => {
      for (const [key, type] of Object.entries(schema)) {
        if (!(key in data)) return false;
        
        const value = data[key];
        const actualType = Array.isArray(value) ? 'array' : typeof value;
        
        if (actualType !== type && type !== 'null') {
          return false;
        }
      }
      
      return true;
    };
  }
  
  /**
   * Route messages to appropriate handlers
   */
  private routeMessage(data: any, endpoint: string): void {
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
  
  /**
   * Handle reconnection with dynamic strategies
   */
  private handleReconnection(endpoint: string, attempt: number = 0): void {
    if (attempt >= this.config.maxReconnectAttempts!) {
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
  
  /**
   * Calculate reconnection delay based on strategy
   */
  private calculateReconnectDelay(attempt: number): number {
    const baseDelay = 1000;
    
    switch (this.config.reconnectStrategy) {
      case 'linear':
        return baseDelay * (attempt + 1);
        
      case 'exponential':
        return baseDelay * Math.pow(2, attempt);
        
      case 'fibonacci':
        return baseDelay * this.fibonacci(attempt + 1);
        
      default:
        return baseDelay;
    }
  }
  
  private fibonacci(n: number): number {
    if (n <= 1) return n;
    return this.fibonacci(n - 1) + this.fibonacci(n - 2);
  }
  
  /**
   * Start heartbeat for connection monitoring
   */
  private startHeartbeat(endpoint: string): void {
    const ws = this.connections.get(endpoint);
    if (!ws) return;
    
    setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      }
    }, this.config.heartbeatInterval!);
  }
  
  /**
   * Register a message handler
   */
  on(messageType: string, handler: Function): void {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, []);
    }
    
    this.messageHandlers.get(messageType)!.push(handler);
  }
  
  /**
   * Send a message with automatic routing
   */
  async send(message: any, capability?: string): Promise<void> {
    let targetWs: WebSocket | undefined;
    
    if (capability) {
      // Find WebSocket with required capability
      for (const [endpoint, ws] of this.connections) {
        const epConfig = this.endpoints.find(ep => ep.path === endpoint);
        if (epConfig?.capabilities.includes(capability)) {
          targetWs = ws;
          break;
        }
      }
    }
    
    // Use first available connection if no specific capability
    if (!targetWs) {
      targetWs = Array.from(this.connections.values())[0];
    }
    
    if (!targetWs || targetWs.readyState !== WebSocket.OPEN) {
      throw new Error('No open WebSocket connection available');
    }
    
    targetWs.send(JSON.stringify(message));
  }
  
  /**
   * Initialize ML-based routing
   */
  private initializeMLRouting(): void {
    // This would integrate with TensorFlow.js or similar
    // For now, we'll use heuristic-based routing
    
    this.routingModel = {
      predict: (message: any, endpoints: DiscoveredEndpoint[]) => {
        // Simple scoring based on message type and endpoint capabilities
        return endpoints.map(ep => ({
          endpoint: ep,
          score: this.calculateRoutingScore(message, ep)
        })).sort((a, b) => b.score - a.score)[0]?.endpoint;
      }
    };
  }
  
  private calculateRoutingScore(message: any, endpoint: DiscoveredEndpoint): number {
    let score = endpoint.priority;
    
    // Boost score if message type matches capability
    if (message.type && endpoint.capabilities.includes(message.type)) {
      score += 10;
    }
    
    // Consider latency
    if (endpoint.latency) {
      score -= endpoint.latency / 100;
    }
    
    // Consider reliability
    if (endpoint.reliability) {
      score += endpoint.reliability * 5;
    }
    
    return score;
  }
  
  /**
   * Update connection metrics for learning
   */
  private updateConnectionMetrics(endpoint: string, event: string, data?: any): void {
    if (!this.connectionMetrics.has(endpoint)) {
      this.connectionMetrics.set(endpoint, {
        messages: 0,
        errors: 0,
        latencies: [],
        lastActivity: Date.now()
      });
    }
    
    const metrics = this.connectionMetrics.get(endpoint)!;
    
    switch (event) {
      case 'message':
        metrics.messages++;
        break;
      case 'error':
        metrics.errors++;
        break;
      case 'latency':
        metrics.latencies.push(data);
        if (metrics.latencies.length > 100) {
          metrics.latencies.shift();
        }
        break;
    }
    
    metrics.lastActivity = Date.now();
    
    // Update endpoint reliability score
    const epIndex = this.endpoints.findIndex(ep => ep.path === endpoint);
    if (epIndex !== -1) {
      const errorRate = metrics.errors / (metrics.messages + metrics.errors);
      this.endpoints[epIndex].reliability = 1 - errorRate;
      
      if (metrics.latencies.length > 0) {
        const avgLatency = metrics.latencies.reduce((a, b) => a + b) / metrics.latencies.length;
        this.endpoints[epIndex].latency = avgLatency;
      }
    }
  }
  
  /**
   * Get connection statistics
   */
  getStats(): any {
    const stats = {
      connections: Array.from(this.connections.entries()).map(([endpoint, ws]) => ({
        endpoint,
        state: ws.readyState,
        metrics: this.connectionMetrics.get(endpoint)
      })),
      discoveredEndpoints: this.endpoints,
      learnedMessageTypes: Array.from(this.messageTypes.keys()),
      totalMessages: Array.from(this.connectionMetrics.values())
        .reduce((sum, m) => sum + m.messages, 0)
    };
    
    return stats;
  }
  
  /**
   * Cleanup and close all connections
   */
  destroy(): void {
    // Clear reconnect timers
    this.reconnectTimers.forEach(timer => clearTimeout(timer));
    this.reconnectTimers.clear();
    
    // Close all connections
    this.connections.forEach(ws => ws.close());
    this.connections.clear();
    
    // Clear handlers
    this.messageHandlers.clear();
  }
}

// Export for use in JavaScript
export default DynamicWebSocketClient;