/**
 * Python HTTP Bridge for WebSocket Router
 * Provides HTTP-based communication with Python backend
 */

import axios, { AxiosInstance } from 'axios';
import { EventEmitter } from 'events';

interface PythonHTTPBridgeConfig {
  pythonBackendUrl: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  headers?: Record<string, string>;
}

interface PythonFunctionCall {
  module: string;
  function: string;
  args?: any[];
  kwargs?: Record<string, any>;
  context?: Record<string, any>;
}

interface PythonResponse {
  success: boolean;
  result?: any;
  error?: string;
  traceback?: string;
}

export class PythonHTTPBridge extends EventEmitter {
  private axiosInstance: AxiosInstance;
  private config: Required<PythonHTTPBridgeConfig>;
  private correlationMap: Map<string, (response: any) => void> = new Map();
  
  constructor(config: PythonHTTPBridgeConfig) {
    super();
    
    this.config = {
      pythonBackendUrl: config.pythonBackendUrl,
      timeout: config.timeout || 30000,
      retryAttempts: config.retryAttempts || 3,
      retryDelay: config.retryDelay || 1000,
      headers: config.headers || {}
    };
    
    // Create axios instance
    this.axiosInstance = axios.create({
      baseURL: this.config.pythonBackendUrl,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...this.config.headers
      }
    });
    
    // Setup interceptors
    this.setupInterceptors();
  }
  
  private setupInterceptors(): void {
    // Request interceptor
    this.axiosInstance.interceptors.request.use(
      (config) => {
        // Add correlation ID to headers
        config.headers['X-Correlation-ID'] = this.generateCorrelationId();
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );
    
    // Response interceptor with retry logic
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      async (error) => {
        const config = error.config;
        
        if (!config || !config.retry) {
          config.retry = 0;
        }
        
        if (config.retry < this.config.retryAttempts) {
          config.retry++;
          
          // Exponential backoff
          const delay = this.config.retryDelay * Math.pow(2, config.retry - 1);
          await new Promise(resolve => setTimeout(resolve, delay));
          
          console.log(`Retrying request (attempt ${config.retry}/${this.config.retryAttempts})`);
          return this.axiosInstance.request(config);
        }
        
        return Promise.reject(error);
      }
    );
  }
  
  /**
   * Call a Python function via HTTP
   */
  async callPythonFunction(
    module: string,
    functionName: string,
    args: any[] = [],
    kwargs: any = {}
  ): Promise<any> {
    const endpoint = this.getEndpointForModule(module);
    
    const payload: PythonFunctionCall = {
      module,
      function: functionName,
      args,
      kwargs,
      context: kwargs.context || {}
    };
    
    try {
      const response = await this.axiosInstance.post<PythonResponse>(
        endpoint,
        payload
      );
      
      if (response.data.success) {
        return response.data.result;
      } else {
        throw new Error(response.data.error || 'Unknown error occurred');
      }
    } catch (error: any) {
      if (error.response) {
        // Server responded with error
        console.error('Python backend error:', error.response.data);
        throw new Error(`Python backend error: ${error.response.data.error || error.message}`);
      } else if (error.request) {
        // Request made but no response
        console.error('No response from Python backend');
        throw new Error('Python backend not responding');
      } else {
        // Something else happened
        console.error('Request error:', error.message);
        throw error;
      }
    }
  }
  
  /**
   * Get endpoint for module
   */
  private getEndpointForModule(module: string): string {
    // Map module names to endpoints
    const moduleEndpoints: Record<string, string> = {
      'backend.api.unified_vision_handler': '/ws/vision/handler',
      'backend.api.general_websocket_handler': '/ws/general/handler',
      'backend.api.voice_websocket_handler': '/ws/voice/handler',
      'backend.vision.vision_system_v2': '/api/vision/execute',
      'backend.autonomy.action_executor': '/api/autonomy/execute',
      'backend.voice.jarvis_agent_voice': '/api/voice/execute'
    };
    
    // Check for exact match
    if (moduleEndpoints[module]) {
      return moduleEndpoints[module];
    }
    
    // Check for partial matches
    for (const [key, endpoint] of Object.entries(moduleEndpoints)) {
      if (module.includes(key) || key.includes(module)) {
        return endpoint;
      }
    }
    
    // Default endpoint
    return '/api/websocket/handler';
  }
  
  /**
   * Send WebSocket message to Python handler
   */
  async handleWebSocketMessage(
    route: string,
    message: any,
    context: any = {}
  ): Promise<any> {
    // Determine module based on route
    const module = this.getModuleForRoute(route);
    
    return this.callPythonFunction(
      module,
      'handle_websocket_message',
      [message],
      { context }
    );
  }
  
  /**
   * Get module for route
   */
  private getModuleForRoute(route: string): string {
    const routeModules: Record<string, string> = {
      '/ws/vision': 'backend.api.unified_vision_handler',
      '/ws/voice': 'backend.api.voice_websocket_handler',
      '/ws': 'backend.api.general_websocket_handler',
      '/ws/ml/audio': 'backend.api.ml_audio_handler',
      '/ws/jarvis': 'backend.api.jarvis_voice_api'
    };
    
    return routeModules[route] || 'backend.api.general_websocket_handler';
  }
  
  /**
   * Check if Python backend is healthy
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await this.axiosInstance.get('/health');
      return response.status === 200;
    } catch (error) {
      return false;
    }
  }
  
  /**
   * Discover available WebSocket endpoints
   */
  async discoverEndpoints(): Promise<any> {
    try {
      const response = await this.axiosInstance.get('/api/websocket/endpoints');
      return response.data;
    } catch (error) {
      console.error('Failed to discover endpoints:', error);
      return [];
    }
  }
  
  /**
   * Execute batch operations
   */
  async executeBatch(operations: PythonFunctionCall[]): Promise<any[]> {
    try {
      const response = await this.axiosInstance.post('/api/batch', {
        operations
      });
      
      return response.data.results;
    } catch (error) {
      console.error('Batch execution failed:', error);
      throw error;
    }
  }
  
  /**
   * Stream data to Python
   */
  async streamData(
    module: string,
    functionName: string,
    dataStream: AsyncIterable<any>
  ): Promise<any> {
    const streamId = this.generateCorrelationId();
    const chunks: any[] = [];
    
    try {
      // Send start stream signal
      await this.callPythonFunction(
        module,
        `${functionName}_start`,
        [streamId],
        {}
      );
      
      // Send chunks
      for await (const chunk of dataStream) {
        chunks.push(chunk);
        await this.callPythonFunction(
          module,
          `${functionName}_chunk`,
          [streamId, chunk],
          {}
        );
      }
      
      // Send end stream signal
      const result = await this.callPythonFunction(
        module,
        `${functionName}_end`,
        [streamId],
        {}
      );
      
      return result;
    } catch (error) {
      // Send abort signal on error
      await this.callPythonFunction(
        module,
        `${functionName}_abort`,
        [streamId],
        {}
      ).catch(() => {}); // Ignore abort errors
      
      throw error;
    }
  }
  
  /**
   * Generate unique correlation ID
   */
  private generateCorrelationId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  /**
   * Get statistics
   */
  getStats(): any {
    return {
      pythonBackendUrl: this.config.pythonBackendUrl,
      pendingRequests: this.correlationMap.size,
      timeout: this.config.timeout,
      retryAttempts: this.config.retryAttempts
    };
  }
  
  /**
   * Cleanup
   */
  destroy(): void {
    this.correlationMap.clear();
    this.removeAllListeners();
  }
}

// Export for use
export default PythonHTTPBridge;