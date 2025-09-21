/**
 * Wake Word Service for Frontend
 * ==============================
 * 
 * Handles wake word detection integration with the backend service.
 */

class WakeWordService {
  constructor() {
    this.websocket = null;
    this.isConnected = false;
    this.config = {
      enabled: true,
      wakeWords: ['hey jarvis', 'jarvis', 'ok jarvis'],
      sensitivity: 'medium',
      useVoiceResponse: true,
      showVisualIndicator: true
    };
    
    // Callbacks
    this.onWakeWordDetected = null;
    this.onStatusChange = null;
    this.onError = null;
    
    // State
    this.isListening = false;
    this.statistics = {
      detections: 0,
      falsePositives: 0,
      lastActivation: null
    };
  }
  
  /**
   * Initialize the wake word service
   */
  async initialize(apiUrl) {
    this.apiUrl = apiUrl;
    
    // Check if wake word is enabled on backend
    try {
      const response = await fetch(`${apiUrl}/api/wake-word/status`);
      if (response.ok) {
        const status = await response.json();
        this.config.enabled = status.enabled;
        this.statistics = status.statistics || this.statistics;
        
        if (this.config.enabled) {
          await this.connect();
        }
        
        return true;
      }
    } catch (error) {
      console.error('Failed to initialize wake word service:', error);
      return false;
    }
  }
  
  /**
   * Connect to wake word WebSocket
   */
  async connect() {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      return;
    }
    
    const wsUrl = this.apiUrl.replace('http://', 'ws://').replace('https://', 'wss://');
    
    try {
      this.websocket = new WebSocket(`${wsUrl}/api/wake-word/stream`);
      
      this.websocket.onopen = () => {
        console.log('✅ Wake word WebSocket connected');
        this.isConnected = true;
        this.isListening = true;
        
        if (this.onStatusChange) {
          this.onStatusChange({
            connected: true,
            listening: true,
            message: 'Wake word detection active'
          });
        }
        
        // Send ping to keep alive
        this.pingInterval = setInterval(() => {
          if (this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000);
      };
      
      this.websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse wake word message:', error);
        }
      };
      
      this.websocket.onerror = (error) => {
        console.error('Wake word WebSocket error:', error);
        if (this.onError) {
          this.onError({
            type: 'connection',
            message: 'Wake word connection error'
          });
        }
      };
      
      this.websocket.onclose = () => {
        console.log('Wake word WebSocket disconnected');
        this.isConnected = false;
        this.isListening = false;
        
        if (this.pingInterval) {
          clearInterval(this.pingInterval);
        }
        
        if (this.onStatusChange) {
          this.onStatusChange({
            connected: false,
            listening: false,
            message: 'Wake word detection inactive'
          });
        }
        
        // Attempt reconnection after 3 seconds
        setTimeout(() => {
          if (this.config.enabled) {
            this.connect();
          }
        }, 3000);
      };
      
    } catch (error) {
      console.error('Failed to connect wake word WebSocket:', error);
      if (this.onError) {
        this.onError({
          type: 'connection',
          message: 'Failed to connect to wake word service'
        });
      }
    }
  }
  
  /**
   * Handle WebSocket messages
   */
  handleMessage(data) {
    switch (data.type) {
      case 'wake_word_activated':
        console.log('🎤 Wake word detected:', data);
        this.statistics.detections++;
        this.statistics.lastActivation = new Date();
        
        // Call the callback with activation data
        if (this.onWakeWordDetected) {
          this.onWakeWordDetected({
            wakeWord: data.wake_word,
            confidence: data.confidence,
            response: data.response
          });
        }
        
        // Notify that JARVIS is listening for command
        this.websocket.send(JSON.stringify({
          type: 'command_received'
        }));
        break;
        
      case 'status':
        if (this.onStatusChange) {
          this.onStatusChange(data);
        }
        break;
        
      case 'error':
        if (this.onError) {
          this.onError({
            type: 'service',
            message: data.message
          });
        }
        break;
        
      case 'pong':
        // Keep-alive response
        break;
        
      default:
        console.log('Unknown wake word message type:', data.type);
    }
  }
  
  /**
   * Enable wake word detection
   */
  async enable() {
    try {
      const response = await fetch(`${this.apiUrl}/api/wake-word/enable`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const result = await response.json();
        this.config.enabled = true;
        await this.connect();
        return result;
      }
    } catch (error) {
      console.error('Failed to enable wake word:', error);
      throw error;
    }
  }
  
  /**
   * Disable wake word detection
   */
  async disable() {
    try {
      const response = await fetch(`${this.apiUrl}/api/wake-word/disable`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const result = await response.json();
        this.config.enabled = false;
        this.disconnect();
        return result;
      }
    } catch (error) {
      console.error('Failed to disable wake word:', error);
      throw error;
    }
  }
  
  /**
   * Update wake word settings
   */
  async updateSettings(settings) {
    try {
      const response = await fetch(`${this.apiUrl}/api/wake-word/settings`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Update local config
        if (settings.wake_words) {
          this.config.wakeWords = settings.wake_words;
        }
        if (settings.sensitivity) {
          this.config.sensitivity = settings.sensitivity;
        }
        if (settings.activation_responses) {
          this.config.activationResponses = settings.activation_responses;
        }
        
        return result;
      }
    } catch (error) {
      console.error('Failed to update wake word settings:', error);
      throw error;
    }
  }
  
  /**
   * Report false positive
   */
  async reportFalsePositive() {
    try {
      const response = await fetch(`${this.apiUrl}/api/wake-word/feedback/false-positive`, {
        method: 'POST'
      });
      
      if (response.ok) {
        this.statistics.falsePositives++;
        return await response.json();
      }
    } catch (error) {
      console.error('Failed to report false positive:', error);
    }
  }
  
  /**
   * Notify command complete
   */
  notifyCommandComplete() {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({
        type: 'command_complete'
      }));
    }
  }
  
  /**
   * Get current status
   */
  async getStatus() {
    try {
      const response = await fetch(`${this.apiUrl}/api/wake-word/status`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error('Failed to get wake word status:', error);
      return null;
    }
  }
  
  /**
   * Get statistics
   */
  async getStatistics() {
    try {
      const response = await fetch(`${this.apiUrl}/api/wake-word/statistics`);
      if (response.ok) {
        const stats = await response.json();
        this.statistics = { ...this.statistics, ...stats };
        return stats;
      }
    } catch (error) {
      console.error('Failed to get wake word statistics:', error);
      return this.statistics;
    }
  }
  
  /**
   * Test wake word activation
   */
  async testActivation() {
    try {
      const response = await fetch(`${this.apiUrl}/api/wake-word/test`, {
        method: 'POST'
      });
      
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error('Failed to test wake word:', error);
      throw error;
    }
  }
  
  /**
   * Disconnect WebSocket
   */
  disconnect() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }
  
  /**
   * Set callbacks
   */
  setCallbacks(callbacks) {
    if (callbacks.onWakeWordDetected) {
      this.onWakeWordDetected = callbacks.onWakeWordDetected;
    }
    if (callbacks.onStatusChange) {
      this.onStatusChange = callbacks.onStatusChange;
    }
    if (callbacks.onError) {
      this.onError = callbacks.onError;
    }
  }
}

export default WakeWordService;