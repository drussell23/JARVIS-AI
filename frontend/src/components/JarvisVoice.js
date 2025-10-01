import React, { useState, useEffect, useRef } from 'react';
import './JarvisVoice.css';
import '../styles/JarvisVoiceError.css';
import MicrophonePermissionHelper from './MicrophonePermissionHelper';
import MicrophoneIndicator from './MicrophoneIndicator';
import WorkflowProgress from './WorkflowProgress'; // Workflow progress component
import mlAudioHandler from '../utils/MLAudioHandler'; // ML-enhanced audio handling
import { getNetworkRecoveryManager } from '../utils/NetworkRecoveryManager'; // Advanced network recovery
import WakeWordService from './WakeWordService'; // Wake word detection service
import configService from '../services/DynamicConfigService'; // Dynamic configuration service

// Inline styles to ensure button visibility
const buttonVisibilityStyle = `
  .jarvis-button {
    display: inline-block !important;
    visibility: visible !important;
    opacity: 1 !important;
  }
  .voice-controls {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
  }
`;

// Dynamic API configuration - will be set after config service is ready
let API_URL = null;
let WS_URL = null;
let configReady = false;

// Create promise to wait for config
const configPromise = new Promise((resolve) => {
  // Check if config is already ready
  const currentApiUrl = configService.getApiUrl();
  if (currentApiUrl) {
    API_URL = currentApiUrl;
    WS_URL = configService.getWebSocketUrl();
    configReady = true;
    console.log('JarvisVoice: Config already ready', { API_URL, WS_URL });
    resolve();
    return;
  }

  // Initialize API URLs when config is ready
  const handleConfigReady = (config) => {
    API_URL = config.API_BASE_URL || configService.getApiUrl() || 'http://localhost:8010';
    WS_URL = config.WS_BASE_URL || configService.getWebSocketUrl() || API_URL.replace('http://', 'ws://').replace('https://', 'wss://');
    configReady = true;
    console.log('JarvisVoice: Config ready', { API_URL, WS_URL });
    resolve();
  };

  configService.once('config-ready', handleConfigReady);
  
  // Check again in case we missed the event
  setTimeout(() => {
    if (!configReady) {
      const currentApiUrl = configService.getApiUrl();
      if (currentApiUrl) {
        handleConfigReady(configService.config);
      }
    }
  }, 100);
  
  // Final fallback after 1 second
  setTimeout(() => {
    if (!configReady) {
      console.log('JarvisVoice: Using fallback config after timeout');
      handleConfigReady({
        API_BASE_URL: 'http://localhost:8010',
        WS_BASE_URL: 'ws://localhost:8010'
      });
    }
  }, 1000);
});

// Also listen for config updates
configService.on('config-updated', (config) => {
  const newApiUrl = config.API_BASE_URL || configService.getApiUrl();
  const newWsUrl = config.WS_BASE_URL || configService.getWebSocketUrl();
  
  if (newApiUrl !== API_URL || newWsUrl !== WS_URL) {
    API_URL = newApiUrl;
    WS_URL = newWsUrl;
    console.log('JarvisVoice: Config updated', { API_URL, WS_URL });
  }
});

// VisionConnection class for real-time workspace monitoring
class VisionConnection {
  constructor(onUpdate, onAction) {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 2000;
    this.workspaceData = null;
    this.actionQueue = [];

    // Callbacks
    this.onWorkspaceUpdate = onUpdate || (() => { });
    this.onActionExecuted = onAction || (() => { });

    // Monitoring state
    this.monitoringActive = false;
    this.updateInterval = 2.0;
  }

  async connect() {
    try {
      console.log('🔌 Connecting to Vision WebSocket...');

      // Wait for config if not ready
      if (!WS_URL) {
        console.log('VisionConnection: Waiting for config...');
        await configPromise;
      }
      
      // Use main backend port for vision WebSocket
      const wsBaseUrl = WS_URL || configService.getWebSocketUrl() || 'ws://localhost:8010';
      const wsUrl = `${wsBaseUrl}/vision/ws`;  // Use consistent WebSocket URL
      console.log('VisionConnection: Connecting to', wsUrl);
      this.socket = new WebSocket(wsUrl);

      this.socket.onopen = () => {
        console.log('✅ Vision WebSocket connected!');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.monitoringActive = true;

        // Request initial analysis
        this.requestWorkspaceAnalysis();
      };

      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleVisionMessage(data);
        } catch (error) {
          console.error('Error parsing vision message:', error);
        }
      };

      this.socket.onerror = (error) => {
        console.error('❌ Vision WebSocket error:', error);
      };

      this.socket.onclose = () => {
        console.log('🔌 Vision WebSocket disconnected');
        this.isConnected = false;
        this.monitoringActive = false;
        this.attemptReconnect();
      };

    } catch (error) {
      console.error('Failed to connect to Vision WebSocket:', error);
      this.attemptReconnect();
    }
  }

  handleVisionMessage(data) {
    console.log('👁️ Vision Update:', data.type);

    switch (data.type) {
      case 'initial_state':
        this.handleInitialState(data);
        break;

      case 'workspace_update':
        this.handleWorkspaceUpdate(data);
        break;

      case 'workspace_analysis':
        this.handleWorkspaceAnalysis(data);
        break;

      case 'action_result':
        this.handleActionResult(data);
        break;

      case 'config_updated':
        console.log('⚙️ Config updated:', data);
        this.updateInterval = data.update_interval;
        break;

      default:
        console.log('Unknown vision message type:', data.type);
    }
  }

  handleInitialState(data) {
    console.log('📊 Initial workspace state:', data.workspace);
    this.workspaceData = data.workspace;
    this.monitoringActive = data.monitoring_active;
    this.updateInterval = data.update_interval;

    this.onWorkspaceUpdate({
      type: 'initial',
      workspace: data.workspace,
      timestamp: data.timestamp
    });
  }

  handleWorkspaceUpdate(data) {
    // Check if workspace data exists
    if (!data || !data.workspace) {
      console.warn('Workspace update missing workspace data:', data);
      return;
    }

    console.log(`🔄 Workspace update: ${data.workspace.window_count || 0} windows`);

    this.workspaceData = data.workspace;

    // Process autonomous actions
    if (data.autonomous_actions && data.autonomous_actions.length > 0) {
      this.processAutonomousActions(data.autonomous_actions);
    }

    // Check for important notifications
    if (data.workspace.notification_details) {
      const details = data.workspace.notification_details;
      const totalNotifs = details.badges + details.messages + details.meetings + details.alerts;

      if (totalNotifs > 0) {
        console.log(`📬 Notifications: ${details.badges} badges, ${details.messages} messages, ${details.meetings} meetings, ${details.alerts} alerts`);
      }
    }

    // Notify UI
    this.onWorkspaceUpdate({
      type: 'update',
      workspace: data.workspace,
      autonomousActions: data.autonomous_actions,
      enhancedData: data.enhanced_data,
      queueStatus: data.queue_status,
      timestamp: data.timestamp
    });
  }

  handleWorkspaceAnalysis(data) {
    console.log('🔍 Workspace analysis received:', data.analysis);

    this.onWorkspaceUpdate({
      type: 'analysis',
      analysis: data.analysis,
      timestamp: data.timestamp
    });
  }

  handleActionResult(data) {
    console.log('⚡ Action result:', data);

    this.onActionExecuted({
      success: data.success,
      action: data.action,
      message: data.message
    });
  }

  processAutonomousActions(actions) {
    // Filter actions that don't require permission
    const autoActions = actions.filter(a => !a.requires_permission && a.confidence > 0.8);

    // Add to action queue
    this.actionQueue = [...this.actionQueue, ...autoActions];

    // Process queue
    this.processActionQueue();

    // Notify about actions requiring permission
    const permissionRequired = actions.filter(a => a.requires_permission);
    if (permissionRequired.length > 0) {
      console.log(`🔐 ${permissionRequired.length} actions require permission`);
      // Here you would show permission UI
    }
  }

  async processActionQueue() {
    if (this.actionQueue.length === 0) return;

    const action = this.actionQueue.shift();
    console.log(`🤖 Executing autonomous action: ${action.type}`);

    // Send action execution request
    this.executeAction(action);

    // Process next action after delay
    setTimeout(() => this.processActionQueue(), 1000);
  }

  requestWorkspaceAnalysis() {
    if (this.isConnected && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({
        type: 'request_analysis'
      }));
    }
  }

  setUpdateInterval(interval) {
    if (this.isConnected && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({
        type: 'set_interval',
        interval: interval
      }));
    }
  }

  executeAction(action) {
    if (this.isConnected && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({
        type: 'execute_action',
        action: action
      }));
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`🔄 Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

      setTimeout(() => {
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('❌ Max reconnection attempts reached. Vision system offline.');
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
      this.monitoringActive = false;
    }
  }

  getWorkspaceData() {
    return this.workspaceData;
  }

  isMonitoring() {
    return this.isConnected && this.monitoringActive;
  }

  async startMonitoring() {
    if (this.isConnected && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({
        type: 'start_monitoring'
      }));
      this.monitoringActive = true;
    }
  }

  async stopMonitoring() {
    if (this.isConnected && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({
        type: 'stop_monitoring'
      }));
      this.monitoringActive = false;
    }
  }
}

const JarvisVoice = () => {
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [jarvisStatus, setJarvisStatus] = useState('offline');
  const [systemMode, setSystemMode] = useState('unknown'); // 'minimal' or 'full'
  const [showUpgradeSuccess, setShowUpgradeSuccess] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [error, setError] = useState(null);
  const [continuousListening, setContinuousListening] = useState(false);
  const [isWaitingForCommand, setIsWaitingForCommand] = useState(false);
  const [isJarvisSpeaking, setIsJarvisSpeaking] = useState(false);
  const [microphonePermission, setMicrophonePermission] = useState('checking');
  const [visionConnected, setVisionConnected] = useState(false);
  const [workspaceData, setWorkspaceData] = useState(null);
  const [autonomousMode, setAutonomousMode] = useState(false);
  const [micStatus, setMicStatus] = useState('unknown');
  const [networkRetries, setNetworkRetries] = useState(0);
  const [maxNetworkRetries] = useState(3);
  const [workflowProgress, setWorkflowProgress] = useState(null);

  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioWebSocketRef = useRef(null);
  const offlineModeRef = useRef(false);
  const commandQueueRef = useRef([]);
  const proxyEndpointRef = useRef(null);
  const recognitionRef = useRef(null);
  const visionConnectionRef = useRef(null);
  const lastSpeechTimeRef = useRef(0);
  const wakeWordServiceRef = useRef(null);
  const continuousListeningRef = useRef(false);
  const isWaitingForCommandRef = useRef(false);

  // API URLs are defined globally at the top of the file
  // Ensure consistent WebSocket URL (fix port mismatch)
  const JARVIS_WS_URL = WS_URL;  // Use same base URL as API

  useEffect(() => {
    // Preload voices to ensure Daniel is available
    if ('speechSynthesis' in window) {
      // Force load voices
      window.speechSynthesis.getVoices();
      
      // Listen for voices to be loaded
      window.speechSynthesis.onvoiceschanged = () => {
        const voices = window.speechSynthesis.getVoices();
        const danielVoice = voices.find(v => v.name.includes('Daniel'));
        if (danielVoice) {
          console.log('✅ Daniel voice preloaded:', danielVoice.name);
        }
      };
    }

    // Auto-activate JARVIS on mount for seamless wake word experience
    const autoActivate = async () => {
      // Wait for config to be ready before making any API calls
      await configPromise;
      console.log('JarvisVoice: Config ready, initializing...');
      
      await checkJarvisStatus();
      await checkMicrophonePermission();
      await initializeWakeWordService();
    };

    autoActivate();

    // Predict potential audio issues - disabled to prevent CORS errors
    // mlAudioHandler.predictAudioIssue();

    // Inject style to ensure button visibility
    const styleElement = document.createElement('style');
    styleElement.textContent = buttonVisibilityStyle;
    document.head.appendChild(styleElement);

    // Button visibility checker - disabled since we now auto-activate
    const checkButtonsInterval = null; // Commented out button injection
    // const checkButtonsInterval = setInterval(() => {
    //   const container = document.querySelector('.jarvis-voice-container');
    //   if (!container) {
    //     console.log('Button checker: Container not found yet');
    //     return;
    //   }

    //   // Look for control buttons
    //   const hasButtons = container.querySelector('.jarvis-button');
    //   // Only log when buttons are missing

    //   if (!hasButtons) {
    //     console.warn('JARVIS buttons missing - injecting emergency button');
    //     // Find the voice-controls div
    //     const voiceControls = container.querySelector('.voice-controls');
    //     if (voiceControls && !document.getElementById('jarvis-emergency-button')) {
    //       // Inject a fallback activate button
    //       const buttonDiv = document.createElement('div');
    //       buttonDiv.id = 'jarvis-emergency-button';
    //       buttonDiv.style.cssText = 'text-align: center; margin-top: 10px;';
    //       buttonDiv.innerHTML = `
    //         <button 
    //           class="jarvis-button activate" 
    //           style="
    //             display: inline-block;
    //             padding: 12px 24px;
    //             font-size: 16px;
    //             background: linear-gradient(135deg, #ff6b35 0%, #ff4500 100%);
    //             color: white;
    //             border: none;
    //             border-radius: 8px;
    //             cursor: pointer;
    //             margin: 5px;
    //             transition: all 0.3s;
    //           "
    //           onclick="
    //             console.log('Emergency activate button clicked');
    //             // Try to call the activate function
    //             const event = new CustomEvent('jarvis-emergency-activate');
    //             window.dispatchEvent(event);
    //           "
    //         >
    //           🚀 Activate JARVIS (Emergency)
    //         </button>
    //         <div style="margin-top: 5px; font-size: 12px; color: #888;">
    //           If buttons are missing, click here to activate JARVIS
    //         </div>
    //       `;
    //       voiceControls.appendChild(buttonDiv);
    //       console.log('Injected emergency JARVIS button');
    //     }
    //   }
    // }, 1000);

    // Set up ML event listeners
    const handleAudioPrediction = (event) => {
      const { prediction, suggestedAction } = event.detail;
      if (prediction.probability > 0.7) {
        console.warn('High probability of audio issue:', prediction);
        // Take proactive action
        if (suggestedAction === 'preemptive_permission_check') {
          checkMicrophonePermission();
        }
      }
    };

    const handleAudioAnomaly = (event) => {
      console.warn('Audio anomaly detected:', event.detail);
      setError('Audio anomaly detected. System is adapting...');
    };

    const handleAudioMetrics = (event) => {
      console.log('Audio metrics update:', event.detail);
    };

    const handleTextFallback = (event) => {
      console.log('Enabling text fallback mode');
      // Focus on text input
      const textInput = document.querySelector('.voice-input input');
      if (textInput) {
        textInput.focus();
        textInput.placeholder = 'Voice unavailable - type your command here...';
      }
    };

    // Add ML event listeners
    window.addEventListener('audioIssuePredicted', handleAudioPrediction);
    window.addEventListener('audioAnomaly', handleAudioAnomaly);
    window.addEventListener('audioMetricsUpdate', handleAudioMetrics);
    window.addEventListener('enableTextFallback', handleTextFallback);

    // Add emergency activate listener
    const handleEmergencyActivate = () => {
      console.log('Emergency activate event received');
      activateJarvis();
    };
    window.addEventListener('jarvis-emergency-activate', handleEmergencyActivate);


    // Initialize Vision Connection
    if (!visionConnectionRef.current) {
      visionConnectionRef.current = new VisionConnection(
        // Workspace update callback
        (data) => {
          setWorkspaceData(data);
          setVisionConnected(true);

          // Process workspace updates
          if (data.type === 'update' && data.workspace) {
            // Check for important notifications
            if (data.workspace.notifications && data.workspace.notifications.length > 0) {
              const notification = data.workspace.notifications[0];
              speakResponse(`I've detected: ${notification}`);
            }

            // Handle autonomous actions
            if (data.autonomousActions && data.autonomousActions.length > 0 && autonomousMode) {
              const highPriorityActions = data.autonomousActions.filter(a =>
                a.priority === 'HIGH' || a.priority === 'CRITICAL'
              );

              if (highPriorityActions.length > 0) {
                const action = highPriorityActions[0];
                speakResponse(`I'm going to ${action.type.replace(/_/g, ' ')} for ${action.target}`);
              }
            }

            // Check queue status
            if (data.queueStatus && data.queueStatus.queue_length > 0) {
              console.log(`📋 Action Queue: ${data.queueStatus.queue_length} actions pending`);
            }
          }
        },
        // Action executed callback
        (result) => {
          console.log('Action executed:', result);
          if (!result.success) {
            speakResponse(`I encountered an issue: ${result.message}`);
          }
        }
      );
    }

    return () => {
      // Clean up WebSocket
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
      // Stop speech recognition
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      // Disconnect vision
      if (visionConnectionRef.current) {
        visionConnectionRef.current.disconnect();
      }
      // Cleanup wake word service
      if (wakeWordServiceRef.current && typeof wakeWordServiceRef.current.disconnect === 'function') {
        wakeWordServiceRef.current.disconnect();
      }
      // Remove ML event listeners
      window.removeEventListener('audioIssuePredicted', handleAudioPrediction);
      window.removeEventListener('audioAnomaly', handleAudioAnomaly);
      window.removeEventListener('audioMetricsUpdate', handleAudioMetrics);
      window.removeEventListener('enableTextFallback', handleTextFallback);

      // Remove emergency activate listener
      window.removeEventListener('jarvis-emergency-activate', handleEmergencyActivate);

      // Clear button checker interval
      // if (checkButtonsInterval) {
      //   clearInterval(checkButtonsInterval);
      // }

      // Remove injected style
      if (styleElement && styleElement.parentNode) {
        styleElement.parentNode.removeChild(styleElement);
      }
    };
  }, [autonomousMode]);

  // Separate effect for auto-activation
  useEffect(() => {
    if (jarvisStatus === 'offline' || jarvisStatus === null) {
      const timer = setTimeout(() => {
        console.log('Auto-activating JARVIS for seamless experience');
        activateJarvis();
      }, 2000);
      return () => clearTimeout(timer);
    } else if (jarvisStatus === 'online' && !continuousListening) {
      // Enable wake word detection when JARVIS comes online
      console.log('JARVIS online, enabling wake word detection');
      // Small delay to ensure speech recognition is initialized
      setTimeout(() => {
        if (recognitionRef.current) {
          enableContinuousListening();
        } else {
          console.log('Speech recognition not initialized yet, retrying...');
          initializeSpeechRecognition();
          setTimeout(() => enableContinuousListening(), 1000);
        }
      }, 1000);
    }
  }, [jarvisStatus, continuousListening]);

  const checkJarvisStatus = async () => {
    // Ensure config is ready
    if (!configReady || !API_URL) {
      console.log('JarvisVoice: Waiting for config before checking status...');
      await configPromise;
    }
    
    try {
      const apiUrl = API_URL || configService.getApiUrl() || 'http://localhost:8010';
      console.log('JarvisVoice: Checking JARVIS status at:', apiUrl);
      const response = await fetch(`${apiUrl}/voice/jarvis/status`);
      const data = await response.json();
      
      // Enhanced logging for mode detection
      const previousMode = systemMode;
      
      if (data.mode === 'minimal') {
        console.log('🔄 JARVIS Status: Running in MINIMAL MODE');
        console.log('  ⏳ This is temporary while full system initializes');
        console.log('  📊 Available features:', {
          voice: data.components?.voice || false,
          vision: data.components?.vision || false,
          memory: data.components?.memory || false,
          tools: data.components?.tools || false,
          rust: data.components?.rust || false
        });
        if (data.upgrader) {
          console.log('  🚀 Upgrade Progress:', {
            monitoring: data.upgrader.monitoring,
            attempts: `${data.upgrader.attempts}/${data.upgrader.max_attempts}`,
            status: 'Waiting for components to initialize...'
          });
        }
        console.log('  ✅ Basic voice commands are available');
        console.log('  ⚠️  Advanced features (wake word, ML audio) temporarily unavailable');
        setSystemMode('minimal');
      } else {
        // Full mode detected!
        if (previousMode === 'minimal') {
          console.log('🎉 JARVIS UPGRADED TO FULL MODE! 🎉');
          console.log('  ✅ All features now available:');
          console.log('    • Wake word detection ("Hey JARVIS")');
          console.log('    • ML-powered audio processing');
          console.log('    • Vision system active');
          console.log('    • Memory system online');
          console.log('    • Advanced tools enabled');
          console.log('  🚀 System running at full capacity!');
          
          // Show success message to user
          setResponse('System upgraded! All features are now available.');
          speakResponse('System upgraded. All features are now available, Sir.');
          
          // Show upgrade success banner
          setShowUpgradeSuccess(true);
          setTimeout(() => setShowUpgradeSuccess(false), 10000); // Hide after 10 seconds
        } else {
          console.log('✅ JARVIS Status: Running in FULL MODE');
          console.log('  🚀 All systems operational');
        }
        setSystemMode('full');
      }

      // Map backend status to frontend status
      const status = data.status || 'offline';
      if (status === 'standby' || status === 'ready' || status === 'available') {
        setJarvisStatus('online'); // Show as online when in standby, ready, or available
      } else {
        setJarvisStatus(status);
      }

      // Connect WebSocket if JARVIS is available (including standby, ready, active, and available)
      if (data.status === 'online' || data.status === 'standby' || data.status === 'active' || data.status === 'ready' || data.status === 'available') {
        if (data.mode === 'minimal') {
          console.log('📡 Minimal mode: WebSocket features limited');
        } else {
          console.log('JARVIS is available, connecting WebSocket...');
        }
        setTimeout(() => {
          connectWebSocket();
        }, 500);
      } else {
        console.log('JARVIS is offline, not connecting WebSocket');
      }
    } catch (err) {
      console.error('Failed to check JARVIS status:', err);
      console.log('Setting status to offline due to error');
      setJarvisStatus('offline');
    }
  };

  const checkMicrophonePermission = async () => {
    try {
      // Check if browser supports speech recognition
      if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        setMicrophonePermission('unsupported');
        setError('Speech recognition not supported in this browser');
        return;
      }

      // Try to get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());

      setMicrophonePermission('granted');
      setMicStatus('ready');
      // Initialize speech recognition after permission granted
      initializeSpeechRecognition();
    } catch (error) {
      console.error('Microphone permission error:', error);
      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        setMicrophonePermission('denied');
        setMicStatus('error');
        setError('Microphone access denied. Please grant permission to use JARVIS.');
      } else if (error.name === 'NotFoundError') {
        setMicrophonePermission('no-device');
        setMicStatus('error');
        setError('No microphone found. Please connect a microphone.');
      } else {
        setMicrophonePermission('error');
        setMicStatus('error');
        setError('Error accessing microphone: ' + error.message);
      }
    }
  };

  const connectWebSocket = async () => {
    // Don't connect if already connected or connecting
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      console.log('WebSocket already connected or connecting');
      return;
    }
    
    // Ensure config is ready
    if (!configReady || !WS_URL) {
      console.log('JarvisVoice: Waiting for config before WebSocket connection...');
      await configPromise;
    }

    try {
      const wsBaseUrl = WS_URL || configService.getWebSocketUrl() || 'ws://localhost:8010';
      const wsUrl = `${wsBaseUrl}/voice/jarvis/stream`;
      console.log('Connecting to WebSocket:', wsUrl);
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('Connected to JARVIS WebSocket');
        setError(null);
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Only show error if not connecting
        if (wsRef.current.readyState !== WebSocket.CONNECTING) {
          setError('Connection error');
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected');
        // Only reconnect if it was a clean close and component is still mounted
        if (event.wasClean && jarvisStatus !== 'offline') {
          setTimeout(() => {
            connectWebSocket();
          }, 3000);
        }
      };
    } catch (err) {
      console.error('Failed to connect WebSocket:', err);
      setError('Failed to connect to JARVIS');
    }
  };

  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'connected':
        setJarvisStatus('online');
        break;
      case 'processing':
        setIsProcessing(true);
        // Don't cancel speech here - it might cancel the wake word response
        break;
      case 'voice_unlock':
        // Handle voice unlock responses
        console.log('Voice unlock response received:', data);
        const voiceUnlockText = data.message || data.text || 'Voice unlock command processed';
        setResponse(voiceUnlockText);
        setIsProcessing(false);
        
        // Speak the response if speak flag is true
        if ((data.message || data.text) && data.speak !== false) {
          console.log('[JARVIS Audio] Speaking voice unlock response:', voiceUnlockText);
          // Don't update display again since we just set it above
          speakResponse(voiceUnlockText, false);
        }
        
        // Reset waiting state after voice unlock command
        if (isWaitingForCommandRef.current) {
          setTimeout(() => {
            setIsWaitingForCommand(false);
            isWaitingForCommandRef.current = false;
            // Ensure continuous listening remains active
            if (!continuousListeningRef.current && jarvisStatus === 'online') {
              console.log('Re-enabling continuous listening after voice unlock command');
              enableContinuousListening();
            }
          }, 1000);
        }
        break;
      case 'response':
        console.log('WebSocket response received:', data);
        
        // Check if this is an error response we should ignore
        const errorText = (data.text || '').toLowerCase();
        if (errorText.includes("don't have a handler for query commands")) {
          console.log('Ignoring query handler error, continuing to listen...');
          setIsProcessing(false);
          // Don't reset waiting state - keep listening
          return;
        }
        
        // Use the EXACT same text for both display and speech
        const responseText = data.text || data.message || 'Response received';
        console.log('[JARVIS Audio] Setting display text:', responseText);
        setResponse(responseText);
        setIsProcessing(false);

        // Use speech synthesis with Daniel voice
        if (responseText && data.speak !== false) {
          // Speak the EXACT same text that we just displayed
          console.log('[JARVIS Audio] Speaking exact text:', responseText);
          speakResponse(responseText, false);
        }
        
        // Reset waiting state after successful command
        if (isWaitingForCommandRef.current && responseText && !responseText.toLowerCase().includes('error')) {
          setTimeout(() => {
            setIsWaitingForCommand(false);
            isWaitingForCommandRef.current = false;
            // Ensure continuous listening remains active
            if (!continuousListeningRef.current && jarvisStatus === 'online') {
              console.log('Re-enabling continuous listening after command completion');
              enableContinuousListening();
            }
          }, 1000);
        }

        // Check for autonomy activation commands in response
        const responseTextLower = responseText.toLowerCase();
        if (data.command_type === 'autonomy_activation' ||
          responseTextLower.includes('autonomous mode activated') ||
          responseTextLower.includes('full autonomy enabled') ||
          responseTextLower.includes('all systems online')) {
          // Activate autonomous mode
          if (!autonomousMode) {
            setAutonomousMode(true);
            // Connect vision system
            if (visionConnectionRef.current && !visionConnectionRef.current.isConnected) {
              visionConnectionRef.current.connect();
            }
            // Enable continuous listening
            enableContinuousListening();
          }
        }
        break;
      case 'autonomy_status':
        // Handle autonomy status updates
        if (data.enabled) {
          setAutonomousMode(true);
          if (visionConnectionRef.current && !visionConnectionRef.current.isConnected) {
            visionConnectionRef.current.connect();
          }
        } else {
          setAutonomousMode(false);
        }
        break;
      case 'vision_status':
        // Handle vision connection status
        setVisionConnected(data.connected);
        break;
      case 'mode_changed':
        // Handle mode change confirmations
        if (data.mode === 'autonomous') {
          setAutonomousMode(true);
        } else {
          setAutonomousMode(false);
        }
        break;
      case 'error':
        setError(data.message);
        setIsProcessing(false);
        break;
      case 'debug_log':
        // Display debug logs in console with styling
        const logStyle = data.level === 'error'
          ? 'color: red; font-weight: bold;'
          : data.level === 'warning'
            ? 'color: orange;'
            : 'color: #4CAF50; font-weight: bold;';
        console.log(`%c[JARVIS DEBUG ${new Date(data.timestamp).toLocaleTimeString()}] ${data.message}`, logStyle);
        if (data.level === 'error') {
          console.error('Full error details:', data);
        }
        break;
      case 'workflow_analysis':
        // Workflow has been analyzed and is about to start
        console.log('🔄 Workflow analysis:', data);
        setWorkflowProgress({
          ...data.workflow,
          status: 'starting'
        });
        break;
      case 'workflow_started':
        // Workflow execution has started
        setWorkflowProgress(prev => ({
          ...prev,
          ...data,
          status: 'running'
        }));
        break;
      case 'action_started':
        // Individual action has started
        setWorkflowProgress(prev => {
          if (!prev || !prev.actions) return prev;
          const updatedActions = [...prev.actions];
          if (updatedActions[data.action_index]) {
            updatedActions[data.action_index].status = 'running';
          }
          return {
            ...prev,
            actions: updatedActions,
            currentAction: data.action_index
          };
        });
        break;
      case 'action_completed':
        // Individual action completed
        setWorkflowProgress(prev => {
          if (!prev || !prev.actions) return prev;
          const updatedActions = [...prev.actions];
          if (updatedActions[data.action_index]) {
            updatedActions[data.action_index].status = 'completed';
            updatedActions[data.action_index].duration = data.duration;
          }
          return {
            ...prev,
            actions: updatedActions
          };
        });
        break;
      case 'action_failed':
        // Individual action failed
        setWorkflowProgress(prev => {
          if (!prev || !prev.actions) return prev;
          const updatedActions = [...prev.actions];
          if (updatedActions[data.action_index]) {
            updatedActions[data.action_index].status = 'failed';
            updatedActions[data.action_index].error = data.error;
            updatedActions[data.action_index].duration = data.duration;
          }
          return {
            ...prev,
            actions: updatedActions
          };
        });
        break;
      case 'action_retry':
        // Action is being retried
        setWorkflowProgress(prev => {
          if (!prev || !prev.actions) return prev;
          const updatedActions = [...prev.actions];
          if (updatedActions[data.action_index]) {
            updatedActions[data.action_index].status = 'retry';
            updatedActions[data.action_index].retry_count = data.retry_count;
          }
          return {
            ...prev,
            actions: updatedActions
          };
        });
        break;
      case 'workflow_completed':
        // Workflow execution completed
        setWorkflowProgress(prev => ({
          ...prev,
          status: 'completed',
          total_duration: data.total_duration,
          success_rate: data.success_rate
        }));
        // Clear workflow progress after 10 seconds
        setTimeout(() => setWorkflowProgress(null), 10000);
        break;
      case 'narration':
        // Handle document writing narration (display only, no voice)
        console.log('📝 Document narration:', data.message);
        if (data.message) {
          setResponse(data.message);
          // Don't speak narration messages to prevent overlap
        }
        break;
      case 'voice_narration':
        // Handle voice narration from document writer
        console.log('🎤 Voice narration received:', data.message);
        console.log('🎤 Speak flag:', data.speak);
        if (data.message && data.speak !== false) {
          // Store the exact message
          const narrationText = data.message;
          console.log('[JARVIS Audio] Narration text:', narrationText);

          // For narrations, WAIT to update display until voice actually starts
          // This ensures perfect synchronization
          speakResponseWithCallback(narrationText, () => {
            // Update display when voice actually starts
            setResponse(narrationText);
          });
        } else {
          console.log('🎤 Skipping narration - speak flag is false or no message');
        }
        break;
      default:
        break;
    }
  };


  // Function to setup recognition handlers
  const setupRecognitionHandlers = (recognition) => {
    if (!recognition) return;

    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    // Move all the existing handlers to be assigned here
    recognition.onresult = (event) => {
      // Existing onresult handler code will be here
      handleSpeechResult(event);
    };

    recognition.onerror = async (event) => {
      // Existing onerror handler code will be here
      handleSpeechError(event);
    };

    recognition.onend = () => {
      // Existing onend handler code will be here
      handleSpeechEnd();
    };

    recognition.onstart = () => {
      console.log('Speech recognition started');
      setError('');
      setMicStatus('ready');
    };

    return recognition;
  };

  const handleSpeechResult = (event) => {
    // This will contain the existing onresult handler logic
    // Moving it here for better organization
  };

  const handleSpeechError = async (event) => {
    // This will contain the existing onerror handler logic
    // Moving it here for better organization
  };

  const handleSpeechEnd = () => {
    // This will contain the existing onend handler logic
    // Moving it here for better organization
  };

  const initializeWakeWordService = async () => {
    // Ensure config is ready
    if (!configReady || !API_URL) {
      console.log('JarvisVoice: Waiting for config before wake word init...');
      await configPromise;
    }
    
    if (!wakeWordServiceRef.current) {
      // Create a simplified wake word handler object
      wakeWordServiceRef.current = {
        onWakeWordDetected: (data) => {
          console.log('🎤 Wake word activated!', data);

          // Clear any previous transcript
          setTranscript('');

          // Set listening state
          setIsWaitingForCommand(true);
          setIsListening(true);

          // Play the response
          const responseText = data.response || "Ready for your command, sir";
          speakResponse(responseText);

          // Start timeout for command (30 seconds - more time to speak)
          setTimeout(() => {
            if (isWaitingForCommand && !isJarvisSpeaking) {
              setIsWaitingForCommand(false);
              console.log('⏱️ Command timeout - returning to wake word listening');
            }
          }, 30000);
        },
        isActive: false
      };

      // Try to connect to backend wake word service if available
      try {
        const wakeService = new WakeWordService();
        const apiUrl = API_URL || configService.getApiUrl() || 'http://localhost:8010';
        console.log('JarvisVoice: Initializing wake word service at:', apiUrl);
        const initialized = await wakeService.initialize(apiUrl);
        if (initialized) {
          console.log('✅ Backend wake word service connected');
          // Use backend service callbacks if available
          wakeService.setCallbacks({
            onWakeWordDetected: wakeWordServiceRef.current.onWakeWordDetected,
            onStatusChange: (status) => console.log('Wake word status:', status),
            onError: (error) => console.error('Wake word error:', error)
          });
        }
      } catch (e) {
        console.log('📢 Using frontend-only wake word detection');
      }

      wakeWordServiceRef.current.isActive = true;
    }
  };

  const initializeSpeechRecognition = () => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();

      // Track if JARVIS is speaking to avoid self-triggering
      let jarvisSpeaking = false;

      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';
      recognitionRef.current.maxAlternatives = 1;

      // Increase timeouts to be more patient
      // Note: These are non-standard but work in some browsers
      if ('speechTimeout' in recognitionRef.current) {
        recognitionRef.current.speechTimeout = 999999999; // Effectively infinite
      }
      if ('noSpeechTimeout' in recognitionRef.current) {
        recognitionRef.current.noSpeechTimeout = 999999999; // Effectively infinite
      }

      recognitionRef.current.onresult = (event) => {
        const last = event.results.length - 1;
        const transcript = event.results[last][0].transcript.toLowerCase();
        const isFinal = event.results[last].isFinal;

        // Debug logging
        console.log(`🎙️ Speech detected: "${transcript}" (final: ${isFinal}) | Waiting: ${isWaitingForCommandRef.current} | Continuous: ${continuousListeningRef.current}`);

        // Process both interim and final results when waiting for command
        if (!isFinal && !isWaitingForCommandRef.current) return;

        // Check for wake words when not waiting for command
        if (!isWaitingForCommandRef.current && continuousListeningRef.current) {
          const wakeWords = ['hey jarvis', 'jarvis', 'ok jarvis', 'hello jarvis'];
          const detectedWakeWord = wakeWords.find(word => transcript.includes(word));

          if (detectedWakeWord) {
            console.log('🎯 Wake word detected:', detectedWakeWord, '| Current state:', {
              isWaitingForCommand: isWaitingForCommandRef.current,
              continuousListening: continuousListeningRef.current,
              isListening
            });

            // Check if there's a command after the wake word in the same sentence
            const fullTranscript = event.results[last][0].transcript;
            let commandAfterWakeWord = fullTranscript.toLowerCase();
            
            // Remove the wake word to get just the command
            const wakeWordIndex = commandAfterWakeWord.indexOf(detectedWakeWord);
            if (wakeWordIndex !== -1) {
              commandAfterWakeWord = commandAfterWakeWord.substring(wakeWordIndex + detectedWakeWord.length).trim();
            }

            // If there's a command after the wake word, process it directly
            if (commandAfterWakeWord.length > 0) {
              console.log('🎯 Command found after wake word:', commandAfterWakeWord);
              // Process the command immediately
              handleVoiceCommand(commandAfterWakeWord);
              return;
            }

            // Otherwise, just activate wake word mode
            setTranscript('');
            console.log('🚀 Triggering wake word handler for listening mode...');
            handleWakeWordDetected();
            return;
          }
        }

        // When waiting for command after wake word, process any speech
        if (isWaitingForCommandRef.current && transcript.length > 0) {
          // Only process final results for commands
          if (!isFinal) return;
          
          // Filter out wake words from commands
          const wakeWords = ['hey jarvis', 'jarvis', 'ok jarvis', 'hello jarvis'];
          let commandText = event.results[last][0].transcript;

          // Remove wake word if it's at the beginning of the command
          wakeWords.forEach(word => {
            if (commandText.toLowerCase().startsWith(word)) {
              commandText = commandText.substring(word.length).trim();
            }
          });

          // Only process if there's actual command content
          if (commandText.length > 0) {
            console.log('📢 Processing command:', commandText);
            console.log('🚀 Sending command to backend via WebSocket');
            handleVoiceCommand(commandText);

            // Reset waiting state
            setIsWaitingForCommand(false);
            isWaitingForCommandRef.current = false;
          } else {
            console.log('⚠️ No command text after removing wake word, continuing to listen...');
          }
        }
      };

      recognitionRef.current.onerror = async (event) => {
        // Handle "no-speech" errors quietly - they're expected
        if (event.error !== 'no-speech') {
          console.error('Speech recognition error:', event.error, event);
        }

        // Use ML-enhanced error handling
        let mlResult = null;
        try {
          mlResult = await mlAudioHandler.handleAudioError(event, recognitionRef.current);
        } catch (error) {
          console.warn('ML audio handler error:', error);
        }

        if (mlResult && mlResult.success) {
          // Only log recovery for non-no-speech errors
          if (event.error !== 'no-speech') {
            console.log('ML audio recovery successful:', mlResult);
          }
          setError('');
          setMicStatus('ready');

          // Restart recognition if needed
          if (mlResult && (mlResult.newContext || (mlResult.message && mlResult.message.includes('granted')))) {
            startListening();
          }
        } else {
          // Fallback to basic error handling
          switch (event.error) {
            case 'audio-capture':
              setError('🎤 Microphone access denied. ML recovery failed.');
              setMicStatus('error');
              break;

            case 'not-allowed':
              setError('🚫 Microphone permission denied. Please enable in browser settings.');
              setMicStatus('error');
              break;

            case 'no-speech':
              // Enhanced indefinite listening - ALWAYS restart
              console.log('No speech detected, enforcing indefinite listening...');
              if (continuousListening) {
                // Don't show error to user for expected silence
                setError('');

                // Immediately restart without delay
                try {
                  recognitionRef.current.stop();
                  // Restart immediately
                  setTimeout(() => {
                    try {
                      recognitionRef.current.start();
                      console.log('✅ Microphone restarted successfully after silence');
                    } catch (e) {
                      // If already started, that's fine
                      if (e.message && !e.message.includes('already started')) {
                        console.log('Restart attempt:', e.message);
                      }
                    }
                  }, 50); // Minimal delay
                } catch (e) {
                  console.log('Stopping for restart:', e);
                }
              }
              break;

            case 'network':
              console.log('🌐 Network error detected, initiating advanced recovery...');
              setError('🌐 Network error. Initiating advanced recovery...');

              // Use advanced network recovery manager
              const networkRecoveryManager = getNetworkRecoveryManager();

              (async () => {
                try {
                  const recoveryResult = await networkRecoveryManager.recoverFromNetworkError(
                    event,
                    recognitionRef.current,
                    {
                      continuousListening,
                      isListening,
                      mlAudioHandler,
                      jarvisStatus,
                      wsRef: wsRef.current
                    }
                  );

                  if (recoveryResult.success) {
                    console.log('✅ Network recovery successful:', recoveryResult);
                    setError('');
                    setNetworkRetries(0);

                    // Handle different recovery types
                    if (recoveryResult.newRecognition) {
                      // Service switched, update reference
                      recognitionRef.current = recoveryResult.newRecognition;
                      setupRecognitionHandlers(recoveryResult.newRecognition);
                    } else if (recoveryResult.useWebSocket) {
                      // Switch to WebSocket mode
                      setError('📡 Switched to WebSocket audio streaming');
                      // Store WebSocket reference for audio streaming
                      audioWebSocketRef.current = recoveryResult.websocket;
                      setTimeout(() => setError(''), 3000);
                    } else if (recoveryResult.offlineMode) {
                      // Enable offline mode
                      setError('📴 Offline mode active - commands will sync when online');
                      offlineModeRef.current = true;
                      commandQueueRef.current = recoveryResult.commandQueue;
                    } else if (recoveryResult.useProxy) {
                      // Use ML backend proxy
                      setError('🤖 Using ML backend for speech processing');
                      proxyEndpointRef.current = recoveryResult.proxyEndpoint;
                      setTimeout(() => setError(''), 3000);
                    }
                  } else {
                    // All strategies failed
                    setError('🌐 Network recovery failed. Manual intervention required.');
                    console.error('All network recovery strategies exhausted');

                    // Show recovery tips
                    setTimeout(() => {
                      setError(
                        '💡 Try:\n' +
                        '1. Check internet connection\n' +
                        '2. Disable VPN/Proxy\n' +
                        '3. Clear browser cache\n' +
                        '4. Restart browser'
                      );
                    }, 3000);
                  }
                } catch (recoveryError) {
                  console.error('Recovery manager error:', recoveryError);
                  setError('🌐 Network recovery system error');
                }
              })();
              break;

            case 'aborted':
              // Don't show error for aborted (usually intentional)
              console.log('Recognition aborted');
              break;

            default:
              setError(`Speech recognition error: ${event.error}`);
          }
        }
      };

      recognitionRef.current.onend = () => {
        console.log('Speech recognition ended - enforcing indefinite listening');

        // ALWAYS restart if continuous listening is enabled (check ref for most current state)
        if (continuousListeningRef.current || continuousListening) {
          console.log('♾️ Indefinite listening active - restarting microphone...');

          // Track restart attempts
          let restartAttempt = 0;
          const maxAttempts = 10;

          const attemptRestart = () => {
            restartAttempt++;

            try {
              recognitionRef.current.start();
              console.log(`✅ Microphone restarted successfully (attempt ${restartAttempt})`);
              setError(''); // Clear any errors
              setMicStatus('ready');

              // Reset speech timestamp
              lastSpeechTimeRef.current = Date.now();
            } catch (e) {
              if (e.message && e.message.includes('already started')) {
                console.log('Microphone already active');
                return;
              }

              console.log(`Restart attempt ${restartAttempt}/${maxAttempts} failed:`, e.message);

              // Keep trying with exponential backoff
              if (restartAttempt < maxAttempts && continuousListening) {
                const delay = Math.min(50 * Math.pow(2, restartAttempt), 2000);
                console.log(`Retrying in ${delay}ms...`);
                setTimeout(attemptRestart, delay);
              } else {
                console.error('Failed to restart microphone after max attempts');
                setError('Microphone restart failed - click to retry');
                setMicStatus('error');
              }
            }
          };

          // Start restart attempts immediately
          setTimeout(attemptRestart, 50);
        } else {
          setIsListening(false);
          setIsWaitingForCommand(false);
          console.log('Continuous listening disabled - microphone stopped');
        }
      };
    } else {
      setError('Speech recognition not supported in this browser');
    }
  };

  const handleWakeWordDetected = () => {
    console.log('🎯 handleWakeWordDetected called!');
    setIsWaitingForCommand(true);
    isWaitingForCommandRef.current = true;
    setIsListening(true);
    
    // Always speak response immediately
    speakResponse("Yes, sir?");

    // Don't send anything to backend - we're handling the wake word response locally
    // Just ensure WebSocket is connected for subsequent commands
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.log('WebSocket not connected, attempting to connect...');
      connectWebSocket();
    }

    // Timeout for command after 30 seconds (longer for conversation)
    setTimeout(() => {
      setIsWaitingForCommand((currentWaiting) => {
        if (currentWaiting) {
          console.log('⏱️ Command timeout - stopping listening');
          setIsListening(false);
          isWaitingForCommandRef.current = false;
          return false;
        }
        return currentWaiting;
      });
    }, 30000);
  };

  const handleVoiceCommand = (command) => {
    console.log('🎯 handleVoiceCommand called with:', command);
    console.log('📡 WebSocket state:', wsRef.current ? wsRef.current.readyState : 'No WebSocket');
    setTranscript(command);

    // Check for autonomy activation commands
    const lowerCommand = command.toLowerCase();
    if (lowerCommand.includes('activate full autonomy') ||
      lowerCommand.includes('enable autonomous mode') ||
      lowerCommand.includes('activate autonomy') ||
      lowerCommand.includes('iron man mode') ||
      lowerCommand.includes('activate all systems')) {
      // Direct autonomy activation
      toggleAutonomousMode();
      return;
    }

    // Send via WebSocket instead of REST API
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'command',
        text: command,
        mode: autonomousMode ? 'autonomous' : 'manual'
      }));
      setResponse('Processing...');
    } else {
      // Fallback to REST API if WebSocket not connected
      sendTextCommand(command);
    }

    // Don't immediately reset waiting state - let the response handler do it
    // This ensures we don't miss the response
    console.log('Command sent, waiting for response...');
  };

  const activateJarvis = async () => {
    // Ensure config is ready
    if (!configReady || !API_URL) {
      console.log('JarvisVoice: Waiting for config before activating...');
      await configPromise;
    }
    
    try {
      const apiUrl = API_URL || configService.getApiUrl() || 'http://localhost:8010';
      console.log('JarvisVoice: Activating JARVIS at:', apiUrl);
      const response = await fetch(`${apiUrl}/voice/jarvis/activate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('JarvisVoice: Activation response:', data);
      setJarvisStatus('activating');
      setTimeout(async () => {
        setJarvisStatus('online');

        // Initialize wake word service if not already done
        if (!wakeWordServiceRef.current) {
          await initializeWakeWordService();
        }

        // Speak activation confirmation  
        speakResponse("JARVIS online. Say 'Hey JARVIS' to begin.");

        // Enable continuous listening for wake word detection
        console.log('🎙️ Enabling continuous listening for wake word...');
        enableContinuousListening();
      }, 2000);
    } catch (err) {
      console.error('Failed to activate JARVIS:', err);
      setError('Failed to activate JARVIS');
    }
  };

  const toggleAutonomousMode = async () => {
    const newMode = !autonomousMode;
    setAutonomousMode(newMode);

    if (newMode) {
      // Enable autonomous mode
      speakResponse("Initiating full autonomy. All systems coming online. Vision system activating. AI brain engaged. Sir, I am now fully autonomous.");

      // Connect vision system
      if (visionConnectionRef.current) {
        console.log('Connecting vision system...');
        await visionConnectionRef.current.connect();
        // Start monitoring immediately
        if (visionConnectionRef.current.isConnected) {
          visionConnectionRef.current.startMonitoring();
        }
      }

      // Enable continuous listening
      enableContinuousListening();

      // Notify backend about autonomy mode
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'set_mode',
          mode: 'autonomous'
        }));
      }
    } else {
      // Disable autonomous mode
      speakResponse("Disabling autonomous mode. Returning to manual control. Standing by for your commands, sir.");

      // Stop vision monitoring
      if (visionConnectionRef.current && visionConnectionRef.current.isConnected) {
        visionConnectionRef.current.stopMonitoring();
        visionConnectionRef.current.disconnect();
      }
      setVisionConnected(false);

      // Notify backend
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'set_mode',
          mode: 'manual'
        }));
      }

      // Keep listening if user wants
    }
  };

  const enableContinuousListening = () => {
    if (recognitionRef.current) {
      setContinuousListening(true);
      continuousListeningRef.current = true;
      setIsListening(true);

      // Configure for INDEFINITE continuous listening
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;

      // Override any browser timeouts
      if ('speechTimeout' in recognitionRef.current) {
        recognitionRef.current.speechTimeout = 999999999;
      }
      if ('noSpeechTimeout' in recognitionRef.current) {
        recognitionRef.current.noSpeechTimeout = 999999999;
      }

      try {
        recognitionRef.current.start();
        console.log('♾️ INDEFINITE listening enabled - microphone will NEVER turn off automatically');

        // Set up keep-alive mechanism
        const keepAliveInterval = setInterval(() => {
          if (!continuousListening) {
            clearInterval(keepAliveInterval);
            return;
          }

          // Check if recognition is still active
          console.log('🔄 Keep-alive check - ensuring microphone stays active');

          // If no speech for a while, send a dummy event to keep it alive
          const timeSinceLastSpeech = Date.now() - lastSpeechTimeRef.current;
          if (timeSinceLastSpeech > 30000) { // 30 seconds
            console.log('⚡ Triggering keep-alive pulse');
            lastSpeechTimeRef.current = Date.now();

            // Force a restart if needed
            if (!isListening) {
              console.log('🔄 Keep-alive: Restarting stopped recognition');
              try {
                recognitionRef.current.stop();
                setTimeout(() => {
                  recognitionRef.current.start();
                  setIsListening(true);
                }, 100);
              } catch (e) {
                console.log('Keep-alive restart:', e.message);
              }
            }
          }
        }, 5000); // Check every 5 seconds

        // Store interval reference for cleanup
        recognitionRef.current._keepAliveInterval = keepAliveInterval;

        // Enhanced notification
        if ('Notification' in window && Notification.permission === 'granted') {
          new Notification('JARVIS Microphone Active ♾️', {
            body: 'Microphone will stay on indefinitely. Say "Hey JARVIS" anytime.',
            icon: '/favicon.ico',
            requireInteraction: true // Keep notification visible
          });
        }

        // Visual indicator in console
        console.log('%c🎤 MICROPHONE STATUS: INDEFINITE MODE ACTIVE',
          'color: #00ff00; font-size: 16px; font-weight: bold; background: #000; padding: 10px;');

        // Set initial timestamp
        lastSpeechTimeRef.current = Date.now();

      } catch (e) {
        if (e.message && e.message.includes('already started')) {
          console.log('Recognition already active - good!');
        } else {
          console.error('Failed to start indefinite listening:', e);
          setError('Failed to start microphone - retrying...');

          // Retry after a moment
          setTimeout(() => enableContinuousListening(), 1000);
        }
      }
    }
  };

  const disableContinuousListening = () => {
    setContinuousListening(false);
    continuousListeningRef.current = false;
    setIsListening(false);
    setIsWaitingForCommand(false);
    isWaitingForCommandRef.current = false;

    if (recognitionRef.current) {
      // Clear keep-alive interval
      if (recognitionRef.current._keepAliveInterval) {
        clearInterval(recognitionRef.current._keepAliveInterval);
        recognitionRef.current._keepAliveInterval = null;
        console.log('🛑 Keep-alive mechanism stopped');
      }

      // Stop recognition
      try {
        recognitionRef.current.stop();
        console.log('🔴 Microphone stopped - indefinite listening disabled');
      } catch (e) {
        console.log('Stop recognition:', e.message);
      }
    }

    // Visual indicator in console
    console.log('%c🎤 MICROPHONE STATUS: STOPPED',
      'color: #ff0000; font-size: 16px; font-weight: bold; background: #000; padding: 10px;');
  };

  const startListening = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await sendAudioToJarvis(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsListening(true);

      // Auto-stop after 5 seconds
      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          stopListening();
        }
      }, 5000);
    } catch (err) {
      console.error('Failed to start recording:', err);
      setError('Microphone access denied');
    }
  };

  const stopListening = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsListening(false);

      // Stop all tracks
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const sendAudioToJarvis = async (audioBlob) => {
    setIsProcessing(true);

    // Convert blob to base64
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = () => {
      const base64Audio = reader.result.split(',')[1];

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'audio',
          data: base64Audio
        }));
      }
    };
  };

  const sendTextCommand = async (text) => {
    if (!text.trim()) return;

    setTranscript(text);
    setIsProcessing(true);
    setResponse('');  // Clear previous response

    // Use WebSocket if connected
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'command',
        text: text,
        mode: autonomousMode ? 'autonomous' : 'manual'
      }));
      // Response will come through WebSocket message handler
    } else {
      // WebSocket not connected
      setError('Not connected to JARVIS. Please refresh the page.');
      setIsProcessing(false);
    }
  };

  const playAudioResponse_UNUSED = async (text) => {
    console.log('Playing audio response:', text.substring(0, 100) + '...');

    try {
      // For long text, always use POST method to avoid URL length limits
      // GET requests have a limit of ~2000 characters in the URL
      const usePost = text.length > 500 || text.includes('\n');

      if (!usePost) {
        // Short text: Use GET method with URL (simpler and more reliable)
        const audioUrl = `${API_URL}/audio/speak/${encodeURIComponent(text)}`;
        const audio = new Audio(audioUrl);
        audio.volume = 1.0;

        setIsJarvisSpeaking(true);

        audio.onended = () => {
          console.log('Audio playback completed');
          setIsJarvisSpeaking(false);
        };

        audio.onerror = async (e) => {
          console.warn('GET method failed, trying POST with blob...');

          // Fallback to POST
          await playAudioUsingPost(text);
        };

        await audio.play();
      } else {
        // Long text: Use POST method directly
        await playAudioUsingPost(text);
      }
    } catch (error) {
      console.error('Audio playback failed:', error);
      setIsJarvisSpeaking(false);
    }
  };

  const playAudioUsingPost = async (text, onStartCallback = null) => {
    console.log('[JARVIS Audio] POST: Attempting to play audio via POST');
    try {
      const apiUrl = API_URL || configService.getApiUrl() || 'http://localhost:8010';
      const response = await fetch(`${apiUrl}/audio/speak`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
      });

      console.log('[JARVIS Audio] POST: Response status:', response.status);
      console.log('[JARVIS Audio] POST: Content-Type:', response.headers.get('content-type'));

      if (response.ok) {
        // Get audio data as blob
        const blob = await response.blob();
        console.log('[JARVIS Audio] POST: Received audio blob:', blob.size, 'bytes');

        const audioUrl = URL.createObjectURL(blob);

        const audio2 = new Audio(audioUrl);
        audio2.volume = 1.0;

        setIsJarvisSpeaking(true);

        audio2.onplay = () => {
          console.log('[JARVIS Audio] POST: Playback started');
          // Call the callback when audio ACTUALLY starts playing (perfect sync!)
          if (onStartCallback && typeof onStartCallback === 'function') {
            console.log('[JARVIS Audio] Calling start callback - text will appear NOW');
            onStartCallback();
          }
        };

        audio2.onended = () => {
          console.log('[JARVIS Audio] POST: Playback completed');
          setIsJarvisSpeaking(false);
          isSpeakingRef.current = false;
          URL.revokeObjectURL(audioUrl); // Clean up
          // Process next in queue after a small delay
          setTimeout(() => processNextInSpeechQueue(), 200);
        };

        audio2.onerror = (e) => {
          console.error('[JARVIS Audio] POST: Playback error:', e);
          if (audio2.error) {
            console.error('[JARVIS Audio] POST: Error details:', {
              code: audio2.error.code,
              message: audio2.error.message
            });
          }
          setIsJarvisSpeaking(false);
          isSpeakingRef.current = false;
          URL.revokeObjectURL(audioUrl); // Clean up
          // Process next in queue even after error
          setTimeout(() => processNextInSpeechQueue(), 200);
        };

        await audio2.play();
        console.log('[JARVIS Audio] POST: Playing audio successfully');
      } else {
        throw new Error(`Audio generation failed: ${response.status}`);
      }
    } catch (postError) {
      console.error('[JARVIS Audio] POST: Failed:', postError);
      setIsJarvisSpeaking(false);
      isSpeakingRef.current = false;
      // Process next in queue even after error
      setTimeout(() => processNextInSpeechQueue(), 200);
    }
  };



  // Speech queue to prevent overlapping
  const speechQueueRef = useRef([]);
  const isSpeakingRef = useRef(false);
  
  const processNextInSpeechQueue = async () => {
    if (speechQueueRef.current.length === 0 || isSpeakingRef.current) {
      return;
    }

    const nextItem = speechQueueRef.current.shift();
    // Handle both old format (string) and new format (object with text and callback)
    const text = typeof nextItem === 'string' ? nextItem : nextItem.text;
    const callback = typeof nextItem === 'object' ? nextItem.callback : null;

    await speakResponseInternal(text, callback);
  };
  
  const stopAllSpeech = () => {
    console.log('[JARVIS Audio] Stopping all speech and clearing queue');
    // Clear the queue
    speechQueueRef.current = [];
    // Stop browser synthesis if active
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    // Reset speaking states
    setIsJarvisSpeaking(false);
    isSpeakingRef.current = false;
  };

  const speakResponse = async (text, updateDisplay = true) => {
    // Only set the response in state if explicitly requested (default true for backward compatibility)
    if (updateDisplay) {
      setResponse(text);
    }

    console.log('[JARVIS Audio] Speaking response:', text.substring(0, 100) + '...');
    console.log('[JARVIS Audio] Current speaking state:', isJarvisSpeaking);

    // Add to speech queue instead of skipping
    speechQueueRef.current.push({ text, callback: null });

    // Process queue if not already speaking
    if (!isSpeakingRef.current) {
      await processNextInSpeechQueue();
    }
  };

  const speakResponseWithCallback = async (text, onStartCallback) => {
    console.log('[JARVIS Audio] Speaking response with callback:', text.substring(0, 100) + '...');

    // Add to speech queue with callback
    speechQueueRef.current.push({ text, callback: onStartCallback });

    // Process queue if not already speaking
    if (!isSpeakingRef.current) {
      await processNextInSpeechQueue();
    }
  };
  
  const speakResponseInternal = async (text, onStartCallback = null) => {
    // Prevent overlapping speech
    if (isSpeakingRef.current) {
      console.log('[JARVIS Audio] Already speaking, adding to queue');
      return;
    }

    try {
      console.log('[JARVIS Audio] Setting speaking state to true');
      console.log('[JARVIS Audio] Text to speak (exact):', text);
      setIsJarvisSpeaking(true);
      isSpeakingRef.current = true;

      // Use backend TTS endpoint for consistent voice quality
      // Use POST for any text with special characters or newlines to avoid URL encoding issues
      const hasSpecialChars = /[^\w\s.,!?-]/.test(text);
      const usePost = text.length > 500 || text.includes('\n') || hasSpecialChars;
      console.log('[JARVIS Audio] Text length:', text.length);
      console.log('[JARVIS Audio] Using POST method:', usePost);

      if (!usePost) {
        // Short text: Use GET method with URL
        const apiUrl = API_URL || configService.getApiUrl() || 'http://localhost:8010';
        const audioUrl = `${apiUrl}/audio/speak/${encodeURIComponent(text)}`;
        console.log('[JARVIS Audio] Using GET method:', audioUrl);

        const audio = new Audio();

        // Set up all event handlers before setting src
        audio.onloadstart = () => {
          console.log('[JARVIS Audio] Loading started');
        };

        audio.oncanplaythrough = () => {
          console.log('[JARVIS Audio] Can play through');
        };

        audio.onplay = () => {
          console.log('[JARVIS Audio] GET method playback started');
          // Call the callback when audio ACTUALLY starts playing (perfect sync!)
          if (onStartCallback && typeof onStartCallback === 'function') {
            console.log('[JARVIS Audio] Calling start callback - text will appear NOW');
            onStartCallback();
          }
        };

        audio.onended = () => {
          console.log('[JARVIS Audio] GET method playback completed');
          setIsJarvisSpeaking(false);
          isSpeakingRef.current = false;
          // Process next in queue after a small delay
          setTimeout(() => processNextInSpeechQueue(), 200);
        };

        audio.onerror = async (e) => {
          console.error('[JARVIS Audio] GET audio error:', e);
          if (audio.error) {
            console.error('[JARVIS Audio] Error details:', {
              code: audio.error.code,
              message: audio.error.message
            });
          }
          console.log('[JARVIS Audio] Falling back to POST method');
          // Fallback to POST method with callback
          await playAudioUsingPost(text, onStartCallback);
        };

        // Set source and properties
        audio.src = audioUrl;
        audio.volume = 1.0;
        audio.crossOrigin = 'anonymous';  // Enable CORS
        
        // Try to play
        console.log('[JARVIS Audio] Attempting to play audio...');
        await audio.play();
        console.log('[JARVIS Audio] Play promise resolved');
      } else {
        // Long text: Use POST method directly with callback
        console.log('[JARVIS Audio] Using POST method for long text');
        await playAudioUsingPost(text, onStartCallback);
      }
    } catch (error) {
      console.error('[JARVIS Audio] Playback failed:', error);
      console.error('[JARVIS Audio] Error details:', {
        name: error.name,
        message: error.message,
        stack: error.stack
      });
      setIsJarvisSpeaking(false);
      isSpeakingRef.current = false;
      // Process next in queue even after error
      setTimeout(() => processNextInSpeechQueue(), 200);

      // Fallback to browser speech synthesis if backend TTS fails
      console.log('[JARVIS Audio] Falling back to browser speech synthesis...');
      if ('speechSynthesis' in window) {
        // Cancel any ongoing speech first
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.7;   // Even slower rate for smooth, non-rushed speech
        utterance.pitch = 0.95; // Slightly lower pitch for more authoritative tone
        utterance.volume = 0.9; // Slightly lower volume for more natural sound

        // Try to find Daniel or other British male voice
        const voices = window.speechSynthesis.getVoices();
        console.log(`[JARVIS Audio] Available voices: ${voices.length}`);

        // First try to find Daniel
        let selectedVoice = voices.find(voice => voice.name.includes('Daniel'));

        if (!selectedVoice) {
          // Try other British male voices
          selectedVoice = voices.find(voice =>
            (voice.lang.includes('en-GB') || voice.lang.includes('en_GB')) &&
            (voice.name.includes('Oliver') || voice.name.includes('James') ||
             voice.name.toLowerCase().includes('male'))
          );
        }

        if (!selectedVoice) {
          // Try any British voice
          selectedVoice = voices.find(voice =>
            voice.lang.includes('en-GB') || voice.lang.includes('en_GB')
          );
        }

        if (selectedVoice) {
          utterance.voice = selectedVoice;
          console.log(`[JARVIS Audio] Using voice: ${selectedVoice.name}`);
        } else {
          console.log('[JARVIS Audio] No British voice found, using default');
        }

        utterance.onstart = () => {
          console.log('[JARVIS Audio] Browser speech synthesis started');
          setIsJarvisSpeaking(true);
          // Call the callback when browser speech ACTUALLY starts playing (perfect sync!)
          if (onStartCallback && typeof onStartCallback === 'function') {
            console.log('[JARVIS Audio] Calling start callback - text will appear NOW');
            onStartCallback();
          }
        };
        utterance.onend = () => {
          console.log('[JARVIS Audio] Browser speech completed');
          setIsJarvisSpeaking(false);
          isSpeakingRef.current = false;
          // Process next in queue after a small delay
          setTimeout(() => processNextInSpeechQueue(), 200);
        };
        utterance.onerror = (e) => {
          console.error('[JARVIS Audio] Browser speech error:', e);
          setIsJarvisSpeaking(false);
          isSpeakingRef.current = false;
          // Process next in queue even after error
          setTimeout(() => processNextInSpeechQueue(), 200);
        };

        window.speechSynthesis.speak(utterance);
      }
    }
  };

  return (
    <div className="jarvis-voice-container">
      {/* JARVIS Header */}
      <div className="jarvis-header">
        <h1 className="jarvis-title">
          <span className="jarvis-logo">J.A.R.V.I.S.</span>
          <span className="jarvis-subtitle">Just A Rather Very Intelligent System</span>
        </h1>
      </div>

      {/* Orange microphone indicator when listening */}
      <MicrophoneIndicator isListening={isListening && continuousListening} />

      {microphonePermission !== 'granted' && (
        <MicrophonePermissionHelper
          onPermissionGranted={() => {
            setMicrophonePermission('granted');
            setMicStatus('ready');
            initializeSpeechRecognition();
          }}
        />
      )}

      <div className={`arc-reactor ${isListening ? 'listening' : ''} ${isProcessing ? 'processing' : ''} ${continuousListening ? 'continuous' : ''} ${isWaitingForCommand ? 'waiting' : ''}`}>
        <div className="core"></div>
        <div className="ring ring-1"></div>
        <div className="ring ring-2"></div>
        <div className="ring ring-3"></div>
      </div>

      <div className="jarvis-status">
        <div className={`status-indicator ${jarvisStatus || 'offline'}`}></div>
        <span className="status-text">
          {jarvisStatus === 'online' || jarvisStatus === 'active' ? (
            <>SYSTEM READY {systemMode === 'minimal' && <span className="mode-badge minimal">[MINIMAL MODE]</span>}</>
          ) : jarvisStatus === 'activating' ? (
            <>INITIALIZING...</>
          ) : (
            <>SYSTEM {(jarvisStatus || 'offline').toUpperCase()}</>
          )}
        </span>
        {micStatus === 'error' && (
          <span className="mic-status error">
            <span className="error-dot"></span> MIC ERROR
          </span>
        )}
      </div>
      
      {/* Mode Information Banner */}
      {systemMode === 'minimal' && jarvisStatus === 'online' && (
        <div className="minimal-mode-banner">
          <div className="mode-info">
            <span className="mode-icon">⚡</span>
            <span className="mode-text">Running in Minimal Mode - Full features loading...</span>
            <span className="mode-spinner">🔄</span>
          </div>
        </div>
      )}
      
      {/* Upgrade Success Banner */}
      {showUpgradeSuccess && (
        <div className="upgrade-success-banner">
          <div className="success-info">
            <span className="success-icon">🎉</span>
            <span className="success-text">System Upgraded to Full Mode!</span>
            <span className="success-check">✅</span>
          </div>
          <div className="success-features">
            All advanced features now available: Wake word • ML Audio • Vision • Memory • Tools
          </div>
        </div>
      )}

      {/* Simplified Status Indicator */}
      <div className="status-indicator-bar">
        {continuousListening && !isWaitingForCommand && (
          <div className="status-item wake-active">
            <span className="status-dot"></span>
            <span className="status-text">Say "Hey JARVIS"</span>
          </div>
        )}
        {isWaitingForCommand && (
          <div className="status-item listening">
            <span className="status-dot active"></span>
            <span className="status-text">Listening...</span>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className={`jarvis-error ${error.includes('Network') ? 'network-error' : ''}`}>
          <div className="error-icon">{error.includes('Network') ? '🌐' : '⚠️'}</div>
          <div className="error-text" style={{ whiteSpace: 'pre-line' }}>
            {error}
            {error.includes('Network') && networkRetries > 0 && networkRetries < maxNetworkRetries && (
              <span className="retry-status"> (Auto-retrying...)</span>
            )}
          </div>
          {error.includes('Microphone') && (
            <button
              onClick={checkMicrophonePermission}
              className="jarvis-button retry-button"
            >
              🎤 Retry Microphone Access
            </button>
          )}
        </div>
      )}

      {/* Transcript Display */}
      {(transcript || response) && (
        <div className="jarvis-transcript">
          {transcript && (
            <div className="user-message">
              <span className="message-label">You:</span>
              <span className="message-text">{transcript}</span>
            </div>
          )}
          {response && (
            <div className="jarvis-message">
              <span className="message-label">JARVIS:</span>
              <span className="message-text">{response}</span>
            </div>
          )}
        </div>
      )}

      {/* Workflow Progress */}
      {workflowProgress && (
        <WorkflowProgress 
          workflow={workflowProgress}
          currentAction={workflowProgress.currentAction}
          onCancel={() => {
            // Send cancel request to backend
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({
                type: 'cancel_workflow',
                workflow_id: workflowProgress.workflow_id
              }));
            }
            setWorkflowProgress(null);
          }}
        />
      )}

      {/* Vision Status */}
      {autonomousMode && (
        <div className="vision-status-bar">
          <div className={`vision-connection ${visionConnected ? 'connected' : 'disconnected'}`}>
            <span className="vision-icon">👁️</span>
            <span>Vision: {visionConnected ? 'Connected' : 'Connecting...'}</span>
          </div>
          {workspaceData && workspaceData.workspace && (
            <div className="workspace-summary">
              <span>{workspaceData.workspace.window_count} windows</span>
              {workspaceData.workspace.focused_app && (
                <span> • Focused: {workspaceData.workspace.focused_app}</span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Simplified Control - Only show when needed */}
      {jarvisStatus === 'activating' && (
        <div className="jarvis-controls">
          <div className="initializing-message">Initializing JARVIS systems...</div>
        </div>
      )}

      {/* Command Input Section */}
      <div className="jarvis-input-section">
        <div className="jarvis-input-container">
          <input
            type="text"
            className="jarvis-input"
            placeholder={jarvisStatus === 'online' ? "Say 'Hey JARVIS' or type a command..." : "Initializing..."}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                sendTextCommand(e.target.value);
                e.target.value = '';
              }
            }}
            disabled={!jarvisStatus || jarvisStatus === 'offline' || jarvisStatus === 'error'}
          />
          <button
            className="jarvis-send-button"
            onClick={() => {
              const input = document.querySelector('.jarvis-input');
              if (input.value) {
                sendTextCommand(input.value);
                input.value = '';
              }
            }}
            disabled={!jarvisStatus || jarvisStatus === 'offline' || jarvisStatus === 'error'}
          >
            <span className="send-icon">→</span>
          </button>
        </div>
      </div>


    </div>
  );
};

export default JarvisVoice;