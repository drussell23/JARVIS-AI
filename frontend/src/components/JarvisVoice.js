import React, { useState, useEffect, useRef } from 'react';
import './JarvisVoice.css';
import '../styles/JarvisVoiceError.css';
import MicrophonePermissionHelper from './MicrophonePermissionHelper';
import MicrophoneIndicator from './MicrophoneIndicator';
import mlAudioHandler from '../utils/MLAudioHandler'; // ML-enhanced audio handling
import { getNetworkRecoveryManager } from '../utils/NetworkRecoveryManager'; // Advanced network recovery

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

      // Use main backend port for vision WebSocket
      const wsUrl = `ws://localhost:8010/vision/ws`;
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

  // Get API URL from environment or use default
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const WS_URL = API_URL.replace('http://', 'ws://').replace('https://', 'wss://');

  useEffect(() => {
    // Check JARVIS status on mount
    checkJarvisStatus();

    // Check microphone permission first
    checkMicrophonePermission();

    // Predict potential audio issues
    mlAudioHandler.predictAudioIssue();

    // Inject style to ensure button visibility
    const styleElement = document.createElement('style');
    styleElement.textContent = buttonVisibilityStyle;
    document.head.appendChild(styleElement);

    // Button visibility checker
    const checkButtonsInterval = setInterval(() => {
      const container = document.querySelector('.jarvis-voice-container');
      if (!container) return;
      
      // Look for control buttons
      const hasButtons = container.querySelector('.jarvis-button');
      
      if (!hasButtons) {
        console.warn('JARVIS buttons missing - injecting emergency button');
        // Find the voice-controls div
        const voiceControls = container.querySelector('.voice-controls');
        if (voiceControls && !document.getElementById('jarvis-emergency-button')) {
          // Inject a fallback activate button
          const buttonDiv = document.createElement('div');
          buttonDiv.id = 'jarvis-emergency-button';
          buttonDiv.style.cssText = 'text-align: center; margin-top: 10px;';
          buttonDiv.innerHTML = `
            <button 
              class="jarvis-button activate" 
              style="
                display: inline-block;
                padding: 12px 24px;
                font-size: 16px;
                background: linear-gradient(135deg, #ff6b35 0%, #ff4500 100%);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                margin: 5px;
                transition: all 0.3s;
              "
              onclick="
                console.log('Emergency activate button clicked');
                // Try to call the activate function
                const event = new CustomEvent('jarvis-emergency-activate');
                window.dispatchEvent(event);
              "
            >
              🚀 Activate JARVIS (Emergency)
            </button>
            <div style="margin-top: 5px; font-size: 12px; color: #888;">
              If buttons are missing, click here to activate JARVIS
            </div>
          `;
          voiceControls.appendChild(buttonDiv);
          console.log('Injected emergency JARVIS button');
        }
      }
    }, 1000);

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
      // Remove ML event listeners
      window.removeEventListener('audioIssuePredicted', handleAudioPrediction);
      window.removeEventListener('audioAnomaly', handleAudioAnomaly);
      window.removeEventListener('audioMetricsUpdate', handleAudioMetrics);
      window.removeEventListener('enableTextFallback', handleTextFallback);
      
      // Remove emergency activate listener
      window.removeEventListener('jarvis-emergency-activate', handleEmergencyActivate);
      
      // Clear button checker interval
      if (checkButtonsInterval) {
        clearInterval(checkButtonsInterval);
      }
      
      // Remove injected style
      if (styleElement && styleElement.parentNode) {
        styleElement.parentNode.removeChild(styleElement);
      }
    };
  }, [autonomousMode]);

  const checkJarvisStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/voice/jarvis/status`);
      const data = await response.json();
      console.log('JARVIS status check result:', data);
      
      // Map backend status to frontend status
      const status = data.status || 'offline';
      if (status === 'standby') {
        setJarvisStatus('online'); // Show as online when in standby
      } else {
        setJarvisStatus(status);
      }

      // Connect WebSocket if JARVIS is available (including standby and active)
      if (data.status === 'online' || data.status === 'standby' || data.status === 'active') {
        console.log('JARVIS is available, connecting WebSocket...');
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

  const connectWebSocket = () => {
    // Don't connect if already connected or connecting
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      console.log('WebSocket already connected or connecting');
      return;
    }

    try {
      const wsUrl = `${WS_URL}/voice/jarvis/stream`;
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
      case 'response':
        console.log('WebSocket response received:', data);
        setResponse(data.text || data.message || 'Response received');
        setIsProcessing(false);
        
        // Use audio endpoint directly
        if (data.text) {
          // Always speak the full response, regardless of length or type
          playAudioResponse(data.text);
        }

        // Check for autonomy activation commands in response
        const responseText = data.text.toLowerCase();
        if (data.command_type === 'autonomy_activation' ||
          responseText.includes('autonomous mode activated') ||
          responseText.includes('full autonomy enabled') ||
          responseText.includes('all systems online')) {
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
        recognitionRef.current.speechTimeout = 60000; // 60 seconds
      }
      if ('noSpeechTimeout' in recognitionRef.current) {
        recognitionRef.current.noSpeechTimeout = 15000; // 15 seconds
      }

      recognitionRef.current.onresult = (event) => {
        const last = event.results.length - 1;
        const transcript = event.results[last][0].transcript.toLowerCase();
        const isFinal = event.results[last].isFinal;

        // Only process final results to avoid duplicate detections
        if (!isFinal) return;

        // Check for wake word (but not while JARVIS is speaking)
        if ((transcript.includes('hey jarvis') || transcript.includes('jarvis')) && !isWaitingForCommand && !isJarvisSpeaking) {
          console.log('Wake word detected:', transcript);

          // Extract command after wake word
          let commandAfterWake = null;
          if (transcript.includes('hey jarvis')) {
            commandAfterWake = transcript.split('hey jarvis')[1].trim();
          } else if (transcript.includes('jarvis')) {
            commandAfterWake = transcript.split('jarvis')[1].trim();
          }

          if (commandAfterWake && commandAfterWake.length > 2) {
            // Send command directly without wake word activation
            console.log('Command with wake word:', commandAfterWake);
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({
                type: 'command',
                text: commandAfterWake
              }));
            }
            setTranscript(commandAfterWake);
            setResponse('Processing...');
          } else {
            // Just wake word, no command
            handleWakeWordDetected();
          }
        } else if (isWaitingForCommand) {
          // Process the command after wake word
          handleVoiceCommand(event.results[last][0].transcript);
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
              // Silently restart for no-speech
              console.log('No speech detected, checking continuous listening state...');
              if (continuousListening && isListening) {
                console.log('Restarting speech recognition for continuous listening...');
                setTimeout(() => {
                  try {
                    recognitionRef.current.start();
                  } catch (e) {
                    console.log('Restarting recognition after no-speech error');
                  }
                }, 100);
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
        console.log('Speech recognition ended');
        // Restart if continuous listening is enabled
        if (continuousListening && isListening) {
          console.log('Restarting continuous listening...');
          setTimeout(() => {
            try {
              recognitionRef.current.start();
              console.log('Successfully restarted speech recognition');
            } catch (e) {
              console.log('Recognition restart failed:', e);
              // Try again after a short delay
              if (continuousListening && isListening) {
                setTimeout(() => {
                  try {
                    recognitionRef.current.start();
                  } catch (e2) {
                    console.log('Second restart attempt failed:', e2);
                  }
                }, 1000);
              }
            }
          }, 100);
        } else {
          setIsListening(false);
          setIsWaitingForCommand(false);
        }
      };
    } else {
      setError('Speech recognition not supported in this browser');
    }
  };

  const handleWakeWordDetected = () => {
    setIsWaitingForCommand(true);
    setIsListening(true);

    // Send wake word to JARVIS
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'command',
        text: 'activate'
      }));
    } else {
      // WebSocket not connected, try to establish connection
      console.log('WebSocket not connected, attempting to connect...');
      connectWebSocket();
      // Use fallback to speak locally
      setTimeout(() => {
        speakResponse("Yes, sir?");
      }, 100);
    }

    // Timeout for command after 10 seconds (longer for conversation)
    setTimeout(() => {
      if (isWaitingForCommand && !isJarvisSpeaking) {
        setIsWaitingForCommand(false);
        setIsListening(false);
      }
    }, 10000);
  };

  const handleVoiceCommand = (command) => {
    console.log('Command received:', command);
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

    // Keep listening for follow-up commands (continuous conversation mode)
    if (continuousListening) {
      // Stay in command mode for 10 seconds after response
      setTimeout(() => {
        if (!isJarvisSpeaking) {
          setIsWaitingForCommand(false);
          setIsListening(false);
        }
      }, 10000);
    } else {
      setIsWaitingForCommand(false);
      setIsListening(false);
    }
  };

  const activateJarvis = async () => {
    try {
      const response = await fetch(`${API_URL}/voice/jarvis/activate`, {
        method: 'POST'
      });
      const data = await response.json();
      setJarvisStatus('activating');
      setTimeout(() => {
        setJarvisStatus('online');
        // Backend JARVIS will speak the activation phrase

        // Start continuous listening after activation
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
      setIsListening(true);
      
      // Configure for continuous listening
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      
      try {
        recognitionRef.current.start();
        console.log('Continuous listening enabled - microphone will stay on');
        
        // Inform user that mic is active
        if ('Notification' in window && Notification.permission === 'granted') {
          new Notification('JARVIS is listening', {
            body: 'Microphone is active. Click the button to stop.',
            icon: '/favicon.ico'
          });
        }
      } catch (e) {
        console.log('Recognition already started');
      }
    }
  };

  const disableContinuousListening = () => {
    setContinuousListening(false);
    setIsListening(false);
    setIsWaitingForCommand(false);
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
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

  const playAudioResponse = async (text) => {
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
  
  const playAudioUsingPost = async (text) => {
    try {
      const response = await fetch(`${API_URL}/audio/speak`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
      });
      
      if (response.ok) {
        // Get audio data as blob
        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        
        const audio2 = new Audio(audioUrl);
        audio2.volume = 1.0;
        
        setIsJarvisSpeaking(true);
        
        audio2.onended = () => {
          console.log('Audio playback completed');
          setIsJarvisSpeaking(false);
          URL.revokeObjectURL(audioUrl); // Clean up
        };
        
        audio2.onerror = () => {
          console.warn('Audio playback not available');
          setIsJarvisSpeaking(false);
          URL.revokeObjectURL(audioUrl); // Clean up
        };
        
        await audio2.play();
      } else {
        throw new Error('Audio generation failed');
      }
    } catch (postError) {
      console.warn('Audio POST method failed:', postError.message);
      setIsJarvisSpeaking(false);
    }
  };

  const speakResponse = async (text) => {
    // Try audio endpoint first
    await playAudioResponse(text);
    
    // If browser supports speech synthesis, we could use it as ultimate fallback
    if (!isJarvisSpeaking && 'speechSynthesis' in window) {
      console.log('Using browser speech synthesis as fallback');
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 0.9;
      utterance.volume = 1.0;
      
      // Optional: Use a specific voice
      const voices = window.speechSynthesis.getVoices();
      const englishVoice = voices.find(voice => 
        voice.lang.startsWith('en') && voice.name.includes('Google') ||
        voice.lang.startsWith('en') && voice.name.includes('Microsoft') ||
        voice.lang.startsWith('en-US')
      );
      
      if (englishVoice) {
        utterance.voice = englishVoice;
      }
      
      utterance.onstart = () => setIsJarvisSpeaking(true);
      utterance.onend = () => setIsJarvisSpeaking(false);
      
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="jarvis-voice-container">
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
        <span>JARVIS {jarvisStatus === 'active' ? 'READY' : (jarvisStatus || 'offline').toUpperCase()}</span>
        {micStatus === 'error' && (
          <span className="mic-status error">
            <span className="error-dot"></span> MIC ERROR
          </span>
        )}
        {micStatus === 'ready' && continuousListening && !isWaitingForCommand && (
          <span className="listening-mode">
            <span className="pulse-dot"></span> LISTENING FOR "HEY JARVIS"
          </span>
        )}
        {isWaitingForCommand && (
          <span className="waiting-mode">
            <span className="pulse-dot active"></span> AWAITING COMMAND
          </span>
        )}
        {autonomousMode && (
          <span className="autonomous-mode">
            <span className="vision-indicator"></span> AUTONOMOUS MODE
          </span>
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

      <div className="voice-controls" style={{ minHeight: '60px', padding: '10px' }}>
        {console.log('Current jarvisStatus:', jarvisStatus)}
        {/* Always show appropriate buttons regardless of exact status string */}
        {(!jarvisStatus || jarvisStatus === 'offline' || jarvisStatus === 'error' || jarvisStatus === 'active' || jarvisStatus === 'ready' || jarvisStatus === 'standby') ? (
          <button 
            onClick={activateJarvis} 
            className="jarvis-button activate"
            style={{ 
              display: 'inline-block', 
              visibility: 'visible',
              opacity: 1,
              margin: '5px'
            }}
          >
            🚀 Activate JARVIS
          </button>
        ) : (
          <>
            <button
              onClick={continuousListening ? disableContinuousListening : enableContinuousListening}
              className={`jarvis-button ${continuousListening ? 'continuous-active' : 'start'}`}
              title={continuousListening ? 'Click to turn off microphone' : 'Click to turn on microphone (stays on)'}
            >
              {continuousListening ? '🔴 Turn Off Microphone' : '🎤 Turn On Microphone'}
            </button>

            <button
              onClick={toggleAutonomousMode}
              className={`jarvis-button ${autonomousMode ? 'autonomous-active' : 'autonomous'}`}
            >
              {autonomousMode ? '🤖 Autonomous ON' : '👤 Manual Mode'}
            </button>
          </>
        )}
        
        {/* Fallback button if status is unclear */}
        {jarvisStatus && !['offline', 'error', 'active', 'ready', 'online', 'activating', 'standby'].includes(jarvisStatus) && (
          <div style={{ marginTop: '10px' }}>
            <button onClick={activateJarvis} className="jarvis-button activate">
              🔧 Initialize JARVIS (Status: {jarvisStatus})
            </button>
          </div>
        )}
        
        {/* Debug status display - remove this after troubleshooting */}
        <div style={{ 
          marginTop: '10px', 
          fontSize: '11px', 
          color: '#666',
          backgroundColor: '#f0f0f0',
          padding: '5px',
          borderRadius: '4px'
        }}>
          Debug: Status="{jarvisStatus}" | Expected: offline/active/online/activating
        </div>
      </div>

      <div className="voice-input">
        <input
          type="text"
          placeholder="Or type your command..."
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              sendTextCommand(e.target.value);
              e.target.value = '';
            }
          }}
          disabled={jarvisStatus !== 'online'}
        />
      </div>

      {transcript && (
        <div className="transcript">
          <strong>You:</strong> {transcript}
        </div>
      )}

      {response && (
        <div className="jarvis-response">
          <strong>JARVIS:</strong> {response}
        </div>
      )}

      {error && (
        <div className="error-message">
          {error}
          {error === 'Connection error' && (
            <button
              onClick={() => {
                setError(null);
                connectWebSocket();
              }}
              className="reconnect-button"
            >
              Reconnect
            </button>
          )}
        </div>
      )}

      <div className="voice-tips">
        <p>Click "Start Listening" then say "Hey JARVIS" to activate</p>
        <p>Available commands: weather, time, calculations, reminders</p>

        {/* Debug audio button */}
        <div style={{ marginTop: '10px' }}>
          <button
            onClick={async () => {
              console.log('Testing audio endpoint...');
              try {
                await playAudioResponse('Testing JARVIS voice. Can you hear me?');
              } catch (error) {
                console.error('Test failed:', error);
                setError('Audio test failed: ' + error.message);
              }
            }}
            style={{
              padding: '5px 10px',
              marginRight: '10px',
              background: '#444',
              border: '1px solid #666',
              color: '#fff',
              cursor: 'pointer',
              borderRadius: '4px'
            }}
          >
            Test Audio
          </button>
        </div>
      </div>
    </div>
  );
};

export default JarvisVoice;