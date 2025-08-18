import React, { useState, useEffect, useRef } from 'react';
import './JarvisVoice.css';
import MicrophonePermissionHelper from './MicrophonePermissionHelper';

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
  
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recognitionRef = useRef(null);
  
  // Get API URL from environment or use default
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const WS_URL = API_URL.replace('http://', 'ws://').replace('https://', 'wss://');

  useEffect(() => {
    // Check JARVIS status on mount
    checkJarvisStatus();
    
    // Check microphone permission first
    checkMicrophonePermission();
    
    // Load voices for speech synthesis
    if ('speechSynthesis' in window) {
      // Load voices
      speechSynthesis.getVoices();
      // Chrome needs this event to load voices
      speechSynthesis.onvoiceschanged = () => {
        speechSynthesis.getVoices();
      };
    }
    
    return () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const checkJarvisStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/voice/jarvis/status`);
      const data = await response.json();
      setJarvisStatus(data.status);
      
      // Only connect WebSocket if JARVIS is available
      if (data.status !== 'offline') {
        setTimeout(() => {
          connectWebSocket();
        }, 500);
      }
    } catch (err) {
      console.error('Failed to check JARVIS status:', err);
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
      // Initialize speech recognition after permission granted
      initializeSpeechRecognition();
    } catch (error) {
      console.error('Microphone permission error:', error);
      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        setMicrophonePermission('denied');
        setError('Microphone access denied. Please grant permission to use JARVIS.');
      } else if (error.name === 'NotFoundError') {
        setMicrophonePermission('no-device');
        setError('No microphone found. Please connect a microphone.');
      } else {
        setMicrophonePermission('error');
        setError('Error accessing microphone: ' + error.message);
      }
    }
  };

  const connectWebSocket = () => {
    // Don't connect if already connected or connecting
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }
    
    try {
      wsRef.current = new WebSocket(`${WS_URL}/voice/jarvis/stream`);
      
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
        setResponse(data.text);
        setIsProcessing(false);
        // Try frontend speech, but backend will also speak
        console.log('Response received, attempting speech...');
        requestAnimationFrame(() => {
          speakResponse(data.text);
        });
        break;
      case 'error':
        setError(data.message);
        setIsProcessing(false);
        break;
      default:
        break;
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
      
      recognitionRef.current.onerror = (event) => {
        // Only log actual errors, not timeouts
        if (event.error !== 'no-speech') {
          console.error('Speech recognition error:', event.error);
        }
        
        if (event.error === 'no-speech') {
          // Silently restart recognition if no speech detected
          if (continuousListening) {
            setTimeout(() => {
              try {
                recognitionRef.current.start();
              } catch (e) {
                // Ignore restart errors
              }
            }, 100);
          }
        } else if (event.error === 'aborted' || event.error === 'network') {
          // More serious errors - notify user
          console.error('Recognition stopped:', event.error);
          setError('Speech recognition stopped. Please reload the page.');
        }
      };
      
      recognitionRef.current.onend = () => {
        // Restart if continuous listening is enabled
        if (continuousListening) {
          setTimeout(() => {
            try {
              recognitionRef.current.start();
            } catch (e) {
              console.log('Recognition restart failed:', e);
            }
          }, 100);
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
    }
    
    // Don't speak here - let the backend handle it
    // speakResponse("Yes?");
    
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
    
    // Send via WebSocket instead of REST API
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'command',
        text: command
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
  
  const enableContinuousListening = () => {
    if (recognitionRef.current) {
      setContinuousListening(true);
      try {
        recognitionRef.current.start();
        console.log('Continuous listening enabled - say "Hey JARVIS"');
      } catch (e) {
        console.log('Recognition already started');
      }
    }
  };
  
  const disableContinuousListening = () => {
    setContinuousListening(false);
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
    
    try {
      const response = await fetch(`${API_URL}/voice/jarvis/command`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
      });
      
      const data = await response.json();
      setResponse(data.response);
      setIsProcessing(false);
      
      speakResponse(data.response);
    } catch (err) {
      console.error('Failed to send command:', err);
      setError('Failed to process command');
      setIsProcessing(false);
    }
  };

  // Pre-select voice for faster speech
  const [selectedVoice, setSelectedVoice] = useState(null);
  
  useEffect(() => {
    // Pre-select the best voice once
    const selectBestVoice = () => {
      const voices = speechSynthesis.getVoices();
      console.log('Available voices:', voices.length);
      
      if (voices.length === 0) {
        // Try again after a delay
        setTimeout(selectBestVoice, 100);
        return;
      }
      
      const preferredVoices = [
        'Daniel', 'Oliver', 'Google UK English Male',
        'Microsoft David - English (United States)', 'Alex',
        'Google US English', 'Microsoft Mark', 'Fred'
      ];
      
      let voiceSelected = false;
      for (const preferredName of preferredVoices) {
        const voice = voices.find(v => 
          v.name.includes(preferredName) && !v.name.includes('Siri')
        );
        if (voice) {
          setSelectedVoice(voice);
          console.log('Selected voice:', voice.name);
          voiceSelected = true;
          break;
        }
      }
      
      // If no preferred voice found, use any English voice
      if (!voiceSelected) {
        const englishVoice = voices.find(v => v.lang.startsWith('en'));
        if (englishVoice) {
          setSelectedVoice(englishVoice);
          console.log('Using fallback English voice:', englishVoice.name);
        }
      }
    };
    
    // Initial attempt
    selectBestVoice();
    
    // Also listen for voice changes
    speechSynthesis.onvoiceschanged = selectBestVoice;
    
    // Force load voices
    speechSynthesis.getVoices();
  }, []);
  
  // Create a queue for speech to prevent overlapping
  const speechQueueRef = useRef([]);
  const isSpeakingRef = useRef(false);
  
  const processSpeechQueue = () => {
    if (isSpeakingRef.current || speechQueueRef.current.length === 0) {
      return;
    }
    
    const { text, voice } = speechQueueRef.current.shift();
    isSpeakingRef.current = true;
    
    const utterance = new SpeechSynthesisUtterance(text);
    
    if (voice) {
      utterance.voice = voice;
    }
    
    utterance.rate = 1.0;
    utterance.pitch = 0.95;
    utterance.volume = 1.0;
    
    utterance.onstart = () => {
      console.log('Speech started:', text);
      setIsJarvisSpeaking(true);
    };
    
    utterance.onend = () => {
      console.log('Speech ended');
      isSpeakingRef.current = false;
      setIsJarvisSpeaking(false);
      // Process next in queue
      setTimeout(processSpeechQueue, 100);
    };
    
    utterance.onerror = (event) => {
      console.error('Speech error:', event.error);
      isSpeakingRef.current = false;
      setIsJarvisSpeaking(false);
      
      // Handle different error types
      if (event.error === 'canceled') {
        // Speech was canceled - likely by another utterance
        console.log('Speech was canceled, checking if retry needed');
        // Only retry if this was the only item and queue is empty
        if (text && speechQueueRef.current.length === 0) {
          console.log('Retrying canceled speech...');
          setTimeout(() => {
            speechQueueRef.current.push({ text, voice });
            processSpeechQueue();
          }, 300);
        } else {
          // Process next in queue
          setTimeout(processSpeechQueue, 100);
        }
      } else if (event.error === 'interrupted') {
        // Speech was interrupted by user
        console.log('Speech interrupted by user');
        // Process next in queue
        setTimeout(processSpeechQueue, 100);
      } else {
        // Other errors - don't retry
        console.error('Speech synthesis error:', event.error);
        // Process next in queue
        setTimeout(processSpeechQueue, 100);
      }
    };
    
    // Only cancel if there's actually something speaking
    if (speechSynthesis.speaking && !speechSynthesis.pending) {
      speechSynthesis.cancel();
      // Wait a bit longer after cancel
      setTimeout(() => {
        try {
          speechSynthesis.speak(utterance);
          console.log('Speech initiated after cancel');
        } catch (error) {
          console.error('Speech failed:', error);
          isSpeakingRef.current = false;
          setIsJarvisSpeaking(false);
        }
      }, 200);
    } else {
      // No need to cancel, speak immediately
      try {
        speechSynthesis.speak(utterance);
        console.log('Speech initiated directly');
      } catch (error) {
        console.error('Speech failed:', error);
        isSpeakingRef.current = false;
        setIsJarvisSpeaking(false);
      }
    }
  };
  
  const speakResponse = (text) => {
    if (!('speechSynthesis' in window)) {
      console.error('Speech synthesis not supported');
      return;
    }
    
    console.log('Queueing speech:', text);
    console.log('Using voice:', selectedVoice?.name || 'default');
    
    // Add to queue
    speechQueueRef.current.push({ text, voice: selectedVoice });
    
    // Process queue
    processSpeechQueue();
  };

  return (
    <div className="jarvis-voice-container">
      {microphonePermission !== 'granted' && (
        <MicrophonePermissionHelper 
          onPermissionGranted={() => {
            setMicrophonePermission('granted');
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
        <div className={`status-indicator ${jarvisStatus}`}></div>
        <span>JARVIS {jarvisStatus.toUpperCase()}</span>
        {continuousListening && !isWaitingForCommand && (
          <span className="listening-mode">
            <span className="pulse-dot"></span> LISTENING FOR "HEY JARVIS"
          </span>
        )}
        {isWaitingForCommand && (
          <span className="waiting-mode">
            <span className="pulse-dot active"></span> AWAITING COMMAND
          </span>
        )}
      </div>
      
      <div className="voice-controls">
        {jarvisStatus === 'offline' && (
          <button onClick={activateJarvis} className="jarvis-button activate">
            Activate JARVIS
          </button>
        )}
        
        {jarvisStatus === 'online' && (
          <button 
            onClick={continuousListening ? disableContinuousListening : enableContinuousListening}
            className={`jarvis-button ${continuousListening ? 'continuous-active' : 'start'}`}
          >
            {continuousListening ? 'Stop Listening' : 'Start Listening'}
          </button>
        )}
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
        
        {/* Debug audio buttons */}
        <div style={{ marginTop: '10px' }}>
          <button 
            onClick={() => {
              console.log('Testing speech with queue...');
              speakResponse('Testing JARVIS voice. Can you hear me, sir?');
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
            Test Audio (Queue)
          </button>
          
          <button 
            onClick={() => {
              console.log('Direct speech test...');
              // Direct test without our system
              speechSynthesis.cancel();
              const u = new SpeechSynthesisUtterance('Direct test. Hello!');
              u.rate = 1.0;
              u.pitch = 1.0;
              u.volume = 1.0;
              
              u.onstart = () => console.log('Direct speech started');
              u.onend = () => console.log('Direct speech ended');
              u.onerror = (e) => console.error('Direct speech error:', e);
              
              setTimeout(() => {
                speechSynthesis.speak(u);
                console.log('Direct speech queued');
              }, 100);
            }}
            style={{
              padding: '5px 10px',
              background: '#444',
              border: '1px solid #666',
              color: '#fff',
              cursor: 'pointer',
              borderRadius: '4px'
            }}
          >
            Test Audio (Direct)
          </button>
        </div>
      </div>
    </div>
  );
};

export default JarvisVoice;