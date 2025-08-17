import React, { useState, useEffect, useRef } from 'react';
import './JarvisVoice.css';

const JarvisVoice = () => {
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [jarvisStatus, setJarvisStatus] = useState('offline');
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [error, setError] = useState(null);
  const [continuousListening, setContinuousListening] = useState(false);
  const [isWaitingForCommand, setIsWaitingForCommand] = useState(false);
  
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recognitionRef = useRef(null);

  useEffect(() => {
    // Check JARVIS status on mount
    checkJarvisStatus();
    
    // Initialize speech recognition
    initializeSpeechRecognition();
    
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
      const response = await fetch('http://localhost:8000/voice/jarvis/status');
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

  const connectWebSocket = () => {
    // Don't connect if already connected or connecting
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }
    
    try {
      wsRef.current = new WebSocket('ws://localhost:8000/voice/jarvis/stream');
      
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
        break;
      case 'response':
        setResponse(data.text);
        setIsProcessing(false);
        // Speak the response
        speakResponse(data.text);
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
      
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';
      
      recognitionRef.current.onresult = (event) => {
        const last = event.results.length - 1;
        const transcript = event.results[last][0].transcript.toLowerCase();
        
        // Check for wake word
        if ((transcript.includes('hey jarvis') || transcript.includes('jarvis')) && !isWaitingForCommand) {
          console.log('Wake word detected:', transcript);
          handleWakeWordDetected();
        } else if (isWaitingForCommand && event.results[last].isFinal) {
          // Process the command after wake word
          handleVoiceCommand(event.results[last][0].transcript);
        }
      };
      
      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'no-speech') {
          // Restart recognition if no speech detected
          if (continuousListening) {
            recognitionRef.current.start();
          }
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
    
    // Play activation sound and show visual feedback
    speakResponse("Yes, sir?");
    
    // Timeout for command after 5 seconds
    setTimeout(() => {
      if (isWaitingForCommand) {
        setIsWaitingForCommand(false);
        setIsListening(false);
      }
    }, 5000);
  };

  const handleVoiceCommand = (command) => {
    console.log('Command received:', command);
    setTranscript(command);
    setIsWaitingForCommand(false);
    setIsListening(false);
    sendTextCommand(command);
  };

  const activateJarvis = async () => {
    try {
      const response = await fetch('http://localhost:8000/voice/jarvis/activate', {
        method: 'POST'
      });
      const data = await response.json();
      setJarvisStatus('activating');
      setTimeout(() => {
        setJarvisStatus('online');
        speakResponse(data.activation_phrase);
        
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
      const response = await fetch('http://localhost:8000/voice/jarvis/command', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
      });
      
      const data = await response.json();
      setResponse(data.response);
      setIsProcessing(false);
      
      // Speak the response
      speakResponse(data.response);
    } catch (err) {
      console.error('Failed to send command:', err);
      setError('Failed to process command');
      setIsProcessing(false);
    }
  };

  const speakResponse = (text) => {
    // Use browser's speech synthesis as fallback
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      
      // Try to use a British voice
      const voices = speechSynthesis.getVoices();
      const britishVoice = voices.find(voice => 
        voice.lang.includes('en-GB') || voice.name.includes('British')
      );
      
      if (britishVoice) {
        utterance.voice = britishVoice;
      }
      
      utterance.rate = 0.9;
      utterance.pitch = 0.95;
      
      speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="jarvis-voice-container">
      <div className={`arc-reactor ${isListening ? 'listening' : ''} ${isProcessing ? 'processing' : ''} ${continuousListening ? 'continuous' : ''} ${isWaitingForCommand ? 'waiting' : ''}`}>
        <div className="core"></div>
        <div className="ring ring-1"></div>
        <div className="ring ring-2"></div>
        <div className="ring ring-3"></div>
      </div>
      
      <div className="jarvis-status">
        <div className={`status-indicator ${jarvisStatus}`}></div>
        <span>JARVIS {jarvisStatus.toUpperCase()}</span>
        {continuousListening && (
          <span className="listening-mode"> - LISTENING FOR "HEY JARVIS"</span>
        )}
        {isWaitingForCommand && (
          <span className="waiting-mode"> - AWAITING COMMAND</span>
        )}
      </div>
      
      <div className="voice-controls">
        {jarvisStatus === 'offline' && (
          <button onClick={activateJarvis} className="jarvis-button activate">
            Activate JARVIS
          </button>
        )}
        
        {jarvisStatus === 'online' && (
          <>
            {!continuousListening && (
              <button 
                onClick={isListening ? stopListening : startListening}
                className={`jarvis-button ${isListening ? 'stop' : 'start'}`}
                disabled={isProcessing}
              >
                {isListening ? 'Stop Listening' : 'Start Listening'}
              </button>
            )}
            <button 
              onClick={continuousListening ? disableContinuousListening : enableContinuousListening}
              className={`jarvis-button ${continuousListening ? 'continuous-active' : 'continuous'}`}
            >
              {continuousListening ? 'Disable Wake Word' : 'Enable Wake Word'}
            </button>
          </>
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
        <p>Say "Hey JARVIS" to activate voice control</p>
        <p>Available commands: weather, time, calculations, reminders</p>
      </div>
    </div>
  );
};

export default JarvisVoice;