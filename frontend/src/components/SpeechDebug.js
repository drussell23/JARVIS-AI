import React, { useState, useEffect } from 'react';
import configService from '../services/DynamicConfigService';

const SpeechDebug = () => {
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(null);
  const [status, setStatus] = useState('Loading...');
  const [volume, setVolume] = useState(1.0);
  const [rate, setRate] = useState(1.0);

  useEffect(() => {
    // Check if speech synthesis is available
    if (!('speechSynthesis' in window)) {
      setStatus('❌ Speech synthesis not supported in this browser');
      return;
    }

    const loadVoices = () => {
      const availableVoices = speechSynthesis.getVoices();
      console.log('Loaded voices:', availableVoices.length);
      setVoices(availableVoices);
      
      if (availableVoices.length > 0) {
        setStatus(`✅ ${availableVoices.length} voices available`);
        
        // Find best English voice
        const englishVoice = availableVoices.find(voice => 
          voice.lang.startsWith('en') && voice.localService
        ) || availableVoices[0];
        
        setSelectedVoice(englishVoice);
      } else {
        setStatus('⏳ Loading voices...');
      }
    };

    // Load immediately
    loadVoices();

    // Also listen for changes (Chrome requirement)
    speechSynthesis.onvoiceschanged = loadVoices;

    // Check audio context state
    if ('AudioContext' in window) {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      console.log('Audio Context State:', audioContext.state);
      
      if (audioContext.state === 'suspended') {
        setStatus(prev => prev + ' (Audio may be blocked - click to enable)');
      }
    }

    return () => {
      speechSynthesis.cancel();
    };
  }, []);

  const testSpeech = (text = "JARVIS speech test. Full autonomy activated. All systems online.") => {
    console.log('Testing speech:', text);
    console.log('Selected voice:', selectedVoice);
    console.log('Speech synthesis state:', {
      speaking: speechSynthesis.speaking,
      pending: speechSynthesis.pending,
      paused: speechSynthesis.paused
    });

    // Cancel any ongoing speech
    speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    
    if (selectedVoice) {
      utterance.voice = selectedVoice;
    }
    
    utterance.volume = volume;
    utterance.rate = rate;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      console.log('Speech started');
      setStatus('🔊 Speaking...');
    };

    utterance.onend = () => {
      console.log('Speech ended');
      setStatus('✅ Speech completed');
    };

    utterance.onerror = (event) => {
      console.error('Speech error:', event);
      setStatus(`❌ Speech error: ${event.error}`);
    };

    try {
      speechSynthesis.speak(utterance);
      console.log('Speech queued successfully');
    } catch (error) {
      console.error('Failed to speak:', error);
      setStatus(`❌ Failed to speak: ${error.message}`);
    }
  };

  const testBackendSpeech = async () => {
    try {
      const apiUrl = configService.getApiUrl() || 'http://localhost:8010';
      const response = await fetch(`${apiUrl}/voice/jarvis/speak`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: "Backend speech test. JARVIS systems online."
        })
      });
      
      if (response.ok) {
        setStatus('✅ Backend speech request sent');
      } else {
        setStatus(`❌ Backend error: ${response.status}`);
      }
    } catch (error) {
      setStatus(`❌ Backend unreachable: ${error.message}`);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      bottom: '20px',
      left: '20px',
      background: 'rgba(0, 0, 0, 0.9)',
      color: '#fff',
      padding: '20px',
      borderRadius: '10px',
      maxWidth: '400px',
      zIndex: 10000,
      fontFamily: 'monospace',
      fontSize: '12px'
    }}>
      <h3 style={{ margin: '0 0 10px 0', color: '#ffd700' }}>🔊 Speech Debug Panel</h3>
      
      <div style={{ marginBottom: '10px' }}>
        <strong>Status:</strong> {status}
      </div>

      <div style={{ marginBottom: '10px' }}>
        <strong>Voices:</strong> {voices.length} available
        {selectedVoice && (
          <div style={{ fontSize: '10px', marginTop: '5px' }}>
            Selected: {selectedVoice.name} ({selectedVoice.lang})
          </div>
        )}
      </div>

      <div style={{ marginBottom: '10px' }}>
        <label>
          Volume: {volume}
          <input 
            type="range" 
            min="0" 
            max="1" 
            step="0.1" 
            value={volume}
            onChange={(e) => setVolume(parseFloat(e.target.value))}
            style={{ width: '100%' }}
          />
        </label>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <label>
          Rate: {rate}
          <input 
            type="range" 
            min="0.5" 
            max="2" 
            step="0.1" 
            value={rate}
            onChange={(e) => setRate(parseFloat(e.target.value))}
            style={{ width: '100%' }}
          />
        </label>
      </div>

      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <button 
          onClick={() => testSpeech()}
          style={{
            padding: '5px 10px',
            background: '#ffd700',
            color: '#000',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          Test Browser Speech
        </button>

        <button 
          onClick={() => testSpeech("Quick test")}
          style={{
            padding: '5px 10px',
            background: '#4CAF50',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          Quick Test
        </button>

        <button 
          onClick={testBackendSpeech}
          style={{
            padding: '5px 10px',
            background: '#2196F3',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          Test Backend
        </button>

        <button 
          onClick={() => window.location.reload()}
          style={{
            padding: '5px 10px',
            background: '#f44336',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          Reload
        </button>
      </div>

      <div style={{ marginTop: '10px', fontSize: '10px', opacity: 0.7 }}>
        Check browser console for detailed logs
      </div>
    </div>
  );
};

export default SpeechDebug;