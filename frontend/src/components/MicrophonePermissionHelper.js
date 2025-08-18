import React, { useState, useEffect } from 'react';
import './MicrophonePermissionHelper.css';

const MicrophonePermissionHelper = ({ onPermissionGranted }) => {
  const [permissionStatus, setPermissionStatus] = useState('checking');
  const [browserInfo, setBrowserInfo] = useState('');
  const [showInstructions, setShowInstructions] = useState(false);

  useEffect(() => {
    checkMicrophonePermission();
    detectBrowser();
  }, []);

  const detectBrowser = () => {
    const userAgent = navigator.userAgent;
    let browser = 'Unknown';
    
    if (userAgent.indexOf('Chrome') > -1) {
      browser = 'Chrome';
    } else if (userAgent.indexOf('Safari') > -1 && userAgent.indexOf('Chrome') === -1) {
      browser = 'Safari';
    } else if (userAgent.indexOf('Firefox') > -1) {
      browser = 'Firefox';
    } else if (userAgent.indexOf('Edge') > -1) {
      browser = 'Edge';
    }
    
    setBrowserInfo(browser);
  };

  const checkMicrophonePermission = async () => {
    try {
      // Check if mediaDevices API is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setPermissionStatus('unsupported');
        return;
      }

      // Check permission status if available
      if (navigator.permissions && navigator.permissions.query) {
        try {
          const permission = await navigator.permissions.query({ name: 'microphone' });
          setPermissionStatus(permission.state);
          
          permission.addEventListener('change', () => {
            setPermissionStatus(permission.state);
            if (permission.state === 'granted') {
              onPermissionGranted && onPermissionGranted();
            }
          });
        } catch (e) {
          // Fallback to requesting permission
          requestMicrophoneAccess();
        }
      } else {
        // Directly request permission
        requestMicrophoneAccess();
      }
    } catch (error) {
      console.error('Error checking microphone permission:', error);
      setPermissionStatus('error');
    }
  };

  const requestMicrophoneAccess = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      setPermissionStatus('granted');
      onPermissionGranted && onPermissionGranted();
    } catch (error) {
      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        setPermissionStatus('denied');
      } else if (error.name === 'NotFoundError') {
        setPermissionStatus('no-device');
      } else {
        setPermissionStatus('error');
      }
    }
  };

  const getBrowserInstructions = () => {
    switch (browserInfo) {
      case 'Chrome':
      case 'Edge':
        return (
          <ol>
            <li>Click the lock icon 🔒 in the address bar</li>
            <li>Find "Microphone" in the permissions list</li>
            <li>Change it from "Block" to "Allow"</li>
            <li>Reload the page</li>
          </ol>
        );
      case 'Safari':
        return (
          <ol>
            <li>Go to Safari → Preferences → Websites → Microphone</li>
            <li>Find localhost:3000 in the list</li>
            <li>Change to "Allow"</li>
            <li>Reload the page</li>
          </ol>
        );
      case 'Firefox':
        return (
          <ol>
            <li>Click the microphone icon 🎤 in the address bar</li>
            <li>Click "Blocked Temporarily" dropdown</li>
            <li>Select "Allow"</li>
            <li>Reload the page</li>
          </ol>
        );
      default:
        return (
          <ol>
            <li>Check your browser settings for microphone permissions</li>
            <li>Allow access for localhost:3000</li>
            <li>Reload the page</li>
          </ol>
        );
    }
  };

  if (permissionStatus === 'granted') {
    return null; // Don't show anything if permission is granted
  }

  return (
    <div className="microphone-permission-helper">
      <div className="permission-status">
        {permissionStatus === 'checking' && (
          <>
            <div className="spinner"></div>
            <p>Checking microphone access...</p>
          </>
        )}
        
        {permissionStatus === 'prompt' && (
          <>
            <p>🎤 Microphone permission required</p>
            <button onClick={requestMicrophoneAccess} className="permission-button">
              Grant Microphone Access
            </button>
          </>
        )}
        
        {permissionStatus === 'denied' && (
          <>
            <p className="error">❌ Microphone access denied</p>
            <p>JARVIS needs microphone access for voice commands</p>
            <button onClick={() => setShowInstructions(!showInstructions)} className="help-button">
              How to Enable Microphone
            </button>
            {showInstructions && (
              <div className="instructions">
                <h4>Enable Microphone in {browserInfo}:</h4>
                {getBrowserInstructions()}
              </div>
            )}
          </>
        )}
        
        {permissionStatus === 'no-device' && (
          <>
            <p className="error">🎤 No microphone found</p>
            <p>Please connect a microphone and reload the page</p>
            <p className="device-list">Available devices: {navigator.mediaDevices ? 'Checking...' : 'Not supported'}</p>
          </>
        )}
        
        {permissionStatus === 'unsupported' && (
          <>
            <p className="error">❌ Browser not supported</p>
            <p>Please use Chrome, Firefox, Safari, or Edge</p>
          </>
        )}
        
        {permissionStatus === 'error' && (
          <>
            <p className="error">❌ Error accessing microphone</p>
            <button onClick={checkMicrophonePermission} className="retry-button">
              Try Again
            </button>
          </>
        )}
      </div>
      
      <div className="troubleshooting-tips">
        <h4>Quick Fixes:</h4>
        <ul>
          <li>Make sure you're using HTTPS or localhost</li>
          <li>Check if other apps are using the microphone</li>
          <li>Try closing and reopening your browser</li>
          <li>On macOS: System Preferences → Security → Privacy → Microphone</li>
        </ul>
      </div>
    </div>
  );
};

export default MicrophonePermissionHelper;