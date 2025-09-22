# JARVIS Proximity + Voice Authentication - Implementation Plan

## Overview
This document outlines the technical implementation plan for the dual-factor authentication system combining Apple Watch proximity detection with voice biometric authentication.

## Phase 1: Foundation (Weeks 1-2)

### 1.1 Swift Proximity Service
```
backend/voice_unlock/proximity_voice_auth/
├── swift/
│   ├── ProximityService/
│   │   ├── AppleWatchDetector.swift
│   │   ├── BluetoothLEScanner.swift
│   │   ├── ProximityCalculator.swift
│   │   └── WatchConnectionManager.swift
│   └── Package.swift
```

**Key Components:**
- **AppleWatchDetector.swift** - Core proximity detection using Core Bluetooth
- **BluetoothLEScanner.swift** - BLE scanning and signal strength monitoring
- **ProximityCalculator.swift** - Distance estimation and confidence scoring
- **WatchConnectionManager.swift** - Connection state management

### 1.2 Python Voice Service Enhancement
```
backend/voice_unlock/proximity_voice_auth/
├── python/
│   ├── voice_biometrics/
│   │   ├── __init__.py
│   │   ├── voice_authenticator.py
│   │   ├── feature_extractor.py
│   │   ├── liveness_detector.py
│   │   └── voice_model.py
│   └── requirements.txt
```

**Key Components:**
- **voice_authenticator.py** - Enhanced voice authentication with biometric analysis
- **feature_extractor.py** - MFCC, spectral, and prosodic feature extraction
- **liveness_detector.py** - Anti-spoofing and replay attack prevention
- **voice_model.py** - Machine learning model for voice recognition

### 1.3 Authentication Engine
```
backend/voice_unlock/proximity_voice_auth/
├── auth_engine/
│   ├── __init__.py
│   ├── dual_factor_auth.py
│   ├── security_logger.py
│   └── auth_config.yaml
```

**Key Components:**
- **dual_factor_auth.py** - Combines proximity and voice scores
- **security_logger.py** - Comprehensive audit logging
- **auth_config.yaml** - Configurable thresholds and settings

## Phase 2: Advanced Features (Weeks 3-4)

### 2.1 Continuous Learning System
```
backend/voice_unlock/proximity_voice_auth/
├── ml/
│   ├── __init__.py
│   ├── online_learning.py
│   ├── voice_adaptation.py
│   ├── anomaly_detection.py
│   └── models/
```

**Key Components:**
- **online_learning.py** - Incremental learning from interactions
- **voice_adaptation.py** - Adapts to voice changes over time
- **anomaly_detection.py** - Detects security threats

### 2.2 Anti-Spoofing Protection
```
backend/voice_unlock/proximity_voice_auth/
├── security/
│   ├── __init__.py
│   ├── replay_detection.py
│   ├── synthetic_speech_detector.py
│   ├── proximity_validator.py
│   └── threat_analyzer.py
```

**Key Components:**
- **replay_detection.py** - Audio fingerprinting to prevent replays
- **synthetic_speech_detector.py** - ML-based synthetic speech detection
- **proximity_validator.py** - Validates genuine Apple Watch signals
- **threat_analyzer.py** - Real-time threat analysis

### 2.3 Security Monitoring
```
backend/voice_unlock/proximity_voice_auth/
├── monitoring/
│   ├── __init__.py
│   ├── security_dashboard.py
│   ├── alert_system.py
│   └── metrics_collector.py
```

## Phase 3: Integration & Polish (Weeks 5-6)

### 3.1 JARVIS Integration
```
backend/api/
├── proximity_voice_integration.py
└── proximity_voice_api.py
```

### 3.2 WebSocket Communication
```
backend/voice_unlock/proximity_voice_auth/
├── websocket/
│   ├── proximity_voice_server.py
│   └── auth_event_broadcaster.py
```

## Technical Implementation Details

### Swift-Python Communication (ZeroMQ)
```python
# Python side (voice_authenticator.py)
import zmq

class VoiceAuthenticator:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://127.0.0.1:5555")
    
    def process_auth_request(self, proximity_score, voice_data):
        # Process voice authentication
        voice_score = self.analyze_voice(voice_data)
        
        # Combine with proximity score
        auth_result = self.dual_factor_decision(
            proximity_score, 
            voice_score
        )
        
        return auth_result
```

```swift
// Swift side (ProximityService.swift)
import ZeroMQ

class ProximityService {
    let context = try! ZMQ.Context()
    let socket: ZMQ.Socket
    
    func sendAuthRequest(proximityScore: Double, voiceData: Data) {
        let request = AuthRequest(
            proximityScore: proximityScore,
            voiceData: voiceData
        )
        
        socket.send(request.serialize())
        let response = socket.receive()
        // Process authentication response
    }
}
```

### Voice Biometric Implementation
```python
# feature_extractor.py
import librosa
import numpy as np

class VoiceFeatureExtractor:
    def extract_features(self, audio_data, sample_rate):
        features = {}
        
        # MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=sample_rate, 
            n_mfcc=13
        )
        features['mfcc'] = mfccs.mean(axis=1)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data, 
            sr=sample_rate
        )
        features['spectral_centroid'] = spectral_centroid.mean()
        
        # Prosodic features
        f0 = librosa.piptrack(y=audio_data, sr=sample_rate)[0]
        features['pitch'] = np.mean(f0[f0 > 0])
        
        return features
```

### Proximity Detection Implementation
```swift
// AppleWatchDetector.swift
import CoreBluetooth
import CoreLocation

class AppleWatchDetector: NSObject {
    private var centralManager: CBCentralManager!
    private var proximityScore: Double = 0.0
    
    func startScanning() {
        centralManager = CBCentralManager(
            delegate: self, 
            queue: nil
        )
    }
    
    func calculateProximity(rssi: Int) -> Double {
        // Convert RSSI to distance estimate
        let txPower = -59 // Calibrated value
        let n = 2.0 // Path loss exponent
        
        let distance = pow(10, 
            Double(txPower - rssi) / (10.0 * n)
        )
        
        // Convert to proximity score (0-100)
        return max(0, min(100, 100 - (distance * 10)))
    }
}
```

### Dual-Factor Authentication Logic
```python
# dual_factor_auth.py
class DualFactorAuthenticator:
    def __init__(self):
        self.proximity_threshold = 80.0  # 80% confidence
        self.voice_threshold = 85.0      # 85% confidence
        self.combined_threshold = 90.0   # 90% combined
    
    def authenticate(self, proximity_score, voice_score):
        # Check individual thresholds
        if proximity_score < self.proximity_threshold:
            return AuthResult(
                success=False,
                reason="Apple Watch not in proximity"
            )
        
        if voice_score < self.voice_threshold:
            return AuthResult(
                success=False,
                reason="Voice authentication failed"
            )
        
        # Calculate combined score
        combined_score = (
            proximity_score * 0.4 + 
            voice_score * 0.6
        )
        
        if combined_score >= self.combined_threshold:
            return AuthResult(
                success=True,
                confidence=combined_score
            )
        else:
            return AuthResult(
                success=False,
                reason="Combined authentication failed"
            )
```

## Configuration File
```yaml
# auth_config.yaml
proximity:
  detection_range: 3.0  # meters
  update_frequency: 2   # seconds
  min_confidence: 80    # percentage

voice:
  min_samples: 3
  sample_duration: 3.0  # seconds
  min_confidence: 85    # percentage
  liveness_required: true

security:
  combined_threshold: 90  # percentage
  max_attempts: 3
  lockout_duration: 300   # seconds
  log_retention: 365      # days

learning:
  enabled: true
  update_frequency: "realtime"
  retention_period: 90    # days
  privacy_mode: "local_only"
```

## Testing Strategy

### Unit Tests
- Proximity detection accuracy
- Voice feature extraction
- Authentication logic
- Security measures

### Integration Tests
- Swift-Python communication
- End-to-end authentication flow
- JARVIS command integration
- Performance benchmarks

### Security Tests
- Spoofing attack simulation
- Replay attack prevention
- Proximity spoofing attempts
- Stress testing

## Deployment Plan

### Development Environment
1. Set up Swift package dependencies
2. Install Python requirements
3. Configure ZeroMQ communication
4. Test individual components

### Staging Environment
1. Deploy to test Mac
2. Pair with test Apple Watch
3. Enroll test voices
4. Run security scenarios

### Production Deployment
1. Code review and security audit
2. Performance optimization
3. Documentation update
4. Gradual rollout to users

## Success Metrics Tracking

### Performance Metrics
- Authentication response time
- Resource usage (CPU/RAM)
- Battery impact
- Service uptime

### Security Metrics
- False acceptance rate
- False rejection rate
- Threat detection accuracy
- Audit log completeness

### User Experience Metrics
- Setup completion rate
- Daily active usage
- User satisfaction score
- Support ticket volume

This implementation plan provides a clear roadmap for building the JARVIS Proximity + Voice Authentication System with all the advanced features outlined in the PRD.