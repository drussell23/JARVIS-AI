# Frontend Integration Guide
## Voice Pipeline Enhancements

**Last Updated**: 2025-01-13
**Purpose**: Integrate frontend with enhanced voice pipeline (VAD + Windowing + Streaming Safeguard)

---

## 🎯 Overview

The backend now has a sophisticated voice processing pipeline that:
1. **Filters silence** with VAD (WebRTC + Silero)
2. **Limits audio duration** (2s for unlock, 3s for commands, 5s general)
3. **Detects commands** during streaming (e.g., "unlock") and signals stream closure

**Frontend Changes Required**:
- Listen for `stream_stop` WebSocket messages
- Stop recording immediately when command detected
- Prevent further audio chunk accumulation

---

## 📝 Required Changes

### 1. Update `HybridSTTClient.js`

**Location**: `frontend/src/utils/HybridSTTClient.js`

**Changes**:
1. Add `stream_stop` message handler
2. Add auto-stop flag to prevent restarts after command detection
3. Expose `oncommanddetected` callback

**Modified Constructor**:
```javascript
class HybridSTTClient {
  constructor(websocket, options = {}) {
    this.ws = websocket;
    this.options = {
      strategy: options.strategy || 'balanced',
      speakerName: options.speakerName || null,
      confidenceThreshold: options.confidenceThreshold || 0.6,
      retryOnLowConfidence: options.retryOnLowConfidence !== false,
      continuous: options.continuous !== false,
      interimResults: options.interimResults !== false,
      maxRetries: options.maxRetries || 2,
      // 🆕 NEW: Auto-stop on command detection
      autoStopOnCommand: options.autoStopOnCommand !== false,
      targetCommands: options.targetCommands || ['unlock', 'lock', 'jarvis'],
      ...options
    };

    // MediaRecorder for audio capture
    this.mediaRecorder = null;
    this.audioStream = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.isPaused = false;

    // 🆕 NEW: Command detection state
    this.commandDetected = false;
    this.detectedCommand = null;

    // Event handlers (compatible with SpeechRecognition API)
    this.onstart = null;
    this.onend = null;
    this.onresult = null;
    this.onerror = null;
    this.onspeechstart = null;
    this.onspeechend = null;
    this.onaudiostart = null;
    this.onaudioend = null;

    // 🆕 NEW: Command detection callback
    this.oncommanddetected = null;

    // ... rest of constructor
  }
```

**Add Message Handler Method**:
```javascript
  /**
   * 🆕 NEW: Handle stream_stop message from backend
   *
   * Called when backend detects a command (e.g., "unlock") and wants
   * to stop the audio stream immediately to prevent accumulation.
   */
  handleStreamStop(data) {
    const { reason, command, message } = data;

    console.log(`🛡️ [HybridSTT] Stream stop received:`, {
      reason,
      command,
      message
    });

    // Set flag to prevent restart
    this.commandDetected = true;
    this.detectedCommand = command;

    // Stop recording immediately
    if (this.isRecording) {
      console.log(`🛡️ [HybridSTT] Stopping recording due to command detection: "${command}"`);
      this.stop();
    }

    // Trigger command detected event
    this._triggerEvent('commanddetected', {
      command,
      reason,
      message,
      timestamp: Date.now()
    });

    // Reset flag after delay (allow new recording sessions)
    setTimeout(() => {
      this.commandDetected = false;
      this.detectedCommand = null;
    }, 2000); // 2 second cooldown
  }
```

**Modify `start()` Method**:
```javascript
  async start() {
    // 🆕 NEW: Check if command was recently detected
    if (this.commandDetected) {
      console.warn(`🛡️ [HybridSTT] Cannot start - command "${this.detectedCommand}" was recently detected`);
      this._triggerEvent('error', {
        error: 'command-detected-cooldown',
        command: this.detectedCommand
      });
      return;
    }

    if (this.isRecording) {
      console.warn('🎤 [HybridSTT] Already recording');
      return;
    }

    try {
      // ... existing start code ...

      // Start recording
      this.mediaRecorder.start();
      this.isRecording = true;

      console.log(`🎤 [HybridSTT] Recording started (${selectedMimeType})`);
      this._triggerEvent('start');
      this._triggerEvent('audiostart');

      // Auto-stop after silence or time limit (if continuous is false)
      if (!this.options.continuous) {
        setTimeout(() => {
          if (this.isRecording && !this.isPaused && !this.commandDetected) { // 🆕 Check command flag
            this.stop();
          }
        }, 10000); // 10 second max recording
      }

    } catch (error) {
      console.error('🎤 [HybridSTT] Failed to start recording:', error);
      this._triggerEvent('error', { error: error.message });
    }
  }
```

---

### 2. Update WebSocket Message Router

**Location**: Where you initialize the WebSocket connection (likely in `JarvisVoice.js` or similar component)

**Add Handler**:
```javascript
// In your WebSocket message handler
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'transcription_result':
      if (hybridSTTClient) {
        hybridSTTClient.handleTranscriptionResult(data);
      }
      break;

    case 'transcription_error':
      if (hybridSTTClient) {
        hybridSTTClient.handleTranscriptionError(data);
      }
      break;

    // 🆕 NEW: Handle stream_stop messages
    case 'stream_stop':
      if (hybridSTTClient) {
        hybridSTTClient.handleStreamStop(data);
      }
      break;

    // ... other message types ...
  }
};
```

---

### 3. Add Command Detection Callback

**Location**: Component using `HybridSTTClient`

**Example Integration**:
```javascript
// Initialize Hybrid STT Client
const hybridSTTClient = new HybridSTTClient(websocket, {
  strategy: 'balanced',
  continuous: true,
  autoStopOnCommand: true,  // 🆕 Enable auto-stop
  targetCommands: ['unlock', 'lock', 'jarvis'] // 🆕 Commands to detect
});

// 🆕 NEW: Handle command detection
hybridSTTClient.oncommanddetected = (event) => {
  const { command, reason, message } = event;

  console.log(`✅ Command detected: "${command}"`);
  console.log(`   Reason: ${reason}`);
  console.log(`   Message: ${message}`);

  // Show UI feedback
  showNotification(`Command "${command}" detected - stopping recording`, 'success');

  // Optional: Trigger specific actions based on command
  if (command === 'unlock') {
    // Show unlock UI
    showUnlockAnimation();
  } else if (command === 'lock') {
    // Show lock UI
    showLockAnimation();
  }
};

// Handle results as usual
hybridSTTClient.onresult = (event) => {
  const { text, confidence, engine } = event;
  console.log(`Transcription: "${text}" (confidence: ${(confidence * 100).toFixed(1)}%)`);
  // ... process result ...
};

// Start listening
hybridSTTClient.start();
```

---

### 4. Optional: Add Visual Feedback

**Location**: Your voice UI component

**Example UI Update**:
```jsx
const [commandDetected, setCommandDetected] = useState(null);

// In your component
hybridSTTClient.oncommanddetected = (event) => {
  setCommandDetected(event.command);

  // Clear after animation
  setTimeout(() => setCommandDetected(null), 3000);
};

// In your render/JSX
{commandDetected && (
  <div className="command-detected-banner">
    🎯 Command detected: "{commandDetected}" - Stopping audio stream
  </div>
)}
```

---

## 🧪 Testing the Integration

### 1. Test Command Detection

```javascript
// 1. Start recording
hybridSTTClient.start();

// 2. Say a wake phrase or command
// "Hey Jarvis, unlock my screen"

// Expected behavior:
// - Backend transcribes "unlock my screen"
// - Backend detects "unlock" command
// - Backend sends stream_stop message
// - Frontend receives stream_stop
// - Frontend calls hybridSTTClient.handleStreamStop()
// - Recording stops immediately
// - oncommanddetected callback fires
// - UI shows "Command detected: unlock"
```

### 2. Test Cooldown

```javascript
// 1. Trigger command detection
hybridSTTClient.start();
// Say "unlock"
// Wait for stream_stop

// 2. Try to start again immediately
hybridSTTClient.start(); // Should fail with cooldown error

// 3. Wait 2 seconds
setTimeout(() => {
  hybridSTTClient.start(); // Should work
}, 2500);
```

### 3. Test Continuous Mode

```javascript
// 1. Enable continuous mode
const hybridSTTClient = new HybridSTTClient(websocket, {
  continuous: true,
  autoStopOnCommand: false // Disable auto-stop for testing
});

// 2. Start recording
hybridSTTClient.start();

// 3. Say multiple commands
// "Unlock" → transcribed but keeps recording
// "Lock" → transcribed but keeps recording

// 4. With auto-stop enabled
const hybridSTTClient2 = new HybridSTTClient(websocket, {
  continuous: true,
  autoStopOnCommand: true // Enable auto-stop
});

hybridSTTClient2.start();
// "Unlock" → transcribed, stream stops immediately
```

---

## 📊 Expected Behavior

### Before Enhancement
```
User speaks: "unlock my screen"
   ↓
Frontend: Keeps recording... (5s... 10s... 60s...)
   ↓
Backend: Receives 60s of audio
   ↓
Whisper: Processes 60s (slow, hallucinations)
   ↓
Result: "Screen unlock timed out."
```

### After Enhancement
```
User speaks: "unlock my screen"
   ↓
Frontend: Recording... (sends audio chunks)
   ↓
Backend: Receives audio
   ↓
Backend: VAD filters silence (60s → 12s speech)
   ↓
Backend: Windowing truncates (12s → 2s unlock window)
   ↓
Whisper: Transcribes 2s audio → "unlock my screen"
   ↓
Backend: Streaming safeguard detects "unlock"
   ↓
Backend: Sends stream_stop message
   ↓
Frontend: Receives stream_stop, stops recording IMMEDIATELY
   ↓
Backend: Executes unlock flow
   ↓
Result: "Screen unlocked, Sir." ✅
```

---

## 🐛 Troubleshooting

### Issue: `stream_stop` not received

**Check**:
1. WebSocket connection is open
2. Message router includes `stream_stop` case
3. Backend has `ENABLE_STREAMING_SAFEGUARD=true`

**Debug**:
```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('📨 [WS] Received:', data.type, data);

  // Look for stream_stop messages in console
};
```

### Issue: Recording doesn't stop

**Check**:
1. `handleStreamStop()` method is defined
2. `stop()` method is being called
3. MediaRecorder state is checked

**Debug**:
```javascript
handleStreamStop(data) {
  console.log('🛡️ [HybridSTT] Stream stop called');
  console.log('   Is recording:', this.isRecording);
  console.log('   MediaRecorder state:', this.mediaRecorder?.state);

  this.stop();

  console.log('   After stop - Is recording:', this.isRecording);
  console.log('   After stop - MediaRecorder state:', this.mediaRecorder?.state);
}
```

### Issue: Command detected but unlock still times out

**This means**:
- Frontend integration is working ✅
- Backend unlock flow might have other issues

**Check backend logs**:
```bash
grep "COMMAND DETECTED" backend/logs/jarvis*.log
grep "unlock" backend/logs/jarvis*.log
grep "timeout" backend/logs/jarvis*.log
```

---

## 📚 Additional Resources

- [Backend Voice Pipeline Documentation](../docs/VOICE_PIPELINE_NOTES.md)
- [Streaming Safeguard Implementation](../backend/voice/streaming_safeguard.py)
- [VAD Pipeline](../backend/voice/vad/pipeline.py)
- [Audio Windowing](../backend/voice/audio_windowing.py)

---

## ✅ Integration Checklist

- [ ] Update `HybridSTTClient.js` with stream_stop handler
- [ ] Add `oncommanddetected` callback to event handlers
- [ ] Update WebSocket message router with `stream_stop` case
- [ ] Add command detection flag and cooldown logic
- [ ] Test with `console.log` debugging
- [ ] Test unlock flow end-to-end
- [ ] Add UI feedback for command detection (optional)
- [ ] Deploy to production

---

**Questions?** Check the main documentation or file an issue in the GitHub repository.
