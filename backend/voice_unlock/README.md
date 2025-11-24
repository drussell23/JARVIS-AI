# JARVIS Voice Unlock System - The Apple Watch Alternative

A sophisticated voice-based authentication system for macOS that provides hands-free screen unlocking through voice commands. Perfect for users who don't have an Apple Watch but want the same convenient unlock experience. Built with a hybrid Objective-C and Python architecture for optimal performance and security.

> **Status**: ✅ Fully integrated with JARVIS startup. Voice Unlock starts automatically when you launch JARVIS and processes commands like "unlock my mac" directly without giving instructions.

## Features

- 🎤 **Voice Authentication**: Wake phrase detection with voice recognition - no Apple Watch needed!
- 🔐 **Automatic Screen Unlock**: Say "Hey JARVIS, unlock my Mac" instead of typing passwords
- 🔒 **Secure Password Storage**: Passwords stored encrypted in macOS Keychain
- 🔄 **Background Monitoring**: Always ready when your screen locks
- 📱 **JARVIS Integration**: Seamless alternative to Apple Watch Unlock
- 🌐 **WebSocket Communication**: Real-time bridge between components
- ⚡ **Native Performance**: Built with Objective-C for optimal macOS integration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JARVIS Voice Unlock                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐        ┌──────────────────┐          │
│  │  Objective-C    │        │     Python       │          │
│  │   Daemon        │◀──────▶│    Backend       │          │
│  └────────┬────────┘        └──────────────────┘          │
│           │                                                 │
│  ┌────────▼────────┐        ┌──────────────────┐          │
│  │ Voice Monitor   │        │ WebSocket Bridge │          │
│  └────────┬────────┘        └──────────────────┘          │
│           │                                                 │
│  ┌────────▼────────┐        ┌──────────────────┐          │
│  │ Authenticator   │        │ Screen Unlock    │          │
│  └─────────────────┘        └──────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- macOS 10.14 (Mojave) or later
- Xcode Command Line Tools
- Python 3.8+ (with miniforge3 or similar)
- Terminal with Full Disk Access
- Accessibility and Microphone permissions

## Quick Start

### 1. Enable Screen Unlock (One-Time Setup)

```bash
cd ~/Documents/repos/JARVIS-AI-Agent/backend/voice_unlock
./enable_screen_unlock.sh
```

This will:
- Prompt for your Mac password (stored securely in Keychain)
- Configure the system for actual screen unlocking
- Set up voice enrollment data

### 2. Start JARVIS (Includes Voice Unlock)

```bash
# From the project root
python start_system.py
```

Voice Unlock starts automatically with JARVIS if you've completed step 1. No need to run `start_voice_unlock_system.sh` separately anymore!

### 3. Using Voice Unlock (No Apple Watch Required!)

**The Apple Watch Alternative in Action:**
- "Hey JARVIS, unlock my mac" - Instantly unlocks without typing passwords
- "Hey JARVIS, unlock my screen" - Works just like Apple Watch Unlock
- "Hey JARVIS, unlock the mac" - No hardware required, just your voice!

**Direct unlock phrases (when screen is locked):**
- "Hello JARVIS, unlock my Mac"
- "JARVIS, this is [Your Name]"
- "Open sesame, JARVIS"

> **Perfect for Non-Apple Watch Users**: Get the same hands-free unlock convenience without needing to buy an Apple Watch!

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/jarvis-ai-agent.git
cd jarvis-ai-agent/backend/voice_unlock
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Build Objective-C Components

```bash
cd objc
make all
```

### 4. Grant Required Permissions

The system requires the following permissions:
- **Microphone access**: For voice detection
- **Accessibility permissions**: For keyboard simulation
- **Keychain access**: For secure password storage

Grant these in System Settings > Privacy & Security

## Usage

### Starting Voice Unlock

Voice Unlock now starts automatically with JARVIS! Just run:

```bash
# From the project root
python start_system.py
```

If you need to start Voice Unlock manually for debugging:

```bash
# Manual start (for debugging only)
./start_voice_unlock_system.sh

# Or start components individually:
# 1. Start WebSocket server
python3 objc/server/websocket_server.py &

# 2. Start daemon
./objc/bin/JARVISVoiceUnlockDaemon

# Check status
./objc/bin/JARVISVoiceUnlockDaemon --status
```

### Stopping Voice Unlock

Voice Unlock stops automatically when you stop JARVIS. If you need to stop it manually:

```bash
# Stop all components
pkill -f websocket_server.py
pkill -f JARVISVoiceUnlockDaemon
```

### Voice Commands

**Through JARVIS (anytime):**
- **"Hey JARVIS, unlock my mac"** - Immediately unlocks your screen
- **"Hey JARVIS, unlock my screen"** - Alternative command  
- **"Hey JARVIS, unlock the mac"** - Another variation

**Direct to Voice Unlock (when screen is locked):**
- **"Hello JARVIS, unlock my Mac"** - Standalone unlock command
- **"JARVIS, this is [Your Name]"** - Personal identification
- **"Open sesame, JARVIS"** - Alternative unlock phrase

> **Note**: When using JARVIS commands, the screen unlocks immediately without requiring additional phrases!

### Setup and Enrollment

#### Initial Setup (Required)
```bash
# Run the setup script to store your password
./enable_screen_unlock.sh
```

#### Voice Enrollment (Optional)
The system uses a default voice profile. For enhanced security, you can enroll your specific voice:

```bash
# Through JARVIS interface
"Hey JARVIS, enable voice unlock"
"Hey JARVIS, enroll my voice for unlock"
```

## Configuration

Configuration file: `~/.jarvis/voice_unlock/config.json`

```json
{
  "unlockPhrases": [
    "Hello JARVIS, unlock my Mac",
    "JARVIS, this is Derek",
    "Open sesame, JARVIS"
  ],
  "options": {
    "livenessDetection": true,
    "antiSpoofing": true,
    "continuousAuth": false,
    "debug": false
  },
  "authenticationTimeout": 10.0,
  "maxFailedAttempts": 5
}
```

## Testing

### Complete System Test
```bash
# Run the comprehensive test
./test_complete_system.sh
```

### Component Tests
```bash
# Test keyboard simulation
./test_keyboard_simulation.py

# Build and run unit tests
cd objc
make test
./bin/JARVISVoiceUnlockTest all
```

### Debug Mode
```bash
# Run with detailed logging
./debug_voice_unlock.sh
```

This will show:
- WebSocket connections
- Voice detection events  
- Screen lock/unlock events
- Authentication attempts

## Security Considerations

1. **Password Storage**: Your Mac password is stored encrypted in the macOS Keychain
2. **Voice Biometric**: Voice authentication provides security similar to Apple Watch proximity
3. **Failed Attempts**: Automatic lockout after 5 failed attempts (5-minute cooldown)
4. **Permission Model**: Requires explicit user consent for all system access
5. **Local Processing**: All voice processing happens locally - no cloud dependency

### Security Best Practices
- Use Voice Unlock only in secure environments
- Regularly update your voice enrollment
- Monitor logs for unauthorized attempts
- Revoke stored passwords if device is compromised:
  ```bash
  security delete-generic-password -s com.jarvis.voiceunlock -a unlock_token
  ```

## Troubleshooting

### Common Issues

1. **"Voice Unlock didn't start with JARVIS"**
   - Check if password is stored: `security find-generic-password -s com.jarvis.voiceunlock -a unlock_token -g`
   - Run setup if needed: `./backend/voice_unlock/enable_screen_unlock.sh`
   - Check JARVIS startup logs for "🔐 Voice Unlock system started"

2. **"Voice commands not detected"**
   - Check microphone permissions in System Settings
   - Ensure you're saying the exact wake phrases
   - Speak clearly and directly to the microphone
   - Check logs: `tail -f /tmp/daemon_debug.log`

3. **"Screen doesn't unlock when I say 'unlock my mac'"**
   - Verify Voice Unlock is running: `lsof -i :8765`
   - Check accessibility permissions in System Settings
   - Look for errors in JARVIS logs
   - Re-run `./enable_screen_unlock.sh` if needed

4. **"JARVIS gives instructions instead of unlocking"**
   - This is the old behavior - restart JARVIS to get the updated integration
   - Verify you have the latest code with integrated unlock commands
   - Check that Voice Unlock WebSocket server is running on port 8765

5. **"WebSocket connection failed"**
   - Voice Unlock should handle port conflicts automatically
   - If issues persist, manually kill processes: `pkill -f websocket_server.py`
   - Check JARVIS logs for Voice Unlock startup status

6. **"Daemon crashes or stops"**
   - Check system logs: `log show --predicate 'subsystem == "com.jarvis.voiceunlock"' --last 5m`
   - Rebuild if needed: `cd objc && make clean && make`
   - Check permissions: Terminal needs Full Disk Access

### Logs and Debugging

```bash
# View all logs in real-time
tail -f /tmp/websocket_debug.log /tmp/daemon_debug.log

# Check if components are running
ps aux | grep -E "(websocket_server|JARVISVoiceUnlockDaemon)"

# Test individual components
./objc/bin/JARVISVoiceUnlockDaemon --debug --test
```

## Architecture Details

### Components

1. **Objective-C Daemon** (`JARVISVoiceUnlockDaemon`)
   - Monitors screen lock/unlock events
   - Manages voice detection through Core Audio
   - Handles keyboard simulation for password entry
   - Communicates via WebSocket

2. **Python WebSocket Server** (`websocket_server.py`)
   - Bridge between JARVIS and the Objective-C daemon
   - Handles command routing
   - Manages system status

3. **Voice Monitor** (`JARVISVoiceMonitor`)
   - Continuous audio monitoring
   - Wake phrase detection
   - Audio preprocessing

4. **Screen Unlock Manager** (`JARVISScreenUnlockManager`)
   - Detects screen states
   - Retrieves password from Keychain
   - Simulates keyboard input

### Communication Flow

```
User Voice → Microphone → Voice Monitor → Wake Phrase Detection
    ↓
Authentication → Keychain Password Retrieval → Keyboard Simulation
    ↓
Screen Unlock → Success Notification → JARVIS Integration
```

## Development

### Building from Source

```bash
# Clean build
cd objc
make clean
make all

# Run tests
make test

# Install as system daemon (optional)
sudo make install
```

### Adding New Voice Commands

1. Edit `~/.jarvis/voice_unlock/config.json`
2. Add new phrases to `unlockPhrases` array
3. Restart the Voice Unlock system

### Project Structure

```
voice_unlock/
├── objc/                    # Objective-C components
│   ├── daemon/             # Main daemon
│   ├── framework/          # Voice processing
│   ├── security/           # Screen unlock
│   ├── bridge/             # IPC bridges
│   └── server/             # WebSocket
├── enable_screen_unlock.sh  # Setup script
├── start_voice_unlock_system.sh # Start script
└── test_*.py/sh            # Test scripts
```

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## Current Implementation Status

### ✅ Completed Features
- Full Objective-C daemon with Core Audio integration
- WebSocket communication bridge
- Python integration server
- Keychain password storage and retrieval
- Screen lock/unlock detection
- Keyboard simulation for password entry
- Wake phrase detection
- Background monitoring service
- JARVIS command integration with direct unlock
- Automatic startup with JARVIS
- Integrated lifecycle management
- **🆕 LangGraph Adaptive Authentication** - Intelligent retry with reasoning
- **🆕 LangChain Multi-Factor Fusion** - Voice + Behavioral + Context authentication
- **🆕 Langfuse Observability** - Full tracing and session management
- **🆕 Anti-Spoofing Detection** - Replay attack and voice cloning protection
- **🆕 Progressive Voice Feedback** - Confidence-aware responses

### 🚧 Known Limitations
- Wake phrase detection requires clear speech in quiet environment
- Screen lock detection may vary with different macOS versions

### 🔜 Future Enhancements
- ChromaDB voice pattern storage for enhanced anti-spoofing
- Claude Computer Use for visual security verification
- Playwright remote unlock for multi-device authentication
- Helicone cost optimization for voice processing

---

## 🧠 Advanced AI Integration (v2.0)

### LangGraph Adaptive Authentication

The voice unlock system uses **LangGraph** for intelligent, multi-step authentication reasoning. Instead of simple pass/fail, JARVIS adapts to challenging conditions:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  LangGraph Authentication Flow                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   analyze_audio ──► verify_speaker ──► check_confidence             │
│                                              │                       │
│                          ┌───────────────────┼───────────────────┐  │
│                          │                   │                   │  │
│                          ▼                   ▼                   ▼  │
│                      success            challenge             retry  │
│                          │                   │                   │  │
│                          └───────┬───────────┴───────────────────┘  │
│                                  ▼                                   │
│                          generate_feedback ──► final_decision        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Intelligent Retry**: Detects why verification failed (noise, illness, microphone change)
- **Challenge Questions**: Borderline confidence triggers contextual questions
- **Environmental Awareness**: Adapts to background noise and microphone changes
- **Sick Voice Detection**: Recognizes voice changes due to illness

**Example Flow:**
```
You: "unlock my screen" (72% confidence - below threshold)

JARVIS reasoning chain:
├── Step 1: Partial match detected (72%)
├── Step 2: Analyze why confidence is low
│   └── Background noise detected (SNR: 12 dB, normally 18 dB)
├── Step 3: Generate intelligent retry strategy
│
JARVIS: "I'm having trouble hearing you clearly - there's some
         background noise. Could you try again?"

You: "unlock my screen" (91% confidence)
JARVIS: "Much better! Unlocking for you now, Derek."
```

### LangChain Multi-Factor Authentication

The system uses **LangChain** to orchestrate multiple authentication signals:

```python
# Factor Weights (dynamically adjusted)
{
    "voice": 0.50,       # Primary voice biometric (ECAPA-TDNN embeddings)
    "behavioral": 0.20,  # Speaking patterns, timing, usage history
    "context": 0.15,     # Location, device state, time of day
    "proximity": 0.10,   # Apple Watch, Bluetooth devices
    "history": 0.05      # Past verification success rate
}
```

**Multi-Factor Fusion Example:**
```
Voice: 72% (BELOW threshold alone)
Behavioral: 94% (STRONG - typical unlock time)
Context: 98% (EXCELLENT - home WiFi, known device)

Weighted combination: 91% → PASS

JARVIS: "Voice confidence was a bit lower than usual (you sound tired!),
         but your behavioral patterns and context are perfect."
```

**Graceful Degradation Chain:**
```
1. Primary Voice Auth (85% threshold) → FAILS
2. Voice + Behavioral Fusion (80% threshold) → PASSES ✓
   OR
3. Challenge Question → "What GCP project are you using?"
   OR
4. Proximity Boost (Apple Watch nearby)
   OR
5. Manual Password Fallback
```

### Langfuse Observability

Full tracing and session management for authentication transparency:

**Dashboard**: https://us.cloud.langfuse.com

**What's Tracked:**

| Metric | Description |
|--------|-------------|
| `session_id` | Groups multiple authentication attempts |
| `trace_id` | Individual authentication trace |
| `voice_confidence` | ECAPA-TDNN embedding match score |
| `behavioral_confidence` | Speaking patterns & timing match |
| `context_confidence` | Environment & device state |
| `fused_confidence` | Final weighted combination |
| `decision` | authenticated / denied / challenge_pending |
| `threat_detected` | none / replay_attack / voice_cloning |
| `duration_ms` | Processing time per phase |
| `api_cost_usd` | Estimated API cost |

**Local Backup Logs:**
```bash
# View authentication logs
cat /tmp/jarvis_auth_logs/auth_$(date +%Y%m%d).jsonl | jq .

# Example log entry:
{
  "trace_id": "auth_8e46b9f078754d7e",
  "speaker_name": "Derek",
  "phases": [
    {"phase": "audio_capture", "duration_ms": 45.2, "metrics": {"snr_db": 18.5}},
    {"phase": "speaker_verification", "duration_ms": 150.0, "metrics": {"confidence": 0.92}}
  ],
  "decision": "authenticated",
  "fused_confidence": 0.92,
  "threat_detected": "none"
}
```

### Enabling Detailed Logging

Set environment variables for comprehensive logging:

```bash
# In your .env file or shell
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://us.cloud.langfuse.com"

# Enable debug logging
export JARVIS_LOG_LEVEL="DEBUG"
export JARVIS_AUTH_LOG_LEVEL="DEBUG"
```

**Log Locations:**
- **Langfuse Dashboard**: https://us.cloud.langfuse.com (real-time traces)
- **Local Backup**: `/tmp/jarvis_auth_logs/auth_YYYYMMDD.jsonl`
- **JARVIS Logs**: Standard JARVIS output
- **Daemon Logs**: `/tmp/daemon_debug.log`
- **WebSocket Logs**: `/tmp/websocket_debug.log`

### Anti-Spoofing Detection

The system includes multiple layers of protection:

| Attack Type | Detection Method |
|-------------|------------------|
| **Replay Attack** | Audio fingerprint + temporal matching |
| **Voice Cloning** | Spectral artifact analysis |
| **Recording Playback** | Room acoustics + liveness detection |
| **Deepfake** | Temporal inconsistencies + breathing patterns |

**Security Alert Example:**
```
Attacker: [Plays recording of your voice]
Voice match: 89% - SHOULD pass

But ChromaDB detects anomalies:
├── Speech rhythm: Too perfect (95% anomaly score)
├── Background noise: Exact same pattern (playback indicator)
├── Breathing pattern: Missing (recorded audio artifact)

JARVIS: "Security alert: I detected characteristics consistent
         with a recording playback. Access denied."
```

### Progressive Voice Feedback

Confidence-aware responses that feel natural:

| Confidence | Response Style |
|------------|----------------|
| **>90%** | "Of course, Derek. Unlocking for you." |
| **85-90%** | "Good morning, Derek. Unlocking now." |
| **80-85%** | "One moment... yes, verified. Unlocking." |
| **75-80%** | "I'm having trouble hearing you. Try again?" |
| **<75%** | "Voice verification didn't match. Use password?" |

**Environmental Awareness:**
```
Noisy environment:
JARVIS: "Give me a second - filtering out background noise...
         Got it - verified despite the coffee shop chatter."

Late night:
JARVIS: "Up late again? Unlocking quietly for you."

Sick voice:
JARVIS: "Your voice sounds different - hope you're feeling okay.
         I can still verify it's you from your speech patterns."
```

---

## 📊 Observability Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Voice Unlock Observability                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │   Voice     │────►│  LangGraph  │────►│  Langfuse   │           │
│  │   Input     │     │  Reasoning  │     │  Tracing    │           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │  ECAPA-TDNN │     │  Multi-     │     │   Cloud     │           │
│  │  Embeddings │     │  Factor     │     │  Dashboard  │           │
│  └─────────────┘     │  Fusion     │     └─────────────┘           │
│         │            └─────────────┘            │                   │
│         ▼                   │                   ▼                   │
│  ┌─────────────┐            │            ┌─────────────┐           │
│  │  ChromaDB   │◄───────────┘            │   Local     │           │
│  │  Patterns   │                         │   Backup    │           │
│  └─────────────┘                         └─────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Session Lifecycle

```python
# Session groups multiple authentication attempts
session_id = audit.start_session(user_id='Derek', device='mac')

# Each attempt creates a trace
trace_id = audit.start_trace(speaker_name='Derek', environment='home')

# Phases are logged as nested spans
audit.log_phase(trace_id, 'audio_capture', 45.2, {'snr_db': 18.5})
audit.log_phase(trace_id, 'speaker_verification', 150.0, {'confidence': 0.92})

# Reasoning steps are tracked
audit.log_reasoning_step(
    trace_id, 'check_confidence',
    input_data={'voice_confidence': 0.92},
    output_data={'decision': 'authenticated'},
    reasoning='Confidence above threshold'
)

# Trace completion
audit.complete_trace(trace_id, 'authenticated', 0.92)

# Session ends
audit.end_session(session_id, 'authenticated')
```

## License

This project is part of the JARVIS AI Agent system. See main project license.

## Support

For issues and questions:
- Check `FINAL_TEST_INSTRUCTIONS.md` for detailed testing steps
- View logs in `/tmp/daemon_debug.log` and `/tmp/websocket_debug.log`
- GitHub Issues: [Link to issues]
- JARVIS Documentation: [Link to docs]

---

**Note**: Voice Unlock requires careful security consideration. Always ensure your Mac is in a secure location when using voice authentication. This feature stores your login password in the Keychain and should only be used on trusted devices.