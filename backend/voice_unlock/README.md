# JARVIS Voice Unlock System

A sophisticated voice-based authentication system for macOS that enables screen unlocking through voice commands. Built with a hybrid Objective-C and Python architecture for optimal performance and security.

> **Status**: ✅ Fully implemented and functional. The system can detect when your Mac screen is locked and unlock it using voice commands.

## Features

- 🎤 **Voice Authentication**: Wake phrase detection with voice recognition
- 🔐 **Automatic Screen Unlock**: Unlocks your Mac screen using stored credentials
- 🔒 **Secure Password Storage**: Passwords stored encrypted in macOS Keychain
- 🔄 **Background Monitoring**: Automatically activates when screen locks
- 📱 **JARVIS Integration**: Works seamlessly with the JARVIS AI assistant
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

### 1. Enable Screen Unlock (Required First Step)

```bash
cd ~/Documents/repos/JARVIS-AI-Agent/backend/voice_unlock
./enable_screen_unlock.sh
```

This will:
- Prompt for your Mac password (stored securely in Keychain)
- Configure the system for actual screen unlocking
- Set up voice enrollment data

### 2. Start the Voice Unlock System

```bash
./start_voice_unlock_system.sh
```

The system will:
- Start the WebSocket server on port 8765
- Launch the Voice Unlock daemon
- Begin monitoring for screen lock events
- Listen for voice commands when screen is locked

### 3. Test Voice Unlock

1. Lock your screen (⌘+Control+Q)
2. Say one of these phrases clearly:
   - "Hello JARVIS, unlock my Mac"
   - "JARVIS, this is [Your Name]"
   - "Open sesame, JARVIS"
3. Your screen will unlock automatically!

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

### Starting the Voice Unlock Service

```bash
# Quick start (recommended)
./start_voice_unlock_system.sh

# Or start components manually:
# 1. Start WebSocket server
python3 objc/server/websocket_server.py &

# 2. Start daemon
./objc/bin/JARVISVoiceUnlockDaemon

# Check status
./objc/bin/JARVISVoiceUnlockDaemon --status
```

### Stopping the Service

```bash
# Stop all components
pkill -f websocket_server.py
pkill -f JARVISVoiceUnlockDaemon
```

### Voice Commands

When your screen is locked, say one of these phrases:
- **"Hello JARVIS, unlock my Mac"** - Direct unlock command
- **"JARVIS, this is [Your Name]"** - Personal identification
- **"Open sesame, JARVIS"** - Alternative unlock phrase

> **Note**: Speak clearly and wait 1-2 seconds after the wake phrase for processing.

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
2. **Access Control**: Only the Voice Unlock daemon can retrieve the stored password
3. **Failed Attempts**: Automatic lockout after 5 failed attempts (5-minute cooldown)
4. **Permission Model**: Requires explicit user consent for all system access
5. **Local Processing**: All voice processing happens locally on your Mac

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

1. **"Voice commands not detected"**
   - Check microphone permissions in System Settings
   - Ensure you're saying the exact wake phrases
   - Speak clearly and directly to the microphone
   - Check logs: `tail -f /tmp/daemon_debug.log`

2. **"Screen doesn't unlock"**
   - Verify password is stored: `security find-generic-password -s com.jarvis.voiceunlock -a unlock_token -g`
   - Check accessibility permissions in System Settings
   - Ensure the daemon detects screen lock: Check `isScreenLocked` in logs
   - Re-run `./enable_screen_unlock.sh` if needed

3. **"WebSocket connection failed"**
   - Check if port 8765 is in use: `lsof -i :8765`
   - Kill existing processes: `pkill -f websocket_server.py`
   - Restart the system: `./start_voice_unlock_system.sh`

4. **"Daemon crashes or stops"**
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
- JARVIS command integration

### 🚧 Known Limitations
- Voice authentication currently uses a default profile (not speaker-specific)
- Wake phrase detection requires clear speech in quiet environment
- Screen lock detection may vary with different macOS versions
- Requires manual start (not yet installed as system daemon)

### 🔜 Future Enhancements
- Speaker-specific voice enrollment
- Machine learning-based voice verification
- Support for multiple user profiles
- System daemon auto-start on boot
- TouchID fallback option

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