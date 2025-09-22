# JARVIS Voice Unlock Integration Guide

## ✅ Integration Complete!

The Voice Unlock system is now fully integrated with JARVIS. Here's how to get started:

## 1. Install Dependencies

First, install the required dependencies:

```bash
cd backend/voice_unlock
./install_dependencies.sh
```

This will install:
- Audio processing libraries (numpy, librosa, scipy, pyaudio)
- Security libraries (cryptography, keyring)
- macOS integration (PyObjC frameworks)

## 2. Start JARVIS

Start the JARVIS system normally:

```bash
cd backend
python start_system.py
```

You should see:
- "✅ Voice Unlock system initialized" in the startup logs
- "✅ Voice Unlock API mounted" when APIs are loaded

## 3. Test Integration

Run the integration test to verify everything is working:

```bash
cd backend
python test_voice_unlock_integration.py
```

This will check:
- JARVIS health status
- Voice Unlock API availability
- Configuration loading
- Audio system functionality

## 4. Available Commands

You can now use these voice commands with JARVIS:

### Setup Commands
- **"Hey JARVIS, enroll my voice"** - Start voice enrollment process
- **"Hey JARVIS, test audio"** - Test microphone and audio quality

### Control Commands
- **"Hey JARVIS, enable voice unlock"** - Start monitoring for voice unlock
- **"Hey JARVIS, disable voice unlock"** - Stop voice unlock monitoring
- **"Hey JARVIS, voice unlock status"** - Check current status

### Security Commands
- **"Hey JARVIS, delete my voiceprint"** - Remove your voice data
- **"Hey JARVIS, test voice unlock"** - Test unlock without actually unlocking

## 5. API Endpoints

The following API endpoints are available:

### Status & Configuration
- `GET /api/voice-unlock/status` - Get service status
- `GET /api/voice-unlock/config` - Get current configuration
- `POST /api/voice-unlock/config/update` - Update configuration

### Enrollment
- `POST /api/voice-unlock/enrollment/start` - Start enrollment session
- `WS /api/voice-unlock/enrollment/ws/{session_id}` - WebSocket for real-time enrollment

### Authentication
- `POST /api/voice-unlock/authenticate` - Test authentication
- `POST /api/voice-unlock/unlock/toggle` - Enable/disable monitoring

### User Management
- `GET /api/voice-unlock/users` - List enrolled users
- `DELETE /api/voice-unlock/users/{user_id}` - Delete user voiceprint

### Backup
- `POST /api/voice-unlock/backup/export` - Export encrypted backup

## 6. Configuration

Voice Unlock can be configured via environment variables:

```bash
# Audio settings
export VOICE_UNLOCK_SAMPLE_RATE=16000
export VOICE_UNLOCK_MIN_DURATION=1.0
export VOICE_UNLOCK_MAX_DURATION=10.0

# Enrollment settings
export VOICE_UNLOCK_MIN_SAMPLES=3
export VOICE_UNLOCK_MAX_SAMPLES=5
export VOICE_UNLOCK_MIN_QUALITY=0.7

# Security settings
export VOICE_UNLOCK_ANTI_SPOOFING=high
export VOICE_UNLOCK_ENCRYPT=true
export VOICE_UNLOCK_MAX_ATTEMPTS=3

# System integration
export VOICE_UNLOCK_MODE=screensaver
export VOICE_UNLOCK_JARVIS_RESPONSES=true
```

## 7. Usage Workflow

### First Time Setup
1. Say "Hey JARVIS, enroll my voice"
2. Follow the prompts to record 3-5 voice samples
3. JARVIS will confirm when enrollment is complete

### Daily Use
1. Say "Hey JARVIS, enable voice unlock" to start monitoring
2. When your Mac locks/screensaver activates:
   - JARVIS will listen for your unlock phrase
   - Say your enrolled phrase to unlock
3. JARVIS will unlock your Mac if authentication succeeds

### Security
- All voice data is encrypted and stored locally in macOS Keychain
- Anti-spoofing protection prevents replay attacks
- Failed attempts trigger lockout for security

## 8. Troubleshooting

### Audio Issues
If audio capture fails:
1. Check microphone permissions in System Preferences
2. Run `python backend/voice_unlock/utils/audio_capture.py` to test
3. Ensure PortAudio is installed: `brew install portaudio`

### Import Errors
If voice unlock doesn't load:
1. Verify all dependencies installed: `pip list | grep -E "librosa|pyaudio|keyring"`
2. Check PyObjC installation: `python -c "import Cocoa; print('OK')"`
3. Reinstall dependencies if needed

### API Not Available
If API endpoints return 404:
1. Check JARVIS logs for "Voice Unlock API mounted"
2. Verify `/health` endpoint shows voice_unlock as enabled
3. Restart JARVIS if needed

## 9. Development

### Adding New Commands
1. Add patterns to `voice_unlock_patterns` in `unified_command_processor.py`
2. Implement handler in `voice_unlock_handler.py`
3. Test with unified command processor

### Extending Features
The modular architecture allows easy extension:
- Add new anti-spoofing methods in `anti_spoofing.py`
- Implement new storage backends in `keychain_service.py`
- Create custom enrollment flows in `enrollment.py`

## 10. Security Considerations

- Voice data never leaves your Mac
- All processing is done locally
- Voiceprints are encrypted before storage
- Keychain integration ensures OS-level security
- Anti-spoofing prevents recording attacks

---

Voice Unlock is now ready to use! Say "Hey JARVIS, enroll my voice" to get started.