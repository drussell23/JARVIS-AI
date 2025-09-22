# JARVIS Voice Unlock - Final Test Instructions

## Current Status ✅
1. **Password stored in Keychain** ✅
2. **WebSocket server running** ✅  
3. **Voice Unlock daemon running** ✅
4. **WebSocket connection established** ✅
5. **System monitoring for voice** ✅

## Test Procedure

### Step 1: Verify System is Running
```bash
# Check processes
ps aux | grep -E "(websocket_server|JARVISVoiceUnlockDaemon)" | grep -v grep

# Check logs
tail -f /tmp/daemon_test.log
```

### Step 2: Test Voice Detection (Screen Unlocked)
1. With your screen unlocked, say "Hello JARVIS"
2. Check the daemon log to see if it detects any audio

### Step 3: Test Screen Lock Detection  
1. Lock your screen (⌘+Control+Q)
2. Check daemon log - it should show:
   - `isScreenLocked = 1`
   - Voice monitoring in high sensitivity mode

### Step 4: Test Voice Unlock
1. With screen locked, clearly say one of:
   - "Hello JARVIS, unlock my Mac"
   - "JARVIS, this is Derek"
   - "Open sesame, JARVIS"

2. The system should:
   - Detect the wake phrase
   - Authenticate your voice
   - Type your password
   - Press Enter to unlock

## Troubleshooting

### If voice is not detected:
1. Check microphone permissions:
   ```bash
   # In System Settings > Privacy & Security > Microphone
   # Ensure Terminal/iTerm has permission
   ```

2. Test microphone:
   ```bash
   # Record a test
   rec test.wav
   # Play it back
   play test.wav
   ```

### If screen doesn't unlock:
1. Verify password is correct:
   ```bash
   # Test password retrieval (won't show actual password)
   security find-generic-password -s com.jarvis.voiceunlock -a unlock_token -g
   ```

2. Check accessibility permissions:
   ```bash
   # In System Settings > Privacy & Security > Accessibility
   # Ensure Terminal/iTerm has permission
   ```

### Debug Commands:
```bash
# Watch all logs
tail -f /tmp/websocket_test.log /tmp/daemon_test.log

# Test keyboard simulation manually
osascript -e 'tell application "System Events" to keystroke "test"'

# Check if screen is locked
pgrep -x "ScreenSaver"
```

## Known Issues

1. **Voice detection sensitivity**: The system needs clear speech. Speak directly to the microphone.

2. **Screen lock detection**: macOS may have different lock states. The system detects:
   - ScreenSaver with password
   - Login window
   - Fast user switching

3. **Timing**: After saying the wake phrase, the system needs 1-2 seconds to:
   - Process the audio
   - Verify the phrase
   - Authenticate
   - Type password

## Next Steps

If the test is successful:
1. The system can be installed as a launch daemon
2. It will start automatically on boot
3. Voice unlock will always be available

If issues persist:
1. Check `/tmp/daemon_test.log` for detailed errors
2. Verify all permissions are granted
3. Test each component individually