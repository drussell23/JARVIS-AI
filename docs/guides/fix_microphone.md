# Microphone Permission Fix for JARVIS

## Quick Steps:

1. **In Chrome:**
   - Click the lock icon (🔒) in the address bar
   - Find "Microphone" and change to "Allow"
   - Reload the page

2. **In Safari:**
   - Safari menu → Settings for This Website
   - Check "Allow" for Microphone
   - Reload the page

3. **In Firefox:**
   - Click the lock icon in address bar
   - Click ">" next to "Connection secure"
   - Under Permissions, allow Microphone
   - Reload the page

## System Level (macOS):

1. Open System Preferences
2. Go to Security & Privacy → Privacy
3. Select Microphone from the left sidebar
4. Ensure your browser is checked ✓

## Test After Fixing:

1. Reload the JARVIS page
2. Click "Enable Wake Word"
3. You should NOT see "audio-capture" error
4. Say "Hey JARVIS" to test

## Still Having Issues?

Try these commands in Terminal:
```bash
# List audio devices
ffmpeg -f avfoundation -list_devices true -i ""

# Test microphone directly
rec -d
```

If microphone works in Terminal but not browser, it's definitely a browser permission issue.