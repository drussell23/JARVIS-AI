# JARVIS Troubleshooting Guide

## Microphone Issues

### Error: "NotReadableError: Could not start audio source"

This error typically occurs when:

1. **Another application is using the microphone**
   - Check if Zoom, Teams, Discord, or other apps are using the mic
   - Close other applications that might be accessing the microphone
   - On macOS: Check System Preferences → Security & Privacy → Privacy → Microphone

2. **Browser permissions need to be reset**
   - Chrome: chrome://settings/content/microphone
   - Safari: Safari → Preferences → Websites → Microphone
   - Firefox: about:preferences#privacy → Permissions → Microphone

3. **System-level microphone issues**
   ```bash
   # macOS: Check if microphone is recognized
   system_profiler SPAudioDataType | grep -A 10 "Input"
   
   # Test microphone with built-in tool
   sox -d test.wav trim 0 5  # Record 5 seconds
   play test.wav             # Play back recording
   ```

### Quick Fixes

1. **Restart Browser**
   - Completely quit and restart your browser
   - This releases microphone locks from crashed processes

2. **Check Activity Monitor (macOS)**
   ```bash
   # Find processes using microphone
   lsof | grep -i "microphone"
   
   # Or check coreaudiod
   ps aux | grep coreaudiod
   ```

3. **Reset Core Audio (macOS)**
   ```bash
   # Kill coreaudiod to force restart
   sudo killall coreaudiod
   ```

4. **Browser-Specific Solutions**
   
   **Chrome:**
   - Navigate to: chrome://settings/content/microphone
   - Remove localhost and re-add it
   - Ensure "Default" microphone is selected
   
   **Safari:**
   - Safari → Preferences → Websites → Microphone
   - Set localhost to "Allow"
   
   **Firefox:**
   - Click the lock icon in address bar
   - Clear permissions and reload

5. **Test Microphone in Browser Console**
   ```javascript
   // Paste this in browser console to test mic access
   navigator.mediaDevices.getUserMedia({ audio: true })
     .then(stream => {
       console.log('✅ Microphone access granted');
       stream.getTracks().forEach(track => track.stop());
     })
     .catch(err => console.error('❌ Microphone error:', err));
   ```

## Other Common Issues

### WebSocket Connection Failed

1. **Check if backend is running**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify WebSocket endpoint**
   ```bash
   # Test WebSocket connection
   websocat ws://localhost:8000/ws/jarvis
   ```

### Voice Recognition Not Working

1. **Check browser compatibility**
   - Chrome/Edge: Full support
   - Safari: Limited support
   - Firefox: No native support

2. **Enable Web Speech API**
   - Chrome: chrome://flags/#enable-experimental-web-platform-features

### JARVIS Not Responding

1. **Check API key**
   ```bash
   echo $ANTHROPIC_API_KEY
   ```

2. **Verify backend logs**
   ```bash
   # Check for errors in backend
   tail -f backend/logs/jarvis.log
   ```

3. **Test API endpoint**
   ```bash
   curl -X POST http://localhost:8000/jarvis/command \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello JARVIS"}'
   ```

## Debug Mode

Enable debug mode for detailed logging:

```javascript
// In browser console
localStorage.setItem('JARVIS_DEBUG', 'true');
location.reload();
```

## Contact Support

If issues persist:
1. Check GitHub Issues: https://github.com/anthropics/claude-code/issues
2. Include:
   - Browser and version
   - Operating system
   - Error messages from console
   - Steps to reproduce