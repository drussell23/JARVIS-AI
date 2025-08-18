# JARVIS Voice Frontend Troubleshooting

## Current Status
✅ Voice synthesis working (210 voices available)
✅ British voice selected (Daniel)
✅ WebSocket connected to backend
✅ Continuous listening enabled

## Common Issues and Solutions

### 1. "no-speech" Error
**Status**: Normal behavior
**Explanation**: This occurs when the microphone doesn't detect speech for ~5 seconds
**Solution**: No action needed - this is expected

### 2. "The message port closed" Error
**Status**: Browser extension conflict
**Explanation**: A browser extension is interfering
**Solution**: 
- Try disabling browser extensions temporarily
- Or use an incognito/private window

### 3. "Unchecked runtime.lastError"
**Status**: Minor browser issue
**Explanation**: Chrome extension API warning
**Solution**: Can be safely ignored

## Testing Voice Commands

1. **Check Microphone Permission**
   - Look for microphone icon in address bar
   - Ensure it's allowed for localhost:3000

2. **Test Wake Word**
   - Say clearly: "Hey JARVIS"
   - Wait for the activation response
   - Speak your command

3. **Example Commands**
   ```
   "Hey JARVIS, what's the weather?"
   "Hey JARVIS, open Chrome"
   "Hey JARVIS, set volume to 50%"
   "Hey JARVIS, tell me a joke"
   ```

## Verify Backend Connection

1. Check backend is running:
   ```bash
   curl http://localhost:8000/voice/jarvis/status
   ```

2. Check WebSocket in browser console:
   ```javascript
   // Should show "OPEN"
   console.log(document.querySelector('iframe').contentWindow.ws.readyState)
   ```

## Browser-Specific Tips

### Chrome/Edge
- Allow microphone in Settings → Privacy → Site Settings → Microphone
- Add localhost:3000 to allowed sites

### Safari
- System Preferences → Security & Privacy → Privacy → Microphone
- Check Safari

### Firefox
- Click the microphone icon in address bar
- Select "Allow" for localhost

## Voice Quality Tips

1. **Speak Clearly**: Enunciate "Hey JARVIS" distinctly
2. **Quiet Environment**: Reduce background noise
3. **Microphone Distance**: Stay 6-12 inches from mic
4. **Consistent Volume**: Don't whisper or shout

## Debug Mode

To see more details, open browser console and run:
```javascript
localStorage.setItem('jarvis_debug', 'true');
location.reload();
```

## Still Having Issues?

1. Check backend logs:
   ```bash
   tail -f backend/logs/jarvis.log
   ```

2. Test microphone:
   ```bash
   cd backend
   python test_microphone.py
   ```

3. Verify API key:
   ```bash
   cat backend/.env | grep ANTHROPIC_API_KEY
   ```