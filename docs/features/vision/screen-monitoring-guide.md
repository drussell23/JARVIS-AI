# Screen Monitoring Guide for JARVIS

## ✅ Screen Monitoring is Working!

JARVIS can now monitor your screen continuously at 30 FPS using advanced video streaming technology.

## How to Use Screen Monitoring

### Starting Monitoring
Say any of these commands:
- "Hey JARVIS, start monitoring my screen"
- "Hey JARVIS, monitor my screen"
- "Hey JARVIS, enable screen monitoring"
- "Hey JARVIS, watch my screen"

### What Happens
1. **Purple Indicator**: macOS will show a purple recording indicator in your menu bar
2. **30 FPS Capture**: JARVIS captures your screen at 30 frames per second
3. **Real-time Analysis**: Can detect changes and important events
4. **Swift Technology**: Uses native Swift video capture for best performance

### Stopping Monitoring
- "Hey JARVIS, stop monitoring"
- "Hey JARVIS, stop watching my screen"
- "Hey JARVIS, disable screen monitoring"

### Response You'll Get
When starting monitoring, JARVIS will respond:
> "I have successfully activated direct Swift video capturing for monitoring your screen. The purple recording indicator should now be visible in your menu bar, confirming that screen recording is active. I'm monitoring continuously at 30 FPS and will watch for any changes or important events on your screen until you tell me to stop."

## Requirements

1. **Screen Recording Permission**: Must be granted to your terminal or IDE
2. **macOS**: Works best on macOS with native Swift capture
3. **Memory**: Requires ~100-200MB for video buffer

## Troubleshooting

### No Purple Indicator?
- Check System Preferences → Privacy & Security → Screen Recording
- Ensure your terminal/IDE has permission enabled

### JARVIS Not Responding?
- Make sure to use wake word: "Hey JARVIS"
- Ensure backend is running: `python start_system.py`
- Check Claude API key is configured

### Performance Issues?
- Video streaming is optimized for low CPU usage
- If experiencing lag, try stopping other screen recording apps

## Technical Details
- **Capture Method**: Direct Swift video capture
- **Frame Rate**: 30 FPS
- **Memory Safe**: Automatic memory management
- **Fallback Options**: macOS native capture, screenshot mode

## Example Usage Session
```
You: "Hey JARVIS, start monitoring my screen"
JARVIS: *starts monitoring, purple indicator appears*

You: "What do you see?"
JARVIS: "I can see you have Visual Studio Code open with Python code..."

You: "Hey JARVIS, stop monitoring"
JARVIS: "I've stopped monitoring your screen..."
```