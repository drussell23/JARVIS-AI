# JARVIS Audio Setup & Troubleshooting Guide

## Overview
JARVIS uses Daniel's British voice (macOS) for all audio output. The backend generates audio using the macOS `say` command and serves it to the frontend via HTTP endpoints.

## Current Setup

### Backend Configuration
- **Voice**: Daniel (British male voice)
- **Endpoints**: 
  - GET: `/audio/speak/{text}` - For short messages
  - POST: `/audio/speak` - For longer messages
- **Format**: MP3 (with fallbacks to WAV/AIFF)
- **Generation**: Uses macOS `say` command

### Frontend Configuration
- Enhanced error handling and logging
- Automatic fallback from GET to POST method
- Browser speech synthesis fallback with Daniel voice preference
- CORS enabled with `crossOrigin: 'anonymous'`

## Troubleshooting Steps

### 1. Test Backend Audio Generation
```bash
python test_jarvis_audio.py
```
This will test:
- GET and POST endpoints
- Daniel voice availability
- CORS headers
- Audio file generation

### 2. Debug in Browser
Open http://localhost:3000/debug-audio.html to:
- Test direct audio playback
- Simulate React component behavior
- Check browser Daniel voice
- Test WebSocket audio responses

### 3. Check Browser Console
Look for messages starting with `[JARVIS Audio]`:
- Loading started
- Can play through
- Playback started/completed
- Error details with codes

### 4. Common Issues & Fixes

#### No Audio Heard
1. Check browser console for errors
2. Verify backend is running on port 8000
3. Test with debug-audio.html page
4. Check system volume and mute settings

#### CORS Errors
- Backend automatically sets CORS headers
- Frontend uses `crossOrigin: 'anonymous'`
- Verify allowed origins include your frontend URL

#### Voice Not British/Daniel
- Backend always uses Daniel voice
- Browser fallback searches for Daniel first
- Then tries other British voices (Oliver, James)
- Falls back to any en-GB voice

#### Audio Delays
- Short messages (<500 chars) use GET method (faster)
- Long messages use POST method
- Browser speech synthesis used as last resort

## Testing Voice Commands

### Quick Test
```bash
# Test Daniel voice directly
say -v Daniel "Hello Sir, this is JARVIS"

# List available voices
say -v ?
```

### Frontend Test
1. Start the app: `npm start`
2. Click "Activate JARVIS"
3. Say "Hey JARVIS"
4. Give a command like "Hello" or "What time is it?"
5. Check browser console for `[JARVIS Audio]` messages

## Audio Flow

1. User gives voice command
2. Backend processes and generates response
3. Backend uses `say -v Daniel` to create audio file
4. Audio converted to MP3 (or WAV/AIFF fallback)
5. Frontend receives audio via HTTP
6. Audio played using HTML5 Audio API
7. Fallback to browser speech if needed

## Configuration Files

### Backend
- `backend/api/jarvis_voice_api.py` - Audio generation endpoints
- `backend/voice/macos_voice.py` - macOS voice integration

### Frontend  
- `frontend/src/components/JarvisVoice.js` - Main voice component
- `frontend/src/utils/audioHelper.js` - Audio utilities (if needed)

## Logs to Monitor

### Backend
```
[JARVIS API] Processing command: 'hello jarvis'
[JARVIS API] Response: 'Good evening, Sir...'
```

### Frontend
```
[JARVIS Audio] Speaking response: Good evening, Sir...
[JARVIS Audio] Using GET method: http://localhost:8000/audio/speak/...
[JARVIS Audio] Loading started
[JARVIS Audio] Can play through
[JARVIS Audio] Playback started
[JARVIS Audio] Playback completed
```

## Next Steps

1. If audio still doesn't work:
   - Run `python test_jarvis_audio.py`
   - Open debug-audio.html in browser
   - Check for JavaScript errors in console
   - Verify microphone permissions

2. To customize voice:
   - Modify Daniel to another voice in `jarvis_voice_api.py`
   - Update browser fallback voices in `JarvisVoice.js`

3. To improve performance:
   - Enable audio caching
   - Preload common responses
   - Use audio streaming for long texts