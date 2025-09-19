# WebSocket Connection Failed - Debug Guide

## Error: `WebSocket connection to 'ws://localhost:8010/voice/jarvis/stream' failed`

This error means the frontend (JarvisVoice.js) cannot connect to the backend WebSocket endpoint.

## Common Causes & Solutions

### 1. Backend Not Running on Port 8010

**Check if backend is running:**
```bash
# Check if anything is listening on port 8010
lsof -i :8010

# Or check all Python processes
ps aux | grep python
```

**Solution:**
```bash
# Start the backend on correct port
cd backend
python main.py --port 8010

# OR if using default port 8000
python main.py
```

### 2. Port Mismatch (Most Likely Issue)

The frontend expects port `8010` but backend might be on `8000`.

**Quick Fix in JarvisVoice.js:**
```javascript
// Change this:
const ws = new WebSocket('ws://localhost:8010/voice/jarvis/stream');

// To this:
const ws = new WebSocket('ws://localhost:8000/voice/jarvis/stream');
```

**OR change backend port:**
```bash
python main.py --port 8010
```

### 3. WebSocket Endpoint Not Available

**Check if endpoint exists:**
```bash
# Test with curl
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: test" \
  http://localhost:8000/voice/jarvis/stream
```

**Check main.py for the route:**
```python
# Should have something like:
@app.websocket("/voice/jarvis/stream")
async def voice_stream_endpoint(websocket: WebSocket):
    # WebSocket handling code
```

### 4. CORS or Security Issues

**Add CORS middleware in main.py:**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 5. Backend Crashed or Errored

**Check backend logs:**
```bash
# Look for errors in terminal where main.py is running
# Common errors:
# - Import errors
# - Missing dependencies
# - API key not set
```

## Quick Diagnostic Steps

### Step 1: Verify Backend Status
```bash
# Check if backend is running and healthy
curl http://localhost:8000/
# Should return: {"status": "JARVIS API is running", ...}
```

### Step 2: Test WebSocket Endpoint
```bash
# Install websocat if needed
brew install websocat

# Test WebSocket connection
websocat ws://localhost:8000/voice/jarvis/stream
```

### Step 3: Check Browser Console
```javascript
// In browser console, test connection directly
const testWs = new WebSocket('ws://localhost:8000/voice/jarvis/stream');
testWs.onopen = () => console.log('Connected!');
testWs.onerror = (e) => console.error('Error:', e);
testWs.onclose = (e) => console.log('Closed:', e);
```

## Most Likely Fix

**1. Start the backend:**
```bash
cd /Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend
python main.py
```

**2. Verify it's running:**
```bash
# In another terminal
curl http://localhost:8000/
```

**3. Update frontend if needed:**
- Change port from 8010 to 8000 in JarvisVoice.js
- OR configure backend to use port 8010

## Complete Working Setup

### Backend (main.py):
```python
# Ensure WebSocket endpoint exists
@app.websocket("/voice/jarvis/stream")
async def voice_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    voice_handler = JARVISVoiceHandler()
    try:
        await voice_handler.handle_stream(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
```

### Frontend (JarvisVoice.js):
```javascript
// Use correct port
const wsUrl = 'ws://localhost:8000/voice/jarvis/stream';
const ws = new WebSocket(wsUrl);

ws.onopen = () => {
    console.log('WebSocket connected to JARVIS');
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    // Implement reconnection logic
};
```

## If Still Failing

1. **Check firewall:** Ensure localhost connections aren't blocked
2. **Try 127.0.0.1:** Instead of localhost
3. **Check for proxy:** Disable any proxy settings
4. **Restart everything:** Kill all Python processes and restart

The error is almost certainly because:
1. Backend isn't running
2. Port mismatch (8010 vs 8000)
3. WebSocket endpoint not implemented

Start with verifying the backend is running on the expected port!