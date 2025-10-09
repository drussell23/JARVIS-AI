# Quick Start Guide for JARVIS

## Starting JARVIS Backend

To fix the WebSocket connection error, you need to start the JARVIS backend server:

### Option 1: Using start_system.py (Recommended)
```bash
cd /Users/derekjrussell/Documents/repos/JARVIS-AI-Agent
python start_system.py
```
This starts both the TypeScript WebSocket Router (port 8001) and Python Backend (port 8010).

### Option 2: Backend only
```bash
cd /Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend
python main.py
```

### Option 3: Using the new start script
```bash
cd /Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend
./start_jarvis_backend.sh
```

## Verify Services are Running

After starting, verify the services:

```bash
# Check if backend is running
curl http://localhost:8010/health

# Check JARVIS status
curl http://localhost:8010/voice/jarvis/status

# Check WebSocket endpoints
curl http://localhost:8001/api/websocket/endpoints
```

## Frontend Connection

Once the backend is running, your frontend should automatically connect. You'll see:
- "Connected to ML Audio Backend" ✅
- WebSocket connection established
- JARVIS ready to use

## Testing Weather with API

Now that you have the OpenWeatherMap API configured:
1. Say "Hey JARVIS, what's the weather today?"
2. You should get an instant response with current weather
3. Try city-specific queries: "What's the weather in New York?"

## Troubleshooting

If you still see connection errors:
1. Check if port 8010 is in use: `lsof -i :8010`
2. Check backend logs: `tail -f backend/logs/main_api.log`
3. Ensure all dependencies are installed: `pip install -r backend/requirements.txt`
4. Try restarting both frontend and backend