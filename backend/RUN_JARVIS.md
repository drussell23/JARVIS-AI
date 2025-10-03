# How to Start JARVIS Backend

## Quick Start

From the backend directory, run:

```bash
cd backend
python start_backend.py
```

Or directly:

```bash
cd backend
python main.py --port 8010
```

## The Backend Will:
- Start on port 8010
- WebSocket available at ws://localhost:8010/ws
- API available at http://localhost:8010

## Verify It's Running

1. Check the terminal - you should see:
   ```
   🚀 Starting FastAPI server on port 8010...
   WebSocket will be available at ws://localhost:8010/ws
   ```

2. In your browser console, the WebSocket errors should stop

3. Try saying "Hey JARVIS" or "Lock my screen"

## If Port 8010 is Already in Use

Kill any existing process:
```bash
lsof -ti:8010 | xargs kill -9
```

Then restart the backend.

## Environment Variables

Make sure you have your API key set:
```bash
export ANTHROPIC_API_KEY='your-key-here'
```

## Troubleshooting

If you still see WebSocket errors:
1. Make sure the backend is actually running
2. Check no firewall is blocking port 8010
3. Try accessing http://localhost:8010 in your browser - you should see the JARVIS API page