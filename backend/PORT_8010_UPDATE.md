# Port 8010 Update Summary

## What Changed
We've updated JARVIS backend to use port 8010 as the default (previously 8000) to match the frontend WebSocket expectations.

## Files Updated

1. **main.py**
   - Default port changed from 8000 to 8010
   - Added startup messages showing WebSocket URL

2. **start_backend.py**
   - Sets `BACKEND_PORT=8010` environment variable
   - Starts server on port 8010
   - Shows WebSocket URL in startup message

3. **start_system_parallel.py**
   - Updated health check URL to use port 8010
   - Added WebSocket URL to output

4. **start_jarvis_correct_port.sh**
   - Updated to use port 8010
   - Shows WebSocket URL

## How to Start JARVIS

### Option 1: Simple Start (Recommended)
```bash
cd backend
python start_backend.py
```

### Option 2: Direct Start
```bash
cd backend
python main.py
```
(Will default to port 8010)

### Option 3: Custom Port
```bash
cd backend
python main.py --port 8010
```

## Verify It's Working

1. Start the backend using any method above
2. You should see:
   ```
   🚀 Starting JARVIS Backend
      HTTP:      http://localhost:8010
      WebSocket: ws://localhost:8010/ws
      API Docs:  http://localhost:8010/docs
   ```
3. In your browser console, WebSocket errors should stop
4. Test with: http://localhost:8010/docs

## Frontend Compatibility

The frontend JavaScript expects:
- WebSocket at `ws://localhost:8010/ws`
- API at `http://localhost:8010`

This update ensures everything works seamlessly!

## Troubleshooting

If port 8010 is already in use:
```bash
# Find and kill the process
lsof -ti:8010 | xargs kill -9

# Then restart
cd backend
python start_backend.py
```