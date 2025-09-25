#!/bin/bash
# Run the voice demo with proper setup

echo "🚀 Running JARVIS Voice Lock/Unlock Demo"
echo "========================================"

# Check if WebSocket server is running
if ! lsof -i:8765 > /dev/null 2>&1; then
    echo "⚠️  Voice unlock WebSocket server not running"
    echo "Starting it now..."
    cd voice_unlock/objc/server
    python websocket_server.py > /tmp/voice_unlock_ws.log 2>&1 &
    cd ../../../
    sleep 2
    echo "✅ WebSocket server started"
fi

# Check if JARVIS is running
if ! lsof -i:8888 > /dev/null 2>&1; then
    echo "⚠️  JARVIS not running on port 8888"
    echo "Please start JARVIS with: python main.py"
    exit 1
fi

echo "✅ All services running"
echo ""

# Run the demo
python demo_full_lock_unlock_flow.py