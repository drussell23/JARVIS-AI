#!/bin/bash
# Start Voice Unlock System
# ========================

echo "🚀 Starting JARVIS Voice Unlock System"
echo "====================================="
echo

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OBJC_DIR="$SCRIPT_DIR/objc"

# Kill any existing processes
echo "🔄 Cleaning up existing processes..."
pkill -f JARVISVoiceUnlockDaemon
pkill -f websocket_server.py
sleep 1

# Start WebSocket server
echo "📡 Starting WebSocket server on port 8765..."
cd "$OBJC_DIR"
python3 server/websocket_server.py > /tmp/voice_unlock_ws.log 2>&1 &
WS_PID=$!
echo "   WebSocket server PID: $WS_PID"

# Wait for server to start
sleep 2

# Check if server started
if ! nc -z localhost 8765; then
    echo "❌ WebSocket server failed to start!"
    cat /tmp/voice_unlock_ws.log
    exit 1
fi

echo "✅ WebSocket server is running"
echo
echo "📋 Voice Unlock System Ready!"
echo "=============================="
echo
echo "The system is now ready to accept voice unlock commands through JARVIS:"
echo "  • 'voice unlock status' - Check system status"
echo "  • 'enable voice unlock' - Start monitoring"
echo "  • 'disable voice unlock' - Stop monitoring"
echo "  • 'test voice unlock' - Test the system"
echo
echo "WebSocket API available at: ws://localhost:8765/voice-unlock"
echo
echo "Press Ctrl+C to stop the system"
echo

# Keep script running
trap "echo 'Shutting down...'; kill $WS_PID; pkill -f JARVISVoiceUnlockDaemon; exit 0" INT TERM

# Monitor processes
while true; do
    if ! kill -0 $WS_PID 2>/dev/null; then
        echo "❌ WebSocket server stopped unexpectedly"
        exit 1
    fi
    sleep 5
done