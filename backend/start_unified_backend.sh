#!/bin/bash

# Unified Backend Startup Script
# Starts both Python FastAPI backend and TypeScript WebSocket Router

echo "🚀 Starting JARVIS Unified Backend System..."

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Kill any existing processes on our ports
echo -e "${YELLOW}Checking for existing processes...${NC}"
PYTHON_PORT=${PYTHON_BACKEND_PORT:-8010}
WEBSOCKET_PORT=${WEBSOCKET_PORT:-8001}

if check_port $PYTHON_PORT; then
    echo "Killing process on port $PYTHON_PORT..."
    lsof -ti:$PYTHON_PORT | xargs kill -9 2>/dev/null
fi
if check_port $WEBSOCKET_PORT; then
    echo "Killing process on port $WEBSOCKET_PORT..."
    lsof -ti:$WEBSOCKET_PORT | xargs kill -9 2>/dev/null
fi

# Skip TypeScript for now - it's not properly configured
echo -e "${YELLOW}Note: TypeScript WebSocket router disabled (not configured)${NC}"
TS_PID=0

# Start Python backend
echo -e "${GREEN}Starting Python FastAPI backend on port $PYTHON_PORT...${NC}"
python main.py --port $PYTHON_PORT &
PY_PID=$!

# Create a PID file for clean shutdown
echo $TS_PID > .typescript.pid
echo $PY_PID > .python.pid

echo -e "${GREEN}✅ Unified Backend System Started!${NC}"
echo "Python Backend PID: $PY_PID (port $PYTHON_PORT)"
echo "TypeScript Router PID: $TS_PID (port $WEBSOCKET_PORT)"
echo ""
echo "WebSocket endpoints available at:"
echo "  - ws://localhost:$WEBSOCKET_PORT/ws/vision (unified vision)"
echo "  - ws://localhost:$WEBSOCKET_PORT/ws/voice (voice commands)"
echo "  - ws://localhost:$WEBSOCKET_PORT/ws/automation (automation)"
echo "  - ws://localhost:$WEBSOCKET_PORT/ws (general)"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    
    # Kill TypeScript process
    if [ -f .typescript.pid ]; then
        TS_PID=$(cat .typescript.pid)
        kill $TS_PID 2>/dev/null
        rm .typescript.pid
    fi
    
    # Kill Python process
    if [ -f .python.pid ]; then
        PY_PID=$(cat .python.pid)
        kill $PY_PID 2>/dev/null
        rm .python.pid
    fi
    
    echo -e "${GREEN}✅ Servers stopped${NC}"
    exit 0
}

# Set up trap to cleanup on Ctrl+C
trap cleanup INT

# Wait for both processes
wait