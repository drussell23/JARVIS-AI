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
if check_port 8000; then
    echo "Killing process on port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
fi
if check_port 8001; then
    echo "Killing process on port 8001..."
    lsof -ti:8001 | xargs kill -9 2>/dev/null
fi

# Install TypeScript dependencies if needed
echo -e "${YELLOW}Checking TypeScript dependencies...${NC}"
cd websocket
if [ ! -d "node_modules" ]; then
    echo "Installing TypeScript dependencies..."
    npm install
fi

# Build TypeScript
echo -e "${YELLOW}Building TypeScript WebSocket Router...${NC}"
npm run build

# Start TypeScript WebSocket Router
echo -e "${GREEN}Starting TypeScript WebSocket Router on port 8001...${NC}"
npm start &
TS_PID=$!

# Give it a moment to start
sleep 2

# Go back to backend directory
cd ..

# Start Python backend
echo -e "${GREEN}Starting Python FastAPI backend on port 8000...${NC}"
python main.py &
PY_PID=$!

# Create a PID file for clean shutdown
echo $TS_PID > .typescript.pid
echo $PY_PID > .python.pid

echo -e "${GREEN}✅ Unified Backend System Started!${NC}"
echo "Python Backend PID: $PY_PID (port 8000)"
echo "TypeScript Router PID: $TS_PID (port 8001)"
echo ""
echo "WebSocket endpoints available at:"
echo "  - ws://localhost:8001/ws/vision (unified vision)"
echo "  - ws://localhost:8001/ws/voice (voice commands)"
echo "  - ws://localhost:8001/ws/automation (automation)"
echo "  - ws://localhost:8001/ws (general)"
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