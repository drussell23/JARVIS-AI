#!/bin/bash

# Unified Backend Startup Script with WebSocket Router
# Starts both the Python backend and TypeScript WebSocket router

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
PYTHON_PORT=${PYTHON_BACKEND_PORT:-8010}
WEBSOCKET_PORT=${WEBSOCKET_PORT:-8001}
BACKEND_PID_FILE="backend.pid"
WEBSOCKET_PID_FILE="websocket/websocket-router.pid"

echo -e "${PURPLE}🚀 Starting Unified Backend with WebSocket Router${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    
    # Stop Python backend
    if [ -f "$BACKEND_PID_FILE" ]; then
        kill $(cat "$BACKEND_PID_FILE") 2>/dev/null || true
        rm -f "$BACKEND_PID_FILE"
    fi
    
    # Stop WebSocket router
    if [ -f "$WEBSOCKET_PID_FILE" ]; then
        kill $(cat "$WEBSOCKET_PID_FILE") 2>/dev/null || true
        rm -f "$WEBSOCKET_PID_FILE"
    fi
    
    echo -e "${GREEN}✓ Services stopped${NC}"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Step 1: Start Python Backend
echo -e "\n${BLUE}1️⃣  Starting Python Backend on port $PYTHON_PORT...${NC}"

# Check if already running
if lsof -Pi :$PYTHON_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port $PYTHON_PORT already in use, attempting to stop existing process...${NC}"
    lsof -ti:$PYTHON_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start Python backend
cd "$(dirname "$0")"
export PYTHON_BACKEND_PORT=$PYTHON_PORT
export WEBSOCKET_PORT=$WEBSOCKET_PORT

# Start in background
python main.py > backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$BACKEND_PID_FILE"

# Wait for backend to start (increased timeout for model loading)
echo -e "${YELLOW}   Waiting for Python backend to start...${NC}"
for i in {1..60}; do
    if curl -s http://localhost:$PYTHON_PORT/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Python backend started successfully (PID: $BACKEND_PID)${NC}"
        break
    fi
    sleep 1
    echo -n "."
done

if ! curl -s http://localhost:$PYTHON_PORT/health >/dev/null 2>&1; then
    echo -e "\n${RED}❌ Python backend failed to start${NC}"
    echo -e "${RED}Check backend.log for errors${NC}"
    tail -20 backend.log
    exit 1
fi

# Step 2: Start WebSocket Router
echo -e "\n${BLUE}2️⃣  Starting TypeScript WebSocket Router on port $WEBSOCKET_PORT...${NC}"

cd websocket

# Check if already running
if lsof -Pi :$WEBSOCKET_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port $WEBSOCKET_PORT already in use, attempting to stop existing process...${NC}"
    lsof -ti:$WEBSOCKET_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}   Installing WebSocket router dependencies...${NC}"
    npm install
fi

# Build TypeScript
echo -e "${YELLOW}   Building TypeScript...${NC}"
npm run build

# Start WebSocket router
export PYTHON_BACKEND_URL="http://localhost:$PYTHON_PORT"
nohup node dist/server.js > websocket-router.log 2>&1 &
WEBSOCKET_PID=$!
echo $WEBSOCKET_PID > "websocket-router.pid"

# Wait for WebSocket router to start
echo -e "${YELLOW}   Waiting for WebSocket router to start...${NC}"
sleep 3

if ps -p $WEBSOCKET_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✅ WebSocket router started successfully (PID: $WEBSOCKET_PID)${NC}"
else
    echo -e "${RED}❌ WebSocket router failed to start${NC}"
    echo -e "${RED}Check websocket/websocket-router.log for errors${NC}"
    tail -20 websocket-router.log
    exit 1
fi

cd ..

# Step 3: Verify Services
echo -e "\n${BLUE}3️⃣  Verifying Services...${NC}"

# Test Python backend
if curl -s http://localhost:$PYTHON_PORT/health >/dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Python backend: http://localhost:$PYTHON_PORT${NC}"
else
    echo -e "${RED}   ❌ Python backend not responding${NC}"
fi

# Test WebSocket router
if timeout 2 bash -c "echo '' | nc localhost $WEBSOCKET_PORT" 2>/dev/null; then
    echo -e "${GREEN}   ✅ WebSocket router: ws://localhost:$WEBSOCKET_PORT${NC}"
else
    echo -e "${RED}   ❌ WebSocket router not responding${NC}"
fi

# Show available routes
echo -e "\n${PURPLE}📍 Available WebSocket Routes:${NC}"
echo -e "   ${BLUE}ws://localhost:$WEBSOCKET_PORT/ws/vision${NC} - Vision & monitoring"
echo -e "   ${BLUE}ws://localhost:$WEBSOCKET_PORT/ws/voice${NC} - Voice commands"
echo -e "   ${BLUE}ws://localhost:$WEBSOCKET_PORT/ws/automation${NC} - Task automation"
echo -e "   ${BLUE}ws://localhost:$WEBSOCKET_PORT/ws/ml_audio${NC} - ML audio processing"
echo -e "   ${BLUE}ws://localhost:$WEBSOCKET_PORT/ws${NC} - General WebSocket"

echo -e "\n${GREEN}✨ All services started successfully!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}\n"

# Show logs
echo -e "${PURPLE}📋 Logs:${NC}"
echo -e "   Python backend: $(pwd)/backend.log"
echo -e "   WebSocket router: $(pwd)/websocket/websocket-router.log"
echo ""

# Keep script running and show logs
tail -f backend.log websocket/websocket-router.log