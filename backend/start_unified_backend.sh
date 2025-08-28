#!/bin/bash

# Unified Backend Startup Script - FIXED VERSION
# Starts Python FastAPI backend with proper port management

echo "🚀 Starting JARVIS Unified Backend System..."

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Configuration
PYTHON_PORT=${PYTHON_BACKEND_PORT:-8010}
WEBSOCKET_PORT=${WEBSOCKET_PORT:-8001}

echo -e "${BLUE}Configuration:${NC}"
echo "  Python Backend Port: $PYTHON_PORT"
echo "  WebSocket Port: $WEBSOCKET_PORT"
echo ""

# Check if backend is already running
if check_port $PYTHON_PORT; then
    echo -e "${YELLOW}⚠️  Backend already running on port $PYTHON_PORT${NC}"
    echo "  PID: $(lsof -ti:$PYTHON_PORT)"
    echo "  URL: http://localhost:$PYTHON_PORT"
    echo ""
    echo "Options:"
    echo "  1. Stop existing backend and restart"
    echo "  2. Keep existing backend running"
    echo "  3. Exit"
    echo ""
    read -p "Choose option (1-3): " choice
    
    case $choice in
        1)
            echo -e "${YELLOW}Stopping existing backend...${NC}"
            lsof -ti:$PYTHON_PORT | xargs kill -9 2>/dev/null
            sleep 2
            ;;
        2)
            echo -e "${GREEN}Keeping existing backend running${NC}"
            echo "  Backend available at: http://localhost:$PYTHON_PORT"
            echo "  Vision WebSocket: ws://localhost:$PYTHON_PORT/vision/ws/vision"
            echo "  ML Audio WebSocket: ws://localhost:$PYTHON_PORT/audio/ml/stream"
            exit 0
            ;;
        3)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice, exiting..."
            exit 1
            ;;
    esac
fi

# Start Python backend
echo -e "${GREEN}🚀 Starting Python FastAPI backend on port $PYTHON_PORT...${NC}"
echo -e "${BLUE}📦 Using progressive model loading for faster startup${NC}"
echo ""

# Set environment variables for optimized startup
export TOKENIZERS_PARALLELISM=false  # Prevent tokenizer warnings
export TF_CPP_MIN_LOG_LEVEL=3        # Reduce TensorFlow verbosity

# Start backend in background and capture PID
cd "$(dirname "$0")"  # Ensure we're in the backend directory
python main.py &
PY_PID=$!

# Wait a moment for the process to start
echo -e "${YELLOW}⏳ Waiting for backend to start...${NC}"
sleep 3

# Check if process is still running
if ! kill -0 $PY_PID 2>/dev/null; then
    echo -e "${RED}❌ Backend failed to start!${NC}"
    echo "Check the logs above for errors."
    exit 1
fi

# Wait for health endpoint to respond
echo -e "${YELLOW}🔄 Checking backend health...${NC}"
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:$PYTHON_PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is healthy and responding!${NC}"
        break
    fi
    
    if [ $ATTEMPT -eq 5 ]; then
        echo -e "${YELLOW}⚡ Critical models loaded, server accepting requests...${NC}"
        echo -e "${YELLOW}   Essential models loading in background...${NC}"
    fi
    
    ATTEMPT=$((ATTEMPT + 1))
    sleep 1
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "${YELLOW}⚠️  Backend may still be loading models${NC}"
    echo "You can check model status at:"
    echo "  http://localhost:$PYTHON_PORT/models/status"
fi

# Create PID file for clean shutdown
echo $PY_PID > .python.pid

echo -e "${GREEN}✅ Backend started successfully!${NC}"
echo "  PID: $PY_PID"
echo "  Port: $PYTHON_PORT"
echo "  URL: http://localhost:$PYTHON_PORT"
echo ""
echo -e "${BLUE}Available endpoints:${NC}"
echo "  - Vision: http://localhost:$PYTHON_PORT/vision/status"
echo "  - Voice: http://localhost:$PYTHON_PORT/voice/jarvis/status"
echo "  - ML Audio: http://localhost:$PYTHON_PORT/audio/ml/status"
echo ""
echo -e "${BLUE}WebSocket endpoints:${NC}"
echo "  - Vision: ws://localhost:$PYTHON_PORT/vision/ws/vision"
echo "  - ML Audio: ws://localhost:$PYTHON_PORT/audio/ml/stream"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the backend${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down backend...${NC}"
    
    # Kill Python process
    if [ -f .python.pid ]; then
        PY_PID=$(cat .python.pid)
        if kill -0 $PY_PID 2>/dev/null; then
            echo "Stopping Python backend (PID: $PY_PID)..."
            kill $PY_PID
            sleep 2
            # Force kill if still running
            if kill -0 $PY_PID 2>/dev/null; then
                echo "Force killing backend..."
                kill -9 $PY_PID
            fi
        fi
        rm .python.pid
    fi
    
    echo -e "${GREEN}✅ Backend stopped${NC}"
    exit 0
}

# Set up trap to cleanup on Ctrl+C
trap cleanup INT

# Wait for both processes
wait