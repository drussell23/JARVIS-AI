#!/bin/bash

# Unified WebSocket Router Startup Script
# Ensures robust startup with proper error handling and health checks

set -e  # Exit on error

echo "🚀 Starting Unified WebSocket Router..."

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WEBSOCKET_PORT=${WEBSOCKET_PORT:-8001}
PYTHON_BACKEND_URL=${PYTHON_BACKEND_URL:-"http://localhost:8010"}
LOG_FILE="websocket-router.log"
PID_FILE="websocket-router.pid"

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}❌ Port $port is already in use${NC}"
        
        # Try to find what's using the port
        echo -e "${YELLOW}Process using port $port:${NC}"
        lsof -Pi :$port -sTCP:LISTEN
        
        return 1
    else
        echo -e "${GREEN}✅ Port $port is available${NC}"
        return 0
    fi
}

# Function to check if process is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to stop existing process
stop_existing() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        echo -e "${YELLOW}Stopping existing WebSocket router (PID: $pid)...${NC}"
        kill $pid 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        if ps -p $pid > /dev/null 2>&1; then
            kill -9 $pid 2>/dev/null || true
        fi
        
        rm -f "$PID_FILE"
    fi
}

# Function to install dependencies
install_dependencies() {
    if [ ! -d "node_modules" ]; then
        echo -e "${BLUE}Installing dependencies...${NC}"
        npm install
    fi
}

# Function to build TypeScript
build_typescript() {
    echo -e "${BLUE}Building TypeScript...${NC}"
    npm run build
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ TypeScript build successful${NC}"
    else
        echo -e "${RED}❌ TypeScript build failed${NC}"
        exit 1
    fi
}

# Main startup sequence
main() {
    # Change to script directory
    cd "$(dirname "$0")"
    
    # Stop any existing process
    stop_existing
    
    # Check port availability
    if ! check_port $WEBSOCKET_PORT; then
        echo -e "${RED}Cannot start WebSocket router - port $WEBSOCKET_PORT is in use${NC}"
        exit 1
    fi
    
    # Install dependencies if needed
    install_dependencies
    
    # Build TypeScript
    build_typescript
    
    # Set environment variables
    export WEBSOCKET_PORT=$WEBSOCKET_PORT
    export PYTHON_BACKEND_URL=$PYTHON_BACKEND_URL
    export NODE_ENV=${NODE_ENV:-development}
    export ENABLE_DYNAMIC_ROUTING=true
    export ENABLE_RATE_LIMIT=true
    
    echo -e "${BLUE}Starting WebSocket router on port $WEBSOCKET_PORT...${NC}"
    echo -e "${BLUE}Python backend URL: $PYTHON_BACKEND_URL${NC}"
    
    # Start the server
    nohup node dist/server.js > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"
    
    # Wait a moment for startup
    sleep 2
    
    # Check if process is still running
    if ps -p $pid > /dev/null 2>&1; then
        echo -e "${GREEN}✅ WebSocket router started successfully (PID: $pid)${NC}"
        echo -e "${GREEN}📋 Log file: $LOG_FILE${NC}"
        
        # Show initial logs
        echo -e "${BLUE}Initial logs:${NC}"
        tail -n 10 "$LOG_FILE"
        
        # Test connection
        echo -e "${BLUE}Testing WebSocket connection...${NC}"
        if command -v wscat &> /dev/null; then
            timeout 2 wscat -c ws://localhost:$WEBSOCKET_PORT/ws 2>/dev/null && {
                echo -e "${GREEN}✅ WebSocket connection test successful${NC}"
            } || {
                echo -e "${YELLOW}⚠️  Could not test WebSocket connection${NC}"
            }
        else
            echo -e "${YELLOW}wscat not installed - skipping connection test${NC}"
        fi
        
    else
        echo -e "${RED}❌ Failed to start WebSocket router${NC}"
        echo -e "${RED}Check the log file for errors: $LOG_FILE${NC}"
        tail -n 20 "$LOG_FILE"
        exit 1
    fi
}

# Run main function
main "$@"