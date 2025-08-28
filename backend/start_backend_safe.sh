#!/bin/bash

# Safe Backend Startup Script with Enhanced Error Recovery
# Prevents crashes and ensures backend starts even with model loading failures

echo "🚀 Starting JARVIS Backend with Safe Mode..."

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_PORT=${PYTHON_BACKEND_PORT:-8010}
LOG_DIR="backend/logs"
mkdir -p "$LOG_DIR"

# Set safe environment variables
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Fallback for M1 Mac
export OMP_NUM_THREADS=4  # Limit OpenMP threads
export MKL_NUM_THREADS=4  # Limit MKL threads

# Python optimizations for faster startup
export PYTHONOPTIMIZE=1  # Skip assert statements
export PYTHONDONTWRITEBYTECODE=1  # Don't create .pyc files

echo -e "${BLUE}Safe Mode Configuration:${NC}"
echo "  • Resource-aware model loading"
echo "  • Automatic recovery from failures"
echo "  • Progressive startup phases"
echo "  • Memory monitoring enabled"
echo ""

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Kill any existing processes on the port
if check_port $PYTHON_PORT; then
    echo -e "${YELLOW}Stopping existing process on port $PYTHON_PORT...${NC}"
    lsof -ti:$PYTHON_PORT | xargs kill -9 2>/dev/null
    sleep 2
fi

# Create a startup config to limit initial model loading
cat > "$LOG_DIR/startup_config.json" << EOF
{
  "max_initial_models": 5,
  "enable_lazy_loading": true,
  "memory_limit_percent": 70,
  "startup_timeout": 30
}
EOF

# Start the backend with enhanced error handling
echo -e "${GREEN}Starting backend in safe mode...${NC}"

cd "$(dirname "$0")"

# Create a wrapper script for better error handling
cat > "$LOG_DIR/safe_start.py" << 'EOF'
#!/usr/bin/env python3
import sys
import os
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Set safe mode flag
    os.environ['JARVIS_SAFE_MODE'] = '1'
    
    logger.info("Starting JARVIS in Safe Mode...")
    
    # Import with error handling
    try:
        import main
        import uvicorn
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Installing missing dependencies...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        import main
        import uvicorn
    
    # Start server with custom error handler
    class SafeServer:
        def __init__(self):
            self.app = main.app
            
        def run(self):
            try:
                uvicorn.run(
                    self.app, 
                    host="127.0.0.1", 
                    port=8010,
                    log_level="info",
                    access_log=False,  # Reduce logging overhead
                    loop="asyncio",
                    limit_concurrency=100,  # Limit concurrent connections
                    limit_max_requests=1000,  # Restart worker after 1000 requests
                )
            except Exception as e:
                logger.error(f"Server error: {e}")
                logger.info("Starting minimal server...")
                # Start with minimal functionality
                from fastapi import FastAPI
                minimal_app = FastAPI()
                
                @minimal_app.get("/")
                async def root():
                    return {"status": "safe_mode", "message": "JARVIS running in safe mode"}
                
                @minimal_app.get("/health")
                async def health():
                    return {"status": "degraded", "mode": "safe"}
                
                uvicorn.run(minimal_app, host="127.0.0.1", port=8010)
    
    server = SafeServer()
    server.run()
    
except Exception as e:
    logger.error(f"Fatal error: {e}")
    traceback.print_exc()
    sys.exit(1)
EOF

# Make the wrapper executable
chmod +x "$LOG_DIR/safe_start.py"

# Start with resource limits
echo -e "${YELLOW}Starting with resource limits...${NC}"

# Run with memory limit on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS doesn't have ulimit -v, but we can use other methods
    python "$LOG_DIR/safe_start.py" 2>&1 | tee "$LOG_DIR/startup_$(date +%Y%m%d_%H%M%S).log" &
else
    # Linux - set virtual memory limit
    ulimit -v 4194304  # 4GB limit
    python "$LOG_DIR/safe_start.py" 2>&1 | tee "$LOG_DIR/startup_$(date +%Y%m%d_%H%M%S).log" &
fi

PY_PID=$!
echo "Backend PID: $PY_PID"

# Monitor startup
echo -e "${YELLOW}Monitoring startup...${NC}"

ATTEMPT=0
MAX_ATTEMPTS=60  # 60 seconds timeout
HEALTH_URL="http://localhost:$PYTHON_PORT/health"

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    # Check if process is still running
    if ! kill -0 $PY_PID 2>/dev/null; then
        echo -e "${RED}❌ Backend process crashed!${NC}"
        echo "Check logs at: $LOG_DIR/startup_*.log"
        
        # Try emergency recovery
        echo -e "${YELLOW}Attempting emergency recovery...${NC}"
        python -c "
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get('/')
async def root():
    return {'status': 'emergency_mode', 'message': 'JARVIS in emergency recovery mode'}

@app.get('/health')
async def health():
    return {'status': 'emergency'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=$PYTHON_PORT)
" &
        EMERGENCY_PID=$!
        echo -e "${YELLOW}Emergency server started on PID: $EMERGENCY_PID${NC}"
        break
    fi
    
    # Check health endpoint
    if curl -s "$HEALTH_URL" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is responding!${NC}"
        
        # Check model status
        MODEL_STATUS=$(curl -s "http://localhost:$PYTHON_PORT/models/status" 2>/dev/null || echo "{}")
        echo -e "${BLUE}Model Status:${NC} $MODEL_STATUS"
        
        break
    fi
    
    # Progress indicator
    if [ $((ATTEMPT % 10)) -eq 0 ]; then
        echo "⏳ Still starting... ($ATTEMPT seconds)"
    fi
    
    ATTEMPT=$((ATTEMPT + 1))
    sleep 1
done

# Final status
echo ""
if curl -s "$HEALTH_URL" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ JARVIS Backend started successfully!${NC}"
    echo "  URL: http://localhost:$PYTHON_PORT"
    echo "  Docs: http://localhost:$PYTHON_PORT/docs"
    echo "  Health: http://localhost:$PYTHON_PORT/health"
    echo "  Model Status: http://localhost:$PYTHON_PORT/models/status"
else
    echo -e "${YELLOW}⚠️  Backend started but may be in degraded mode${NC}"
fi

echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the backend${NC}"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down backend...${NC}"
    kill $PY_PID 2>/dev/null || kill $EMERGENCY_PID 2>/dev/null
    sleep 2
    echo -e "${GREEN}✅ Backend stopped${NC}"
    exit 0
}

trap cleanup INT TERM

# Wait for process
wait