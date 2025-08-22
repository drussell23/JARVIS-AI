#!/bin/bash
# Start backend in detached mode

cd "$(dirname "$0")"

# Kill any existing backend process
pkill -f "uvicorn main:app" || true
pkill -f "python.*start_backend" || true

# Start backend in background
echo "Starting JARVIS backend..."
nohup python start_backend.py > backend.log 2>&1 &
BACKEND_PID=$!

echo "Backend starting with PID: $BACKEND_PID"
echo "Waiting for backend to be ready..."

# Wait for backend to be ready
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is running and ready!"
        echo "Log file: backend/backend.log"
        echo "PID: $BACKEND_PID"
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "❌ Backend failed to start. Check backend/backend.log for errors."
exit 1