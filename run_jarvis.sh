#!/bin/bash
# Simple JARVIS launcher that bypasses dependency checks

echo "🚀 Starting JARVIS (Simple Mode)"
echo "================================"

# Set environment variables
export USE_QUANTIZED_MODELS=true
export PREFER_LANGCHAIN=0
export PYTHONPATH="$PWD/backend:$PYTHONPATH"

# Change to backend directory
cd backend

# Start the server
echo "Starting backend server..."
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Note: Press Ctrl+C to stop