#!/bin/bash
# Start JARVIS on port 8000 with all the latest audio fixes

echo "🚀 Starting JARVIS on port 8000..."
echo "This instance will have all the audio fixes including Daniel's voice"
echo ""

cd /Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend

# Export environment variables
export BACKEND_PORT=8000
export PYTHONUNBUFFERED=1

# Start JARVIS
echo "Starting JARVIS with updated audio system..."
python main.py