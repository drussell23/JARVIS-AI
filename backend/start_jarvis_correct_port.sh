#!/bin/bash
# Start JARVIS on the correct default port

cd ~/Documents/repos/JARVIS-AI-Agent/backend

echo "🚀 Starting JARVIS on default port 8000..."
python main.py &

echo "✅ JARVIS started!"
echo ""
echo "Now go to: http://localhost:8000"
echo "And try: 'lock my screen'"