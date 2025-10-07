#!/bin/bash
# Start JARVIS on the correct default port

cd ~/Documents/repos/JARVIS-AI-Agent/backend

echo "🚀 Starting JARVIS on port 8010..."
python main.py --port 8010 &

echo "✅ JARVIS started!"
echo ""
echo "Now go to: http://localhost:8010"
echo "WebSocket available at: ws://localhost:8010/ws"
echo "And try: 'lock my screen'"