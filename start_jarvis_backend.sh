#!/bin/bash

# JARVIS Backend Startup Script
# Ensures all services are properly initialized

echo "🤖 Starting JARVIS Backend Services..."
echo "========================================"

# Navigate to backend directory
cd backend

# Check for .env file
if [ ! -f .env ]; then
    echo "❌ ERROR: backend/.env file not found!"
    echo "Please create backend/.env with:"
    echo "ANTHROPIC_API_KEY=your-api-key-here"
    exit 1
fi

# Check if ANTHROPIC_API_KEY is set
if ! grep -q "ANTHROPIC_API_KEY=" .env; then
    echo "❌ ERROR: ANTHROPIC_API_KEY not found in .env!"
    echo "Please add: ANTHROPIC_API_KEY=your-api-key-here"
    exit 1
fi

echo "✅ Environment configuration found"

# Check Python dependencies
echo ""
echo "📦 Checking Python dependencies..."
python -c "import fastapi, websockets, anthropic" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing..."
    pip install -r requirements.txt
fi

# Kill any existing backend process
echo ""
echo "🔄 Checking for existing backend process..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Killed existing backend process"
    sleep 1
fi

# Start the backend
echo ""
echo "🚀 Starting JARVIS Backend on port 8000..."
echo "========================================"
echo ""
echo "Backend will include:"
echo "  ✅ JARVIS Voice API"
echo "  ✅ Vision WebSocket (/vision/ws/vision)"
echo "  ✅ Autonomy Handler"
echo "  ✅ Notification Intelligence"
echo "  ✅ Navigation API"
echo ""
echo "Press Ctrl+C to stop the backend"
echo ""

# Start with explicit host and port
python main.py --host 0.0.0.0 --port 8000