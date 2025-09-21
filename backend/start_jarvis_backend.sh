#!/bin/bash
# Start JARVIS Backend Server

echo "🚀 Starting JARVIS Backend Server..."
echo "=================================="

# Change to backend directory
cd /Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Set environment variables
export PYTHONPATH=/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend:$PYTHONPATH

# Start the backend server
echo "🎯 Starting backend on port 8010..."
python main.py

echo "✅ JARVIS Backend is running!"