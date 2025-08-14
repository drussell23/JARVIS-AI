#!/bin/bash
echo "Starting AI Chatbot Backend Server..."
cd /Users/derekjrussell/Documents/repos/AI-Powered-Chatbot/backend
source venv/bin/activate

# Set environment variables for M1 optimization
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "Server starting on http://localhost:8000"
echo "Press Ctrl+C to stop"
python main.py