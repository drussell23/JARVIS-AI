#!/bin/bash

echo "Stopping AI-Powered Chatbot System..."

# Function to stop a service
stop_service() {
    local name=$1
    local pid_file="logs/${name}.pid"
    
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "✓ Stopped $name"
        fi
        rm "$pid_file"
    fi
}

# Stop all services
stop_service "main_api"
stop_service "training_api"
stop_service "frontend"

# Additional cleanup
pkill -f "python.*main.py" 2>/dev/null
pkill -f "python.*training_interface.py" 2>/dev/null
pkill -f "http.server.*3000" 2>/dev/null

echo "✓ All services stopped"
