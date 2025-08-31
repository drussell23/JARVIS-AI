#!/bin/bash
# Start just the WebSocket router for testing

echo "🚀 Starting WebSocket Router..."
cd backend/websocket

# Build first
echo "Building TypeScript..."
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Build successful"
    echo "Starting WebSocket router on port 8001..."
    npm start
else
    echo "❌ Build failed"
    exit 1
fi