#!/bin/bash

echo "🔧 Fixing JARVIS offline issue..."

# 1. Ensure API key is exported
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ANTHROPIC_API_KEY not set!"
    echo "Please run: export ANTHROPIC_API_KEY='your-key-here'"
    exit 1
fi

echo "✅ API key is set"

# 2. Restart the backend with the API key
echo "🔄 Restarting backend with API key..."
pkill -f "python.*main.py"
sleep 2

# 3. Start backend with environment variable
cd backend
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY python3 main.py > logs/jarvis_fixed_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"

# 4. Wait for backend to be ready
echo "⏳ Waiting for backend..."
sleep 5

# 5. Check health
curl -s http://localhost:8010/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# 6. Activate JARVIS
echo "🤖 Activating JARVIS..."
curl -X POST http://localhost:8010/voice/jarvis/activate -H "Content-Type: application/json" -d '{}' > /dev/null 2>&1

# 7. Check JARVIS status
STATUS=$(curl -s http://localhost:8010/voice/jarvis/status | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
echo "📊 JARVIS status: $STATUS"

if [ "$STATUS" = "online" ] || [ "$STATUS" = "standby" ]; then
    echo "✅ JARVIS is ready!"
else
    echo "⚠️  JARVIS status is: $STATUS"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Refresh the frontend (http://localhost:3000)"
echo "2. JARVIS should now show as ONLINE"
echo "3. Try: 'Hey JARVIS, start monitoring my screen'"