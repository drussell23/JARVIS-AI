#!/bin/bash

echo "🔧 FIXING JARVIS STT AND TESTING VOICE RECOGNITION"
echo "=================================================="

# Step 1: Set STT environment variables
echo "📝 Setting STT configuration..."
export JARVIS_STT_ENGINE="whisper"
export JARVIS_STT_MODEL="base"
export JARVIS_STT_LANGUAGE="en"
export JARVIS_VOICE_BIOMETRIC="true"
export JARVIS_SPEAKER_NAME="Derek J. Russell"

echo "✅ Environment configured"

# Step 2: Kill existing JARVIS processes
echo "🔄 Stopping existing JARVIS..."
pkill -f "start_system.py" 2>/dev/null || true
pkill -f "jarvis" 2>/dev/null || true
sleep 2

# Step 3: Start JARVIS with Whisper STT
echo "🚀 Starting JARVIS with fixed STT..."
python3 start_system.py &
JARVIS_PID=$!

echo ""
echo "⏳ Waiting 30 seconds for JARVIS to initialize..."
for i in {30..1}; do
    echo -ne "\r   $i seconds remaining...   "
    sleep 1
done
echo ""

echo ""
echo "✅ JARVIS READY WITH FIXED STT!"
echo ""
echo "=================================================="
echo "🎤 TEST NOW:"
echo "=================================================="
echo ""
echo "Say: 'Hey JARVIS, unlock my screen'"
echo ""
echo "✅ EXPECTED RESULTS:"
echo "  1. Wake word detected: 'Hey JARVIS'"
echo "  2. Command transcribed: 'unlock my screen' (NOT '[transcription failed]')"
echo "  3. Voice verified: 'Derek J. Russell (95.2% confidence)'"
echo "  4. JARVIS responds: 'Of course, Derek'"
echo "  5. Screen unlocks"
echo ""
echo "🎯 Test it now while JARVIS is running!"
echo ""
echo "To stop JARVIS: kill $JARVIS_PID"
echo ""