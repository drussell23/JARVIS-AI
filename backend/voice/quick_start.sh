#!/bin/bash
# Quick start script for JARVIS voice system with Picovoice

echo "🚀 JARVIS Voice System Quick Start"
echo "=================================="

# Set environment variables
export PICOVOICE_ACCESS_KEY="e9AVn4el49rhJxxUDILvK2vYOTbFZx1ZSlSnsZqEu3kLY3Ix8Ghckg=="
export USE_PICOVOICE=true
export WAKE_WORD_THRESHOLD=0.55
export CONFIDENCE_THRESHOLD=0.6
export ENABLE_VAD=true
export ENABLE_STREAMING=true
export JARVIS_OPTIMIZATION_LEVEL=balanced

echo "✅ Environment variables set"

# Check if pvporcupine is installed
python3 -c "import pvporcupine" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing Picovoice..."
    pip install pvporcupine
fi

# Check if webrtcvad is installed
python3 -c "import webrtcvad" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing WebRTC VAD..."
    pip install webrtcvad
fi

echo ""
echo "🎯 Testing Picovoice setup..."
python3 setup_picovoice.py

echo ""
echo "✨ Setup complete!"
echo ""
echo "To use in your code:"
echo "-------------------"
echo "export PICOVOICE_ACCESS_KEY='$PICOVOICE_ACCESS_KEY'"
echo ""
echo "Then in Python:"
echo "from voice.optimized_voice_system import create_optimized_jarvis"
echo "system = await create_optimized_jarvis(api_key, '16gb_macbook_pro')"