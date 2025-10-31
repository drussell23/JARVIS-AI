#!/bin/bash

echo "🔧 FIXING JARVIS STT TRANSCRIPTION"
echo "=================================="

# Step 1: Update JARVIS configuration
echo "📝 Creating STT configuration..."
cat > jarvis_config_stt.json <<EOF
{
  "stt": {
    "engine": "whisper",
    "model": "base",
    "language": "en",
    "sample_rate": 16000,
    "chunk_duration": 3,
    "energy_threshold": 1000,
    "pause_threshold": 0.8
  },
  "voice_recognition": {
    "enabled": true,
    "speaker": "Derek J. Russell",
    "confidence_threshold": 75
  }
}
EOF

echo "✅ Configuration created"

# Step 2: Kill existing JARVIS
echo "🔄 Restarting JARVIS..."
pkill -f "jarvis" 2>/dev/null || true
sleep 2

# Step 3: Start JARVIS with new config
echo "🚀 Starting JARVIS with Whisper STT..."
python3 start_system.py --config jarvis_config_stt.json &

echo ""
echo "✅ JARVIS STT FIX APPLIED!"
echo ""
echo "📢 Wait 30 seconds for initialization, then try:"
echo "   'Hey JARVIS, unlock my screen'"
echo ""
echo "Expected behavior:"
echo "  1. JARVIS hears: 'unlock my screen' (not '[transcription failed]')"
echo "  2. JARVIS recognizes your voice as Derek J. Russell"
echo "  3. JARVIS responds: 'Of course, Derek'"
echo ""