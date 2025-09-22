#!/bin/bash
#
# Install Voice Unlock Dependencies
# =================================
#

echo "🔧 Installing Voice Unlock dependencies..."
echo "========================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. It's recommended to use one."
    echo "   Create with: python3 -m venv venv && source venv/bin/activate"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install core dependencies
echo
echo "📦 Installing core dependencies..."
pip install --upgrade pip

# Install required packages
pip install fastapi uvicorn httpx
pip install anthropic
pip install numpy scipy scikit-learn joblib
pip install librosa sounddevice pyaudio
pip install webrtcvad SpeechRecognition
pip install psutil matplotlib
pip install bleak  # For Apple Watch Bluetooth
pip install cryptography
pip install aiofiles

# Install optional ML optimization packages
echo
echo "📦 Installing optional ML optimization packages..."
read -p "Install TensorFlow for macOS? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if on Apple Silicon
    if [[ $(uname -m) == 'arm64' ]]; then
        pip install tensorflow-macos tensorflow-metal
    else
        pip install tensorflow
    fi
fi

read -p "Install ONNX optimization tools? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install onnx onnxruntime
fi

# Create necessary directories
echo
echo "📁 Creating necessary directories..."
mkdir -p ~/.jarvis/voice_unlock/{models,logs,audit}
mkdir -p ~/.jarvis/voice_unlock/proximity_voice_auth/models

# Set permissions
chmod -R 755 ~/.jarvis/voice_unlock

# Test imports
echo
echo "🧪 Testing imports..."
python3 -c "
import fastapi
import anthropic
import sklearn
import librosa
import sounddevice
import bleak
print('✅ All core imports successful!')
"

if [ $? -ne 0 ]; then
    echo "❌ Some imports failed. Please check the errors above."
    exit 1
fi

# Check voice unlock module
echo
echo "🔍 Checking Voice Unlock module..."
python3 -c "
from backend.voice_unlock import check_dependencies
deps = check_dependencies()
print('\nDependency Status:')
for dep, available in deps.items():
    status = '✅' if available else '❌'
    print(f'  {status} {dep}')
"

echo
echo "✅ Voice Unlock dependencies installed!"
echo
echo "Next steps:"
echo "1. Run: jarvis-voice-unlock --help"
echo "2. Enroll your voice: jarvis-voice-unlock enroll <your_name>"
echo "3. Test authentication: jarvis-voice-unlock test"
echo
echo "For Apple Watch integration:"
echo "- Ensure Bluetooth is enabled"
echo "- Your Apple Watch should be paired with your Mac"