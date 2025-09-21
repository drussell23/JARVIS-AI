#!/bin/bash

# Voice Unlock Dependencies Installation Script
# ============================================

echo "🎯 Installing JARVIS Voice Unlock Dependencies..."

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected. It's recommended to use a virtual environment."
    echo "   Run: python -m venv venv && source venv/bin/activate"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install system dependencies (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📦 Installing system dependencies for macOS..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install PortAudio for pyaudio
    if ! brew list portaudio &>/dev/null; then
        echo "🔊 Installing PortAudio..."
        brew install portaudio
    else
        echo "✅ PortAudio already installed"
    fi
    
    # Install other dependencies
    echo "📚 Installing other system libraries..."
    brew list libsndfile &>/dev/null || brew install libsndfile
    brew list ffmpeg &>/dev/null || brew install ffmpeg
else
    echo "⚠️  This script is optimized for macOS. Linux users may need to:"
    echo "   - Install portaudio19-dev"
    echo "   - Install libsndfile1-dev"
    echo "   - Install ffmpeg"
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "🐍 Installing Python packages..."

# Core audio processing
pip install numpy>=1.21.0
pip install librosa>=0.9.2
pip install scipy>=1.7.0

# PyAudio with special handling for M1 Macs
echo "🎤 Installing PyAudio..."
if [[ $(uname -m) == 'arm64' ]]; then
    # M1 Mac specific installation
    export CFLAGS="-I/opt/homebrew/include"
    export LDFLAGS="-L/opt/homebrew/lib"
    pip install pyaudio
else
    # Intel Mac
    pip install pyaudio
fi

# Security and encryption
echo "🔐 Installing security packages..."
pip install cryptography>=3.4.8
pip install keyring>=23.5.0

# macOS integration
echo "🍎 Installing macOS integration packages..."
pip install pyobjc-core>=8.0
pip install pyobjc-framework-Cocoa>=8.0
pip install pyobjc-framework-Quartz>=8.0
pip install pyobjc-framework-Security>=8.0

# Testing tools
echo "🧪 Installing testing tools..."
pip install pytest>=7.0.0
pip install pytest-asyncio>=0.18.0
pip install pytest-mock>=3.6.1

# Development tools (optional)
read -p "Install development tools (black, flake8, mypy)? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install black>=22.0.0
    pip install flake8>=4.0.0
    pip install mypy>=0.910
fi

# Verify installations
echo ""
echo "🔍 Verifying installations..."
python -c "import numpy; print(f'✅ NumPy {numpy.__version__}')"
python -c "import librosa; print(f'✅ Librosa {librosa.__version__}')"
python -c "import scipy; print(f'✅ SciPy {scipy.__version__}')"
python -c "import pyaudio; print('✅ PyAudio installed')"
python -c "import cryptography; print('✅ Cryptography installed')"
python -c "import keyring; print('✅ Keyring installed')"

# Test PyObjC
python -c "import Cocoa; print('✅ PyObjC Cocoa installed')"
python -c "import Quartz; print('✅ PyObjC Quartz installed')"

echo ""
echo "✨ Voice Unlock dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Test audio capture: python backend/voice_unlock/utils/audio_capture.py"
echo "2. Run tests: python -m pytest backend/voice_unlock/tests/"
echo "3. Start integration: python backend/start_system.py"