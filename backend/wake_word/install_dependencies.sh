#!/bin/bash

echo "🎤 Installing Wake Word Dependencies..."
echo "======================================"

# Check if we're in the right directory
if [ ! -f "config.py" ]; then
    echo "❌ Error: Please run this script from the wake_word directory"
    echo "   cd backend/wake_word && ./install_dependencies.sh"
    exit 1
fi

# Python dependencies
echo "📦 Installing Python packages..."
pip install --upgrade pip

# Core audio packages
pip install pyaudio webrtcvad noisereduce

# Wake word engines
echo "🎯 Installing wake word engines..."

# Porcupine (Free tier)
pip install pvporcupine

# Vosk (Offline speech recognition)
pip install vosk

# Download Vosk model if needed
echo "📥 Downloading Vosk model..."
python -c "
import os
import urllib.request
import zipfile

model_name = 'vosk-model-small-en-us-0.15'
model_url = f'https://alphacephei.com/vosk/models/{model_name}.zip'
model_dir = os.path.expanduser('~/.jarvis/models/vosk')
model_path = os.path.join(model_dir, model_name)

if not os.path.exists(model_path):
    print('Downloading Vosk model...')
    os.makedirs(model_dir, exist_ok=True)
    
    zip_path = os.path.join(model_dir, f'{model_name}.zip')
    urllib.request.urlretrieve(model_url, zip_path)
    
    print('Extracting model...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)
    
    os.remove(zip_path)
    print('✅ Vosk model downloaded successfully')
else:
    print('✅ Vosk model already exists')
"

# macOS specific setup
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected - installing audio dependencies..."
    
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install PortAudio for PyAudio
    if ! brew list portaudio &> /dev/null; then
        echo "Installing PortAudio..."
        brew install portaudio
    else
        echo "✅ PortAudio already installed"
    fi
fi

echo ""
echo "✅ Wake word dependencies installed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Start JARVIS: python start_system.py"
echo "2. Wake word detection will start automatically"
echo "3. Say 'Hey JARVIS' to activate!"
echo ""
echo "🔧 Configuration:"
echo "- Wake words can be customized via environment variables"
echo "- Set WAKE_WORDS='hey jarvis,jarvis,ok jarvis'"
echo "- Set WAKE_WORD_SENSITIVITY='medium' (very_low/low/medium/high/very_high)"
echo ""

# Make script executable for future use
chmod +x install_dependencies.sh