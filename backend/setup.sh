#!/bin/bash

echo "🚀 Setting up AI-Powered Chatbot Backend..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python dependencies..."
echo "⏳ This may take several minutes as it includes ML models..."

# Install PyTorch first (CPU version for compatibility)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Download spaCy language model
echo "📦 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Install system dependencies notice
echo ""
echo "⚠️  System Dependencies Notice:"
echo "For full voice functionality, you may need to install:"
echo ""
echo "🔊 On macOS:"
echo "   brew install portaudio ffmpeg"
echo ""
echo "🔊 On Ubuntu/Debian:"
echo "   sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg"
echo ""
echo "🔊 On Windows:"
echo "   - Download PyAudio wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio"
echo "   - Install ffmpeg from: https://ffmpeg.org/download.html"
echo ""

# Create necessary directories
mkdir -p logs
mkdir -p temp

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the server:"
echo "   1. Activate the virtual environment: source venv/bin/activate"
echo "   2. Run the server: python main.py"
echo ""
echo "📚 API documentation will be available at: http://localhost:8000/docs"
echo "🎤 Voice demo interface: Open voice_demo.html in your browser"