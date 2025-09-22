#!/bin/bash
#
# Install Dependencies for Proximity + Voice Auth
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "🔧 Installing JARVIS Proximity + Voice Auth Dependencies..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    echo "❌ Python $REQUIRED_VERSION or higher required. Found: $PYTHON_VERSION"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python packages..."
cd "$SCRIPT_DIR/python"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Check if on Apple Silicon
if [[ $(uname -m) == 'arm64' ]]; then
    echo "🍎 Detected Apple Silicon Mac"
    # Install TensorFlow for M1/M2
    pip install tensorflow-macos tensorflow-metal
    # Remove from requirements to avoid conflict
    grep -v "tensorflow==" requirements.txt > requirements_temp.txt
    mv requirements_temp.txt requirements.txt
fi

# Install requirements
pip install -r requirements.txt

echo "✅ Python dependencies installed"

# Install Swift dependencies
echo "📦 Installing Swift dependencies..."
cd "$SCRIPT_DIR/swift"

# Resolve Swift package dependencies
swift package resolve

# Install ZeroMQ if not present
if ! brew list zeromq &>/dev/null; then
    echo "🍺 Installing ZeroMQ via Homebrew..."
    brew install zeromq
fi

# Check for pyzmq in system
echo "🔍 Checking ZeroMQ Python bindings..."
python3 -c "import zmq" 2>/dev/null || {
    echo "Installing pyzmq..."
    pip install pyzmq
}

echo "✅ All dependencies installed successfully!"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p "$SCRIPT_DIR/models"
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/bin"

# Set up configuration
if [ ! -f "$SCRIPT_DIR/auth_engine/auth_config.json" ]; then
    echo "⚙️  Creating default configuration..."
    python3 -c "
import json
from pathlib import Path

config = {
    'proximity': {
        'min_confidence': 80.0,
        'detection_range': 3.0,
        'update_frequency': 2
    },
    'voice': {
        'min_confidence': 85.0,
        'min_samples': 3,
        'sample_duration': 3.0,
        'liveness_required': True
    },
    'security': {
        'combined_threshold': 90.0,
        'max_attempts': 3,
        'lockout_duration': 300,
        'proximity_weight': 0.4,
        'voice_weight': 0.6
    },
    'learning': {
        'enabled': True,
        'update_frequency': 'realtime',
        'retention_period': 90,
        'privacy_mode': 'local_only'
    }
}

config_path = Path('$SCRIPT_DIR/auth_engine/auth_config.json')
config_path.parent.mkdir(parents=True, exist_ok=True)

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
"
fi

echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Build Swift package: ./build_swift.sh"
echo "2. Run integration tests: ./test_integration.sh"
echo "3. Start the service: ./start_proximity_auth.sh"