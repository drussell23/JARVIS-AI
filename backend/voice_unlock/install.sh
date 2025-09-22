#!/bin/bash
#
# JARVIS Voice Unlock Installation Script
# ======================================
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
USER_NAME=$(whoami)

echo "🚀 JARVIS Voice Unlock Installation"
echo "==================================="
echo

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create necessary directories
echo "Creating directories..."
mkdir -p ~/.jarvis/voice_unlock/{models,logs,audit}
mkdir -p ~/.jarvis/voice_unlock/proximity_voice_auth/models

# Install Python dependencies
echo "Installing Python dependencies..."
cd "$PROJECT_ROOT"
if [ ! -d "backend/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv backend/venv
fi

echo "Activating virtual environment..."
source backend/venv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install -r backend/voice_unlock/requirements.txt

# Additional ML dependencies
pip install scikit-learn joblib numpy scipy librosa sounddevice

# Install proximity service
echo "Building proximity service..."
cd "$SCRIPT_DIR/proximity_voice_auth/swift"
swift build -c release

# Copy binary to bin directory
mkdir -p ~/.jarvis/bin
cp .build/release/ProximityService ~/.jarvis/bin/

# Install LaunchAgent
echo "Installing LaunchAgent..."
PLIST_SRC="$SCRIPT_DIR/com.jarvis.voiceunlock.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.jarvis.voiceunlock.plist"

# Update paths in plist
sed "s|/Users/derekjrussell|$HOME|g" "$PLIST_SRC" > "$PLIST_DEST"

# Load the agent
launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"

# Create command line tool symlink
echo "Creating command line tool..."
sudo ln -sf "$SCRIPT_DIR/jarvis_voice_unlock.py" /usr/local/bin/jarvis-voice-unlock
sudo chmod +x /usr/local/bin/jarvis-voice-unlock

# Set up permissions
echo "Setting up permissions..."
# Request microphone access
osascript -e 'tell application "System Events" to display dialog "JARVIS Voice Unlock needs microphone access. Please grant permission in System Preferences > Security & Privacy > Privacy > Microphone" buttons {"OK"} default button 1'

# Test installation
echo
echo "Testing installation..."
python3 -c "from backend.voice_unlock.ml import VoiceUnlockMLSystem; print('✓ ML system imported successfully')"
python3 -c "from backend.voice_unlock.config import get_config; print('✓ Configuration loaded successfully')"

echo
echo "✅ Installation complete!"
echo
echo "Usage:"
echo "  jarvis-voice-unlock enroll <username>  - Enroll a new user"
echo "  jarvis-voice-unlock test              - Test authentication"
echo "  jarvis-voice-unlock status            - Show system status"
echo "  jarvis-voice-unlock configure         - Configure settings"
echo
echo "The voice unlock service is now running in the background."
echo "Check logs at: ~/.jarvis/voice_unlock/logs/"
echo
echo "To stop the service:"
echo "  launchctl unload ~/Library/LaunchAgents/com.jarvis.voiceunlock.plist"
echo
echo "To start the service:"
echo "  launchctl load ~/Library/LaunchAgents/com.jarvis.voiceunlock.plist"