#!/bin/bash
# Install SocketRocket Framework for WebSocket support

echo "Installing SocketRocket framework..."

# Create temp directory
TEMP_DIR="/tmp/socketrocket_build"
mkdir -p $TEMP_DIR
cd $TEMP_DIR

# Clone SocketRocket
echo "Cloning SocketRocket..."
git clone https://github.com/facebookincubator/SocketRocket.git
cd SocketRocket

# Build framework
echo "Building SocketRocket framework..."
xcodebuild -project SocketRocket.xcodeproj \
           -scheme SocketRocket-macOS \
           -configuration Release \
           -derivedDataPath build \
           ONLY_ACTIVE_ARCH=NO

# Find the built framework
FRAMEWORK_PATH=$(find build -name "SocketRocket.framework" -type d | grep -E "Release|Products" | head -1)

if [ -z "$FRAMEWORK_PATH" ]; then
    echo "Error: Could not find built framework"
    exit 1
fi

# Install framework
echo "Installing framework to /usr/local/lib..."
sudo mkdir -p /usr/local/lib
sudo cp -R "$FRAMEWORK_PATH" /usr/local/lib/

echo "SocketRocket framework installed successfully!"

# Cleanup
cd /
rm -rf $TEMP_DIR

echo "Installation complete."