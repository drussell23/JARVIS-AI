#!/bin/bash
#
# Build Swift Proximity Service
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SWIFT_DIR="$SCRIPT_DIR/swift"

echo "🔨 Building JARVIS Proximity Service..."

# Check if Swift is installed
if ! command -v swift &> /dev/null; then
    echo "❌ Swift is not installed. Please install Xcode."
    exit 1
fi

# Navigate to Swift directory
cd "$SWIFT_DIR"

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf .build

# Build the package
echo "🏗️  Building Swift package..."
swift build -c release

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Create bin directory
    BIN_DIR="$SCRIPT_DIR/bin"
    mkdir -p "$BIN_DIR"
    
    # Copy executable
    cp .build/release/ProximityService "$BIN_DIR/"
    chmod +x "$BIN_DIR/ProximityService"
    
    echo "📦 Executable copied to: $BIN_DIR/ProximityService"
else
    echo "❌ Build failed!"
    exit 1
fi

echo "🎉 Swift Proximity Service build complete!"