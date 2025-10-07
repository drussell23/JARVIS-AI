#!/bin/bash
# Build script for CoreML Voice Engine

set -e  # Exit on error

echo "====================================="
echo "CoreML Voice Engine - Build Script"
echo "====================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$SCRIPT_DIR/build"

# Clean old build
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning old build..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Run CMake
echo "Running CMake configuration..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building C++ library..."
make -j$(sysctl -n hw.ncpu)

# Install (copy to parent directory)
echo "Installing library..."
make install

# Verify
if [ -f "$SCRIPT_DIR/libvoice_engine.dylib" ]; then
    echo "✅ SUCCESS: libvoice_engine.dylib created"
    echo "Location: $SCRIPT_DIR/libvoice_engine.dylib"

    # Print library info
    echo ""
    echo "Library info:"
    file "$SCRIPT_DIR/libvoice_engine.dylib"
    otool -L "$SCRIPT_DIR/libvoice_engine.dylib" | head -10
else
    echo "❌ ERROR: Failed to create libvoice_engine.dylib"
    exit 1
fi

echo ""
echo "====================================="
echo "Build complete!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Train or download CoreML models (.mlmodelc)"
echo "2. Test with: python3 voice_engine_bridge.py"
echo "3. Integrate with jarvis_voice.py"
