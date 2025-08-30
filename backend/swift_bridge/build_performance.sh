#!/bin/bash
# Build Swift Performance Libraries for JARVIS
# Optimized for Apple Silicon with no hardcoding

set -e

echo "🚀 Building Swift Performance Libraries..."
echo "====================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
swift package clean

# Build with release optimizations
echo "🔨 Building with optimizations..."
swift build -c release \
    -Xswiftc -O \
    -Xswiftc -whole-module-optimization \
    -Xswiftc -cross-module-optimization \
    -Xswiftc -target \
    -Xswiftc arm64-apple-macosx11.0

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # List built products
    echo ""
    echo "📦 Built products:"
    ls -la .build/release/*.dylib 2>/dev/null || echo "No dynamic libraries found"
    ls -la .build/release/jarvis-* 2>/dev/null || echo "No executables found"
    
    # Create symlinks for easier access
    if [ -f ".build/release/libPerformanceCore.dylib" ]; then
        ln -sf .build/release/libPerformanceCore.dylib libPerformanceCore.dylib
        echo "✅ Created symlink: libPerformanceCore.dylib"
    fi
    
    if [ -f ".build/release/libCommandClassifierDynamic.dylib" ]; then
        ln -sf .build/release/libCommandClassifierDynamic.dylib libCommandClassifierDynamic.dylib
        echo "✅ Created symlink: libCommandClassifierDynamic.dylib"
    fi
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "🎉 Swift Performance Libraries ready for use!"
echo "You can now use the performance bridge from Python."