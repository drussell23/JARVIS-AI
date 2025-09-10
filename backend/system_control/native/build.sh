#!/bin/bash

# Build script for JARVIS Weather native tool

echo "Building JARVIS Weather CLI..."

# Check if Swift is available
if ! command -v swift &> /dev/null; then
    echo "Error: Swift is not installed or not in PATH"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build the Swift tool
cd "$SCRIPT_DIR"

echo "Compiling jarvis-weather..."

# Check if we have entitlements file
if [ -f "jarvis-weather.entitlements" ]; then
    echo "Building with entitlements..."
    swiftc jarvis-weather.swift \
        -o jarvis-weather \
        -O \
        -whole-module-optimization \
        -framework CoreLocation \
        -framework WeatherKit \
        -Xlinker -sectcreate \
        -Xlinker __TEXT \
        -Xlinker __entitlements \
        -Xlinker jarvis-weather.entitlements
else
    echo "Building without entitlements (WeatherKit may not work)..."
    swiftc jarvis-weather.swift \
        -o jarvis-weather \
        -O \
        -whole-module-optimization \
        -framework CoreLocation \
        -framework WeatherKit
fi

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Binary created at: $SCRIPT_DIR/jarvis-weather"
    
    # Make it executable
    chmod +x jarvis-weather
    
    # Test basic functionality
    echo ""
    echo "Testing jarvis-weather..."
    ./jarvis-weather --version
    
    echo ""
    echo "To test weather functionality, run:"
    echo "  ./jarvis-weather current --pretty"
    echo ""
    echo "Note: You may need to grant location permissions on first run."
else
    echo "Build failed!"
    exit 1
fi