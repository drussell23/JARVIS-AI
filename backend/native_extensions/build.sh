#!/bin/bash
#
# Build script for JARVIS Fast Capture C++ Extension
# This script handles the compilation and installation of the high-performance screen capture module
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 JARVIS Fast Capture Build Script${NC}"
echo "=================================="

# Check dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}❌ CMake not found. Please install CMake first.${NC}"
    echo "   On macOS: brew install cmake"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found.${NC}"
    exit 1
fi

# Check for pybind11
echo "Checking for pybind11..."
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  pybind11 not found. Installing...${NC}"
    pip3 install pybind11
fi

# Clean previous builds
if [ "$1" == "clean" ] || [ "$1" == "--clean" ]; then
    echo -e "\n${YELLOW}Cleaning previous builds...${NC}"
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info
    rm -f *.so
    rm -f *.dylib
    echo -e "${GREEN}✓ Cleaned${NC}"
    
    if [ "$1" == "clean" ]; then
        exit 0
    fi
fi

# Create build directory
echo -e "\n${YELLOW}Creating build directory...${NC}"
mkdir -p build
cd build

# Configure with CMake
echo -e "\n${YELLOW}Configuring with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo -e "\n${YELLOW}Building C++ extension...${NC}"
make -j$(sysctl -n hw.ncpu)

# Install
echo -e "\n${YELLOW}Installing extension...${NC}"
make install

# Go back to native_extensions directory
cd ..

# Test import
echo -e "\n${YELLOW}Testing import...${NC}"
if python3 -c "import fast_capture; print('✓ Fast Capture version:', fast_capture.VERSION)" 2>/dev/null; then
    echo -e "${GREEN}✓ Import successful!${NC}"
else
    echo -e "${RED}❌ Import failed${NC}"
    exit 1
fi

# Run performance test if requested
if [ "$2" == "test" ] || [ "$1" == "test" ]; then
    echo -e "\n${YELLOW}Running performance test...${NC}"
    cd ../vision
    python3 test_enhanced_vision.py
fi

echo -e "\n${GREEN}✅ Build complete!${NC}"
echo -e "\nTo use the Fast Capture engine:"
echo -e "  ${YELLOW}from backend.native_extensions.fast_capture_wrapper import FastCaptureEngine${NC}"
echo -e "\nTo run tests:"
echo -e "  ${YELLOW}./build.sh test${NC}"