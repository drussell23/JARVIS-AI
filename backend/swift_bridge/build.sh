#!/bin/bash
#
# Build script for JARVIS Swift Command Classifier
# Builds the Swift package and prepares it for Python integration
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Building JARVIS Swift Command Classifier${NC}"
echo "========================================"

# Change to script directory
cd "$(dirname "$0")"

# Check for Swift
if ! command -v swift &> /dev/null; then
    echo -e "${RED}❌ Swift not found. Please install Xcode.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Swift version:${NC}"
swift --version

# Clean previous builds
if [ "$1" == "clean" ] || [ "$1" == "--clean" ]; then
    echo -e "\n${YELLOW}Cleaning previous builds...${NC}"
    rm -rf .build
    rm -rf .swiftpm
    echo -e "${GREEN}✓ Cleaned${NC}"
    
    if [ "$1" == "clean" ]; then
        exit 0
    fi
fi

# Build the Swift package
echo -e "\n${YELLOW}Building Swift package...${NC}"
swift build -c release

# Check if build succeeded
if [ -f ".build/release/jarvis-classifier" ]; then
    echo -e "${GREEN}✓ Command-line tool built successfully${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

# Build dynamic library for Python
echo -e "\n${YELLOW}Building dynamic library...${NC}"
swift build -c release --product CommandClassifierDynamic

# Make CLI tool executable
chmod +x .build/release/jarvis-classifier

# Test the classifier
echo -e "\n${YELLOW}Testing classifier...${NC}"
if ./.build/release/jarvis-classifier "close whatsapp" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Classifier test passed${NC}"
else
    echo -e "${RED}❌ Classifier test failed${NC}"
    exit 1
fi

# Test Python integration
echo -e "\n${YELLOW}Testing Python integration...${NC}"
if python3 -c "import sys; sys.path.append('.'); from python_bridge import SwiftCommandClassifier; print('✓ Python import successful')" 2>/dev/null; then
    echo -e "${GREEN}✓ Python integration working${NC}"
else
    echo -e "${YELLOW}⚠️  Python integration needs testing${NC}"
fi

echo -e "\n${GREEN}✅ Build complete!${NC}"
echo -e "\nUsage:"
echo -e "  ${YELLOW}Command-line:${NC} ./.build/release/jarvis-classifier \"your command\""
echo -e "  ${YELLOW}Python:${NC} from python_bridge import IntelligentCommandRouter"
echo -e "\nTo run tests:"
echo -e "  ${YELLOW}swift test${NC}"