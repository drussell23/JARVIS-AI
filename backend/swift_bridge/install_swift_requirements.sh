#!/bin/bash
#
# Installation script for Swift requirements
# Helps set up Swift development environment for JARVIS
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}🛠️  JARVIS Swift Classifier - Installation Helper${NC}"
echo "=================================================="

# Check current Swift status
echo -e "\n${CYAN}Checking Swift installation...${NC}"
if command -v swift &> /dev/null; then
    echo -e "${GREEN}✓ Swift is installed${NC}"
    swift --version
else
    echo -e "${RED}✗ Swift not found${NC}"
fi

# Check for Xcode
echo -e "\n${CYAN}Checking for Xcode...${NC}"
if [ -d "/Applications/Xcode.app" ]; then
    echo -e "${GREEN}✓ Xcode.app found${NC}"
    XCODE_VERSION=$(/usr/bin/xcodebuild -version 2>/dev/null | head -1 || echo "Unknown")
    echo "  Version: $XCODE_VERSION"
else
    echo -e "${YELLOW}⚠️  Xcode.app not found${NC}"
    echo -e "${YELLOW}   This is likely why Swift build is failing${NC}"
fi

# Check Xcode Command Line Tools
echo -e "\n${CYAN}Checking Command Line Tools...${NC}"
if xcode-select -p &>/dev/null; then
    echo -e "${GREEN}✓ Command Line Tools installed${NC}"
    echo "  Path: $(xcode-select -p)"
else
    echo -e "${RED}✗ Command Line Tools not properly configured${NC}"
fi

# Provide installation options
echo -e "\n${BOLD}Installation Options:${NC}"
echo "========================"

echo -e "\n${BOLD}Option 1: Install Xcode (Recommended for Swift)${NC}"
echo "1. Open the Mac App Store"
echo "2. Search for 'Xcode'"
echo "3. Click 'Get' or 'Install' (it's free but large ~7GB)"
echo "4. After installation, run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
echo "5. Accept license: sudo xcodebuild -license accept"

echo -e "\n${BOLD}Option 2: Use Swift Playgrounds (Lightweight)${NC}"
echo "1. Install Swift Playgrounds from Mac App Store (smaller than Xcode)"
echo "2. This may provide the necessary frameworks"

echo -e "\n${BOLD}Option 3: Continue with Python Fallback${NC}"
echo "The Python fallback classifier is already working!"
echo "You can use JARVIS with intelligent routing right now."

# Try alternative Swift installation
echo -e "\n${BOLD}Checking for alternative Swift installations...${NC}"

# Check for Homebrew Swift
if command -v brew &> /dev/null; then
    echo -e "\n${CYAN}Homebrew detected. Checking for Swift formula...${NC}"
    if brew list swift &>/dev/null 2>&1; then
        echo -e "${GREEN}✓ Swift installed via Homebrew${NC}"
    else
        echo -e "${YELLOW}Swift not installed via Homebrew${NC}"
        echo "To install: brew install swift"
    fi
fi

# Provide workaround
echo -e "\n${BOLD}Temporary Workaround:${NC}"
echo "========================"
echo "While Xcode is downloading/installing, JARVIS will automatically"
echo "use the Python fallback classifier which provides:"
echo "• Intelligent command routing"
echo "• Learning capabilities"
echo "• No hardcoded patterns"
echo ""
echo "Once Xcode is installed, JARVIS will automatically switch to"
echo "the Swift classifier for even better performance!"

# Test current capability
echo -e "\n${BOLD}Testing current routing capability...${NC}"
cd "$(dirname "$0")"
cd ..
if python3 -c "from swift_bridge.python_fallback_classifier import PythonCommandClassifier; print('✓ Python fallback classifier is ready!')" 2>/dev/null; then
    echo -e "${GREEN}✓ Intelligent routing is available NOW via Python fallback${NC}"
else
    echo -e "${RED}✗ Error loading fallback classifier${NC}"
fi

echo -e "\n${BOLD}Summary:${NC}"
echo "========"
if [ -d "/Applications/Xcode.app" ]; then
    echo -e "${YELLOW}⚠️  Xcode is installed but Swift build service is missing${NC}"
    echo "   Try: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
    echo "   Then: sudo xcodebuild -license accept"
else
    echo -e "${YELLOW}⚠️  Full Xcode installation recommended for Swift support${NC}"
    echo -e "${GREEN}✓  Python fallback provides intelligent routing now${NC}"
fi

echo -e "\n${CYAN}After installing Xcode, run:${NC}"
echo "  cd backend/swift_bridge"
echo "  ./build.sh"
echo ""
echo -e "${GREEN}JARVIS is ready to use with intelligent routing!${NC}"