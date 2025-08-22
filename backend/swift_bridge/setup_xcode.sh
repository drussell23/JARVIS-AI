#!/bin/bash
#
# Setup script for Xcode and Swift classifier
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

echo -e "${BOLD}🛠️  Setting up Xcode for JARVIS Swift Classifier${NC}"
echo "================================================"

# Check if Xcode is installed
if [ -d "/Applications/Xcode.app" ]; then
    echo -e "${GREEN}✓ Xcode is installed${NC}"
else
    echo -e "${RED}✗ Xcode not found at /Applications/Xcode.app${NC}"
    exit 1
fi

# Check current xcode-select path
CURRENT_PATH=$(xcode-select -p 2>/dev/null || echo "none")
echo -e "\nCurrent Xcode path: ${CYAN}$CURRENT_PATH${NC}"

# Set Xcode path if needed
if [ "$CURRENT_PATH" != "/Applications/Xcode.app/Contents/Developer" ]; then
    echo -e "\n${YELLOW}Setting Xcode path...${NC}"
    echo "Please run: ${BOLD}sudo xcode-select -s /Applications/Xcode.app/Contents/Developer${NC}"
fi

# Check license
echo -e "\n${CYAN}Checking Xcode license...${NC}"
if xcodebuild -checkFirstLaunchStatus &>/dev/null; then
    echo -e "${GREEN}✓ Xcode license already accepted${NC}"
else
    echo -e "${YELLOW}⚠️  Xcode license needs to be accepted${NC}"
    echo -e "\nPlease run: ${BOLD}sudo xcodebuild -license accept${NC}"
    echo -e "Or open Xcode.app and accept the license there"
fi

echo -e "\n${BOLD}Quick Setup Commands:${NC}"
echo "===================="
echo "1. Accept Xcode license:"
echo "   ${CYAN}sudo xcodebuild -license accept${NC}"
echo ""
echo "2. Build Swift classifier:"
echo "   ${CYAN}cd backend/swift_bridge${NC}"
echo "   ${CYAN}./build.sh${NC}"
echo ""
echo "3. Test the classifier:"
echo "   ${CYAN}cd backend${NC}"
echo "   ${CYAN}python3 test_intelligent_routing.py${NC}"

echo -e "\n${YELLOW}Alternative: Open Xcode.app${NC}"
echo "If the command line doesn't work, you can:"
echo "1. Open /Applications/Xcode.app"
echo "2. Accept any licenses or prompts"
echo "3. Then run ./build.sh again"

echo -e "\n${GREEN}Once the license is accepted, the Swift classifier will build successfully!${NC}"