#!/bin/bash

echo "🚀 Terminal Location Services Setup"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Create entitlements file for the app
echo -e "${BLUE}Step 1: Creating entitlements file...${NC}"
cat > terminal-location-entitlements.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.personal-information.location</key>
    <true/>
</dict>
</plist>
EOF

# Step 2: Create Info.plist for the app
echo -e "${BLUE}Step 2: Creating Info.plist...${NC}"
cat > Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>TerminalLocationEnabler</string>
    <key>CFBundleIdentifier</key>
    <string>com.jarvis.terminal-location-enabler</string>
    <key>CFBundleName</key>
    <string>Terminal Location Enabler</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSLocationWhenInUseUsageDescription</key>
    <string>This app enables Terminal to access location services for weather and location-based features.</string>
    <key>NSLocationAlwaysAndWhenInUseUsageDescription</key>
    <string>This app enables Terminal to access location services for weather and location-based features.</string>
    <key>NSLocationUsageDescription</key>
    <string>This app enables Terminal to access location services for weather and location-based features.</string>
</dict>
</plist>
EOF

# Step 3: Compile the Swift app
echo -e "${BLUE}Step 3: Compiling Terminal Location Enabler...${NC}"
if swiftc terminal-location-enabler.swift \
    -o TerminalLocationEnabler \
    -framework CoreLocation \
    -framework AppKit \
    -import-objc-header /dev/null 2>/dev/null; then
    echo -e "${GREEN}✅ Compilation successful!${NC}"
else
    echo -e "${RED}❌ Compilation failed. Trying alternative method...${NC}"
    # Try without import-objc-header flag
    swiftc terminal-location-enabler.swift \
        -o TerminalLocationEnabler \
        -framework CoreLocation \
        -framework AppKit
fi

# Step 4: Create app bundle structure
echo -e "${BLUE}Step 4: Creating app bundle...${NC}"
APP_NAME="TerminalLocationEnabler.app"
rm -rf "$APP_NAME"
mkdir -p "$APP_NAME/Contents/MacOS"
mkdir -p "$APP_NAME/Contents/Resources"

# Move files to app bundle
cp TerminalLocationEnabler "$APP_NAME/Contents/MacOS/"
cp Info.plist "$APP_NAME/Contents/"

# Step 5: Try to code sign (may fail without developer certificate)
echo -e "${BLUE}Step 5: Attempting to sign the app...${NC}"
if codesign --force --sign - --entitlements terminal-location-entitlements.plist "$APP_NAME" 2>/dev/null; then
    echo -e "${GREEN}✅ App signed successfully!${NC}"
else
    echo -e "${YELLOW}⚠️  Could not sign app (this is normal without a developer certificate)${NC}"
fi

# Step 6: Make the binary executable
chmod +x "$APP_NAME/Contents/MacOS/TerminalLocationEnabler"

echo ""
echo -e "${GREEN}=================================="
echo "✅ Setup Complete!"
echo "==================================${NC}"
echo ""
echo -e "${YELLOW}NEXT STEPS:${NC}"
echo ""
echo "1. Run the app with ONE of these methods:"
echo ""
echo "   Method A (Recommended):"
echo -e "   ${BLUE}open TerminalLocationEnabler.app${NC}"
echo ""
echo "   Method B (Direct execution):"
echo -e "   ${BLUE}./TerminalLocationEnabler.app/Contents/MacOS/TerminalLocationEnabler${NC}"
echo ""
echo "2. Click 'Request Location Permission' in the app window"
echo ""
echo "3. When the system dialog appears, click 'Allow'"
echo ""
echo "4. Check System Settings > Privacy & Security > Location Services"
echo "   - Terminal should now appear in the list"
echo "   - Enable the checkbox next to Terminal"
echo ""
echo -e "${YELLOW}TROUBLESHOOTING:${NC}"
echo ""
echo "If Terminal doesn't appear in Location Services:"
echo "  • Close ALL Terminal windows"
echo "  • Open a brand new Terminal window"
echo "  • Run this setup script again"
echo "  • Try running from /Applications/Utilities/Terminal.app directly"
echo ""
echo "Alternative approach:"
echo "  • Install CoreLocationCLI: brew install corelocationcli"
echo "  • Run: CoreLocationCLI -once"
echo ""

# Offer to run the app now
echo -e "${BLUE}Would you like to run the app now? (y/n)${NC}"
read -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Launching Terminal Location Enabler...${NC}"
    open TerminalLocationEnabler.app
fi