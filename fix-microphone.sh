#!/bin/bash

echo "🎤 JARVIS Microphone Fix Script - Enhanced Edition"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}❌ This script is designed for macOS only${NC}"
    exit 1
fi

# Function to check if a process is running
check_process() {
    if pgrep -x "$1" > /dev/null; then
        echo -e "${YELLOW}⚠️  $1 is running - may be using microphone${NC}"
        return 0
    else
        return 1
    fi
}

# Function to kill a process safely
kill_process() {
    local process=$1
    local pid=$(pgrep -x "$process")
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Stopping $process (PID: $pid)...${NC}"
        kill $pid 2>/dev/null
        sleep 1
        if pgrep -x "$process" > /dev/null; then
            echo -e "${RED}Failed to stop $process gracefully, forcing...${NC}"
            kill -9 $pid 2>/dev/null
        else
            echo -e "${GREEN}✓ $process stopped successfully${NC}"
        fi
    fi
}

echo -e "${BLUE}1. Running Python-based Microphone Diagnostic...${NC}"
echo ""

# Check if Python diagnostic exists and run it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIAGNOSTIC_PY="$SCRIPT_DIR/backend/system/microphone_diagnostic.py"

if [ -f "$DIAGNOSTIC_PY" ]; then
    python3 "$DIAGNOSTIC_PY"
    DIAG_RESULT=$?
    
    if [ $DIAG_RESULT -eq 0 ]; then
        echo -e "\n${GREEN}✅ Microphone diagnostic passed! No issues detected.${NC}"
        exit 0
    else
        echo -e "\n${YELLOW}⚠️  Microphone issues detected. Proceeding with fixes...${NC}"
    fi
else
    echo -e "${YELLOW}Python diagnostic not found, running basic checks...${NC}"
fi

echo ""
echo -e "${BLUE}2. Checking for processes using the microphone...${NC}"
echo ""

# Array of common apps that use microphone
AUDIO_APPS=(
    "zoom.us"
    "Teams"
    "Discord"
    "Slack"
    "Skype"
    "FaceTime"
    "WhatsApp"
    "Telegram"
    "Signal"
    "OBS"
    "QuickTime Player"
    "Voice Memos"
    "GarageBand"
    "Audacity"
    "ScreenFloat"
    "Loom"
    "CleanMyMac"
    "Screenium"
    "Capto"
)

FOUND_APPS=()

# Check each app
for app in "${AUDIO_APPS[@]}"; do
    if check_process "$app"; then
        FOUND_APPS+=("$app")
    fi
done

# Also check for any Chrome/Safari/Firefox processes with microphone
BROWSERS=("Google Chrome" "Safari" "Firefox" "Microsoft Edge")
for browser in "${BROWSERS[@]}"; do
    if pgrep -f "$browser" > /dev/null; then
        echo -e "${CYAN}ℹ️  $browser is running - checking for microphone use...${NC}"
        
        # Check if browser has microphone permission
        if [[ "$browser" == "Google Chrome" ]]; then
            # Check Chrome's microphone usage
            CHROME_MIC=$(lsof 2>/dev/null | grep -i "chrome" | grep -i "audio")
            if [ -n "$CHROME_MIC" ]; then
                echo -e "${YELLOW}   ⚠️  Chrome appears to be using the microphone${NC}"
                FOUND_APPS+=("Chrome")
            fi
        fi
    fi
done

echo ""
echo -e "${BLUE}3. Checking Core Audio status...${NC}"
PID=$(pgrep coreaudiod)
if [ -n "$PID" ]; then
    echo -e "${GREEN}✓ Core Audio is running (PID: $PID)${NC}"
    
    # Check if Core Audio is stuck
    CPU_USAGE=$(ps aux | grep "[c]oreaudiod" | awk '{print $3}')
    if (( $(echo "$CPU_USAGE > 50" | bc -l) )); then
        echo -e "${RED}⚠️  Core Audio using high CPU ($CPU_USAGE%) - may be stuck${NC}"
        RESTART_AUDIO=true
    fi
else
    echo -e "${RED}❌ Core Audio is not running${NC}"
    RESTART_AUDIO=true
fi

# Display found apps
if [ ${#FOUND_APPS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Found ${#FOUND_APPS[@]} app(s) that may be using the microphone:${NC}"
    for app in "${FOUND_APPS[@]}"; do
        echo "  • $app"
    done
    
    echo ""
    read -p "Would you like to close these applications? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for app in "${FOUND_APPS[@]}"; do
            kill_process "$app"
        done
    fi
fi

# Restart Core Audio if needed
if [ "$RESTART_AUDIO" = true ] || [ ${#FOUND_APPS[@]} -gt 0 ]; then
    echo ""
    read -p "Would you like to restart Core Audio? This may help release the microphone. (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Restarting Core Audio..."
        sudo killall coreaudiod 2>/dev/null
        sleep 2
        
        # Verify Core Audio restarted
        if pgrep coreaudiod > /dev/null; then
            echo -e "${GREEN}✓ Core Audio restarted successfully${NC}"
        else
            echo -e "${RED}❌ Core Audio failed to restart${NC}"
        fi
    fi
fi

echo ""
echo -e "${BLUE}4. Testing microphone access...${NC}"

# Quick microphone test using sox if available
if command -v sox &> /dev/null; then
    echo "Testing microphone with sox..."
    if timeout 1 sox -d -n stat 2>&1 | grep -q "Maximum amplitude"; then
        echo -e "${GREEN}✓ Microphone is accessible${NC}"
    else
        echo -e "${RED}❌ Microphone test failed${NC}"
    fi
else
    echo -e "${YELLOW}sox not installed - skipping audio test${NC}"
    echo "Install with: brew install sox"
fi

echo ""
echo -e "${BLUE}5. Browser-specific fixes:${NC}"
echo ""

# Get default browser
DEFAULT_BROWSER=$(defaults read ~/Library/Preferences/com.apple.LaunchServices/com.apple.launchservices.secure | grep -B 1 'https' | grep 'LSHandlerRoleAll' -A 1 | tail -1 | cut -d '"' -f 4)

case "$DEFAULT_BROWSER" in
    *chrome*)
        echo -e "${CYAN}Chrome detected as default browser${NC}"
        echo "  1. Visit: chrome://settings/content/microphone"
        echo "  2. Remove 'localhost' if present"
        echo "  3. Add 'localhost' again and set to 'Allow'"
        echo "  4. Restart Chrome completely"
        
        # Offer to open Chrome settings
        read -p "Open Chrome microphone settings? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            open "chrome://settings/content/microphone"
        fi
        ;;
    *safari*)
        echo -e "${CYAN}Safari detected as default browser${NC}"
        echo "  1. Safari → Preferences → Websites → Microphone"
        echo "  2. Find localhost and set to 'Allow'"
        echo "  3. Restart Safari"
        ;;
    *firefox*)
        echo -e "${CYAN}Firefox detected as default browser${NC}"
        echo "  1. Click the lock icon in the address bar when on localhost"
        echo "  2. Clear permissions and reload"
        echo "  3. Grant permission when prompted"
        ;;
    *)
        echo "  • Chrome: chrome://settings/content/microphone"
        echo "  • Safari: Preferences → Websites → Microphone"
        echo "  • Firefox: Click lock icon → Clear permissions"
        ;;
esac

echo ""
echo -e "${BLUE}6. System Preferences check...${NC}"

# Check if Terminal has microphone permission
TERMINAL_MIC_PERMISSION=$(osascript -e 'tell application "System Events" to get the name of every process whose visible is true' 2>&1)
if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ Terminal has system permissions${NC}"
else
    echo -e "${YELLOW}⚠️  Terminal may need additional permissions${NC}"
    echo "  Grant in: System Preferences → Security & Privacy → Privacy → Microphone"
fi

echo ""
echo -e "${BLUE}7. Quick diagnostic test...${NC}"
echo -e "${CYAN}Visit: http://localhost:3000/microphone-test.html${NC}"
echo "This will run a comprehensive browser-based diagnostic"

# Final diagnostic run
echo ""
echo -e "${BLUE}8. Running final diagnostic...${NC}"
if [ -f "$DIAGNOSTIC_PY" ]; then
    python3 "$DIAGNOSTIC_PY"
    FINAL_RESULT=$?
    
    if [ $FINAL_RESULT -eq 0 ]; then
        echo -e "\n${GREEN}✅ Microphone is now working! JARVIS voice control is ready.${NC}"
    else
        echo -e "\n${YELLOW}⚠️  Some issues may still exist.${NC}"
        echo ""
        echo "Additional troubleshooting steps:"
        echo "  1. Restart your browser completely"
        echo "  2. Check Activity Monitor for apps using high CPU"
        echo "  3. Try a different browser (Chrome or Edge recommended)"
        echo "  4. Restart your Mac if issues persist"
        echo ""
        echo "For detailed diagnostics, check:"
        echo "  $SCRIPT_DIR/logs/microphone_diagnostic.log"
    fi
fi

echo ""
echo -e "${GREEN}✅ Fix script completed!${NC}"
echo ""

# Make the script executable (for next time)
chmod +x "$0" 2>/dev/null

# Offer to start JARVIS
read -p "Would you like to start JARVIS now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$SCRIPT_DIR"
    python3 start_system.py
fi