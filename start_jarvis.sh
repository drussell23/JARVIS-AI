#!/bin/bash
###############################################################################
# JARVIS Startup Script with Goal Inference Configuration
###############################################################################

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Set default optimal settings if not already set
export JARVIS_GOAL_PRESET="${JARVIS_GOAL_PRESET:-balanced}"
export JARVIS_GOAL_AUTOMATION="${JARVIS_GOAL_AUTOMATION:-true}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   🤖 JARVIS AI ASSISTANT                       ║${NC}"
echo -e "${BLUE}║              Goal Inference & Learning System                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for preset argument
PRESET=""
AUTOMATION_FLAG=""

if [ ! -z "$1" ]; then
    PRESET="$1"
    echo -e "${YELLOW}🎯 Configuration Preset: ${PRESET}${NC}"

    # Check for automation flag as second argument
    if [ "$2" == "--enable-automation" ] || [ "$2" == "-a" ]; then
        AUTOMATION_FLAG="--enable-automation"
        echo -e "${GREEN}✓ Automation: ENABLED${NC}"
    elif [ "$2" == "--disable-automation" ] || [ "$2" == "-d" ]; then
        AUTOMATION_FLAG="--disable-automation"
        echo -e "${YELLOW}⚠️ Automation: DISABLED${NC}"
    fi
    echo ""
fi

# Display preset options if no argument
if [ -z "$PRESET" ]; then
    echo -e "${GREEN}Available Configuration Presets:${NC}"
    echo ""
    echo "  ${BLUE}aggressive${NC}   - Highly proactive, learns quickly, suggests often"
    echo "                 (Goal Confidence: 0.65, Automation: ON)"
    echo ""
    echo "  ${BLUE}balanced${NC}     - Default balanced settings (recommended)"
    echo "                 (Goal Confidence: 0.65, Automation: ON by default)"
    echo ""
    echo "  ${BLUE}conservative${NC} - Cautious, requires high confidence"
    echo "                 (Goal Confidence: 0.85, Automation: OFF)"
    echo ""
    echo "  ${BLUE}learning${NC}     - Optimized for learning your patterns quickly"
    echo "                 (Min Patterns: 2, High Boost, Automation: ON)"
    echo ""
    echo "  ${BLUE}performance${NC}  - Maximum speed, aggressive caching"
    echo "                 (Cache: 200 entries, TTL: 600s, Preload: ON)"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./start_jarvis.sh [preset] [automation_flag]"
    echo ""
    echo "  Examples:"
    echo "    ./start_jarvis.sh aggressive"
    echo "    ./start_jarvis.sh learning --enable-automation"
    echo "    ./start_jarvis.sh balanced -a                  # -a is short for --enable-automation"
    echo "    ./start_jarvis.sh conservative --disable-automation"
    echo "    ./start_jarvis.sh                               # Use existing/default config"
    echo ""
    echo -e "${YELLOW}Or use start_system.py directly:${NC}"
    echo "  python start_system.py --goal-preset learning --enable-automation"
    echo ""
    read -p "Press Enter to start with default config, or Ctrl+C to exit..."
fi

# Start JARVIS
echo ""
echo -e "${GREEN}🚀 Starting JARVIS...${NC}"
echo ""

# Build command with preset and automation flags
if [ ! -z "$PRESET" ]; then
    python start_system.py --goal-preset "$PRESET" $AUTOMATION_FLAG
else
    python start_system.py $AUTOMATION_FLAG
fi

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  JARVIS exited with code ${EXIT_CODE}${NC}"
    echo ""
fi

exit $EXIT_CODE
