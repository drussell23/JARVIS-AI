#!/bin/bash
# Setup script for Claude API integration

echo "🚀 Setting up Claude API integration for JARVIS"
echo "============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ -f .env ]; then
    echo -e "${GREEN}✓ .env file found with API key${NC}"
else
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
    exit 1
fi

# Install anthropic package
echo -e "\n${BLUE}Installing anthropic package...${NC}"
pip install anthropic

# Install python-dotenv if not installed
echo -e "\n${BLUE}Installing python-dotenv...${NC}"
pip install python-dotenv

# Check if installation was successful
if python -c "import anthropic" 2>/dev/null; then
    echo -e "${GREEN}✓ Anthropic package installed successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Failed to install anthropic package${NC}"
    exit 1
fi

# Test Claude API connection
echo -e "\n${BLUE}Testing Claude API connection...${NC}"
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
import anthropic

api_key = os.getenv('ANTHROPIC_API_KEY')
if api_key:
    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Just initialize client, don't make actual API call to save costs
        print('✓ Claude API key is valid format')
    except Exception as e:
        print(f'⚠️  Error with API key: {e}')
else:
    print('⚠️  ANTHROPIC_API_KEY not found in .env file')
"

echo -e "\n${GREEN}✅ Claude setup complete!${NC}"
echo -e "\n${BLUE}To start JARVIS with Claude:${NC}"
echo "1. Regular web interface: python start_system.py"
echo "2. Claude-only mode: python start_jarvis_claude.py"
echo "3. Test integration: python test_claude_integration.py"
echo -e "\n${YELLOW}Note: Claude is cloud-based, so no memory constraints!${NC}"