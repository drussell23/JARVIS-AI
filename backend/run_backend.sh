#!/bin/bash

# Simple backend startup script

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting AI-Powered Chatbot Backend${NC}"
echo "========================================"

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  Warning: ANTHROPIC_API_KEY not set${NC}"
    echo "   Limited functionality will be available"
else
    echo -e "${GREEN}✅ Anthropic API key found${NC}"
fi

# Run the backend
echo -e "\n${GREEN}Starting server on http://localhost:8000${NC}"
echo -e "${YELLOW}Note: Some ML features may be limited due to library compatibility${NC}\n"

# Start with environment variables to handle import issues
export TF_CPP_MIN_LOG_LEVEL=3
export USE_TORCH=1
export USE_TF=0

python main.py