#!/bin/bash
# Start JARVIS with miniforge Python (has working llama-cpp-python)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 Starting JARVIS with real language models...${NC}"

# Check if miniforge Python exists
if [ ! -f "/Users/derekjrussell/miniforge3/bin/python" ]; then
    echo -e "${RED}❌ Miniforge Python not found at /Users/derekjrussell/miniforge3/bin/python${NC}"
    echo -e "${YELLOW}Please install miniforge or update the path in this script${NC}"
    exit 1
fi

# Check if models exist
if [ ! -d "models" ]; then
    echo -e "${YELLOW}⚠️  Models directory not found. Creating...${NC}"
    mkdir -p models
fi

# Check for model files
if [ ! -f "models/tinyllama-1.1b.gguf" ] || [ ! -f "models/phi-2.gguf" ] || [ ! -f "models/mistral-7b-instruct.gguf" ]; then
    echo -e "${YELLOW}⚠️  Some models are missing. Run this to download:${NC}"
    echo -e "${GREEN}/Users/derekjrussell/miniforge3/bin/python download_jarvis_models.py${NC}"
    echo ""
fi

# Start the system
echo -e "${GREEN}Using miniforge Python with llama-cpp-python support${NC}"
/Users/derekjrussell/miniforge3/bin/python start_system.py "$@"