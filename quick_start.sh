#!/bin/bash

# Quick start script for AI-Powered Chatbot System
# This script provides a simple way to start the system

echo "🤖 AI-Powered Chatbot Quick Start"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python $REQUIRED_VERSION or higher is required. Current version: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to kill processes on a port
kill_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&2; then
        echo -e "${YELLOW}Port $port is in use. Attempting to free it...${NC}"
        lsof -Pi :$port -sTCP:LISTEN -t | xargs kill -9 2>/dev/null
        sleep 1
    fi
}

# Parse command line arguments
SKIP_INSTALL=false
INSTALL_ONLY=false
BACKEND_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --install-only)
            INSTALL_ONLY=true
            shift
            ;;
        --backend-only)
            BACKEND_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-install    Skip dependency installation"
            echo "  --install-only    Only install dependencies, don't start services"
            echo "  --backend-only    Only start backend services"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ] && [ "$SKIP_INSTALL" = false ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

# Install dependencies
if [ "$SKIP_INSTALL" = false ]; then
    echo -e "${BLUE}Installing dependencies...${NC}"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install backend dependencies
    if [ -f "backend/requirements.txt" ]; then
        pip install -r backend/requirements.txt
        echo -e "${GREEN}✓ Backend dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠️  backend/requirements.txt not found${NC}"
    fi
    
    # Download NLTK data
    echo -e "${BLUE}Downloading NLTK data...${NC}"
    python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)" 2>/dev/null
    echo -e "${GREEN}✓ NLTK data downloaded${NC}"
    
    # Download spaCy model
    echo -e "${BLUE}Downloading spaCy model...${NC}"
    python3 -m spacy download en_core_web_sm 2>/dev/null || echo -e "${YELLOW}⚠️  Could not download spaCy model${NC}"
fi

# Exit if install-only mode
if [ "$INSTALL_ONLY" = true ]; then
    echo -e "${GREEN}✓ Installation complete${NC}"
    exit 0
fi

# Create necessary directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p backend/{data,models,checkpoints,logs,domain_knowledge,faiss_index,chroma_db,knowledge_base}
mkdir -p frontend
echo -e "${GREEN}✓ Directories created${NC}"

# Kill any processes using our ports
echo -e "${BLUE}Checking ports...${NC}"
kill_port 8000
kill_port 8001
kill_port 3000

# Function to start a service
start_service() {
    local name=$1
    local command=$2
    local dir=$3
    local port=$4
    
    echo -e "${BLUE}Starting $name on port $port...${NC}"
    cd "$dir" || exit
    nohup $command > "logs/${name}.log" 2>&1 &
    echo $! > "logs/${name}.pid"
    cd - > /dev/null || exit
    sleep 2
    
    # Check if service started
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&2; then
        echo -e "${GREEN}✓ $name started${NC}"
    else
        echo -e "${RED}❌ Failed to start $name${NC}"
        echo -e "${YELLOW}Check logs/backend/${name}.log for details${NC}"
    fi
}

# Start backend services
echo -e "\n${BLUE}Starting backend services...${NC}"

# Start main API
start_service "main_api" "uvicorn main:app --host 127.0.0.1 --port 8000" "backend" 8000

# Start training API
start_service "training_api" "uvicorn models.training_interface:app --host 0.0.0.0 --port 8001" "backend" 8001

# Start frontend if not backend-only
if [ "$BACKEND_ONLY" = false ]; then
    echo -e "\n${BLUE}Starting frontend...${NC}"
    
    # Check if frontend exists
    if [ -d "frontend" ]; then
        # Check for package.json
        if [ -f "frontend/package.json" ] && command_exists npm; then
            cd frontend || exit
            npm install --silent
            nohup npm start > ../logs/frontend.log 2>&1 &
            echo $! > ../logs/frontend.pid
            cd - > /dev/null || exit
            echo -e "${GREEN}✓ Frontend started${NC}"
        else
            # Create basic frontend if it doesn't exist
            if [ ! -f "frontend/index.html" ]; then
                python3 -c "
import sys
sys.path.append('.')
from start_system import SystemManager
manager = SystemManager()
manager.create_basic_frontend()
"
            fi
            
            # Start simple HTTP server
            cd frontend || exit
            nohup python3 -m http.server 3000 > ../logs/frontend.log 2>&1 &
            echo $! > ../logs/frontend.pid
            cd - > /dev/null || exit
            echo -e "${GREEN}✓ Frontend started (basic mode)${NC}"
        fi
    fi
fi

# Wait a moment for services to fully start
sleep 3

# Print access information
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 System is ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Access your services at:"
echo -e "  📱 Frontend:        ${BLUE}http://localhost:3000${NC}"
echo -e "  🔌 Main API:        ${BLUE}http://localhost:8000${NC}"
echo -e "  🔧 Training API:    ${BLUE}http://localhost:8001${NC}"
echo ""
echo "Demo interfaces:"
echo -e "  📚 API Docs:        ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  🎤 Voice Demo:      ${BLUE}http://localhost:8000/voice_demo.html${NC}"
echo -e "  ⚡ Automation:      ${BLUE}http://localhost:8000/automation_demo.html${NC}"
echo -e "  🧠 RAG System:      ${BLUE}http://localhost:8000/rag_demo.html${NC}"
echo -e "  🔧 LLM Training:    ${BLUE}http://localhost:8001/llm_demo.html${NC}"
echo ""
echo -e "${YELLOW}To stop all services, run: ./stop_system.sh${NC}"
echo ""

# Open browser
if command_exists open; then
    # macOS
    sleep 2
    open http://localhost:3000
elif command_exists xdg-open; then
    # Linux
    sleep 2
    xdg-open http://localhost:3000
fi

# Create stop script
cat > stop_system.sh << 'EOF'
#!/bin/bash

echo "Stopping AI-Powered Chatbot System..."

# Function to stop a service
stop_service() {
    local name=$1
    local pid_file="logs/${name}.pid"
    
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "✓ Stopped $name"
        fi
        rm "$pid_file"
    fi
}

# Stop all services
stop_service "main_api"
stop_service "training_api"
stop_service "frontend"

# Additional cleanup
pkill -f "python.*main.py" 2>/dev/null
pkill -f "python.*training_interface.py" 2>/dev/null
pkill -f "http.server.*3000" 2>/dev/null

echo "✓ All services stopped"
EOF

chmod +x stop_system.sh

echo -e "${GREEN}✓ System started successfully!${NC}"