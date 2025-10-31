#!/bin/bash

###############################################################################
# JARVIS Status Checker & Auto-Starter
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

###############################################################################
# Check Status
###############################################################################

print_header "JARVIS System Status Check"

# Check if backend is running
if pgrep -f "python.*main.py\|python.*start_system" > /dev/null 2>&1; then
    BACKEND_PID=$(pgrep -f "python.*main.py\|python.*start_system" | head -1)
    print_success "Backend is running (PID: $BACKEND_PID)"
    BACKEND_RUNNING=true
else
    print_error "Backend is NOT running"
    BACKEND_RUNNING=false
fi

# Check API endpoint
if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    print_success "API endpoint responding on port 8000"
    API_RESPONDING=true
else
    print_error "API endpoint NOT responding on port 8000"
    API_RESPONDING=false
fi

# Check Cloud SQL Proxy
if pgrep -f "cloud.*sql.*proxy" > /dev/null 2>&1; then
    PROXY_PID=$(pgrep -f "cloud.*sql.*proxy" | head -1)
    print_success "Cloud SQL Proxy running (PID: $PROXY_PID)"
    PROXY_RUNNING=true
else
    print_warning "Cloud SQL Proxy is NOT running"
    PROXY_RUNNING=false
fi

# Check microphone
print_info "Checking microphone..."
python3 << 'EOF'
import sounddevice as sd
devices = sd.query_devices()
mic_count = sum(1 for d in devices if d['max_input_channels'] > 0)
print(f"✅ Found {mic_count} microphone(s)")
EOF

# Check voice models
if [ -d "$HOME/.cache/jarvis/speechbrain" ]; then
    print_success "Voice models cached"
else
    print_warning "Voice models NOT cached (will download on first use)"
fi

# Check speaker profile
python3 << 'EOF'
import sys
sys.path.insert(0, 'backend')
try:
    from intelligence.learning_database import LearningDatabase
    import asyncio

    async def check_profile():
        db = LearningDatabase()
        await db.initialize()
        profiles = await db.get_all_speaker_profiles()
        if profiles:
            for p in profiles:
                print(f"✅ Speaker profile: {p['speaker_name']} ({p['total_samples']} samples)")
        else:
            print("⚠️  No speaker profiles found")

    asyncio.run(check_profile())
except Exception as e:
    print(f"⚠️  Could not check speaker profiles: {e}")
EOF

echo ""

###############################################################################
# Diagnosis
###############################################################################

print_header "Diagnosis"

if [ "$BACKEND_RUNNING" = true ] && [ "$API_RESPONDING" = true ]; then
    print_success "JARVIS is fully operational!"
    echo ""
    print_info "Voice unlock should work. Try:"
    echo "  1. Say: 'Hey JARVIS'"
    echo "  2. Wait for activation"
    echo "  3. Say: 'unlock my screen'"
    exit 0
fi

if [ "$BACKEND_RUNNING" = false ]; then
    print_error "JARVIS backend is not running - this is why voice commands fail!"
    echo ""
    print_info "To fix:"
    echo "  Option 1 (Full system):"
    echo "    ./start_system.py"
    echo ""
    echo "  Option 2 (Backend only):"
    echo "    cd backend && python main.py"
    echo ""

    read -p "Would you like to start JARVIS now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Starting JARVIS backend..."

        # Start Cloud SQL Proxy if not running
        if [ "$PROXY_RUNNING" = false ]; then
            print_info "Starting Cloud SQL Proxy..."
            if [ -f "./cloud_sql_proxy" ]; then
                ./cloud_sql_proxy -instances=jarvis-473803:us-central1:jarvis-learning-db --port 5432 &
                sleep 2
                print_success "Cloud SQL Proxy started"
            else
                print_warning "Cloud SQL Proxy not found, skipping..."
            fi
        fi

        # Start backend
        print_info "Starting JARVIS backend..."
        python3 start_system.py &

        # Wait for startup
        print_info "Waiting for JARVIS to initialize (30s)..."
        sleep 30

        # Check if it started
        if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
            print_success "JARVIS started successfully!"
            print_info "You can now use voice commands"
        else
            print_error "JARVIS failed to start. Check logs:"
            echo "  tail -f jarvis_startup.log"
        fi
    fi
fi

if [ "$BACKEND_RUNNING" = true ] && [ "$API_RESPONDING" = false ]; then
    print_warning "Backend is running but API not responding"
    print_info "The backend may still be initializing. Wait 30s and try again."
    print_info "Or check logs: tail -f jarvis_startup.log"
fi
