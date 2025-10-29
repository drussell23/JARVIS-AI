#!/bin/bash
#
# JARVIS GCP Spot VM Startup Script
# ==================================
#
# Automatically sets up a fresh GCP VM with JARVIS backend
# This script:
# 1. Installs system dependencies
# 2. Clones JARVIS repo (or uses pre-baked image)
# 3. Installs Python dependencies
# 4. Configures environment
# 5. Starts JARVIS backend on port 8010
#

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "🚀 JARVIS GCP VM Startup Script"
echo "================================"
echo "Starting at: $(date)"
echo "Instance: $(hostname)"
echo "Zone: $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)"

# Get instance metadata
JARVIS_COMPONENTS=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/jarvis-components || echo "")
JARVIS_TRIGGER=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/jarvis-trigger || echo "")

echo "📦 Components to run: ${JARVIS_COMPONENTS:-all}"
echo "🎯 Trigger reason: ${JARVIS_TRIGGER:-manual}"

# Update system
echo "📥 Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3.10-dev \
    htop \
    screen

# Install Python 3.10 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Clone JARVIS repository
echo "📥 Cloning JARVIS repository..."
cd /home
if [ ! -d "JARVIS-AI-Agent" ]; then
    # Note: In production, you'd want to use a service account with read access
    # or use a pre-baked image with the code already installed
    git clone https://github.com/YOUR_USERNAME/JARVIS-AI-Agent.git || {
        echo "⚠️  Git clone failed, using fallback method..."
        # Fallback: Download release tarball or use pre-installed code
        # For now, we'll create a minimal structure
        mkdir -p JARVIS-AI-Agent/backend
        echo "⚠️  Repository not available - using minimal setup"
    }
fi

cd JARVIS-AI-Agent/backend

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt
fi

# Install GCP-specific dependencies
python3 -m pip install \
    fastapi \
    uvicorn \
    google-cloud-storage \
    google-cloud-sql-connector \
    asyncpg \
    python-dotenv

# Set up environment variables
echo "⚙️  Configuring environment..."
cat > /home/JARVIS-AI-Agent/backend/.env.gcp << EOF
# GCP VM Environment Configuration
GCP_PROJECT_ID=jarvis-473803
GCP_REGION=us-central1
GCP_ZONE=us-central1-a

# Backend Configuration
BACKEND_PORT=8010
BACKEND_HOST=0.0.0.0

# Component Configuration
JARVIS_COMPONENTS=${JARVIS_COMPONENTS:-VISION,CHATBOTS,ML_MODELS}

# Optimization
OPTIMIZE_STARTUP=true
LAZY_LOAD_MODELS=true
DYNAMIC_LOADING_ENABLED=true

# Disable local-only components
ENABLE_VOICE_UNLOCK=false
ENABLE_WAKE_WORD=false
ENABLE_MACOS_AUTOMATION=false

# Cloud SQL (use Cloud SQL Proxy)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=jarvis_learning
DB_USER=jarvis
# DB_PASSWORD will be set via Cloud SQL Proxy auth

# Logging
LOG_LEVEL=INFO
ENABLE_ML_LOGGING=true
EOF

# Download Cloud SQL Proxy
echo "📥 Installing Cloud SQL Proxy..."
wget -q https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64 -O cloud_sql_proxy
chmod +x cloud_sql_proxy

# Start Cloud SQL Proxy in background
echo "🔗 Starting Cloud SQL Proxy..."
./cloud_sql_proxy jarvis-473803:us-central1:jarvis-learning-db --port 5432 &
PROXY_PID=$!
echo "   Cloud SQL Proxy PID: $PROXY_PID"

# Wait for proxy to be ready
sleep 5

# Start JARVIS backend
echo "🚀 Starting JARVIS backend..."
cd /home/JARVIS-AI-Agent/backend

# Create startup log
mkdir -p /var/log/jarvis
LOG_FILE="/var/log/jarvis/backend.log"

# Start backend in screen session for easy access
screen -dmS jarvis bash -c "python3 main.py --port 8010 > $LOG_FILE 2>&1"

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 10

# Health check
MAX_RETRIES=30
RETRY_COUNT=0
BACKEND_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8010/health > /dev/null; then
        BACKEND_READY=true
        break
    fi
    echo "   Waiting for backend... ($((RETRY_COUNT + 1))/$MAX_RETRIES)"
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ "$BACKEND_READY" = true ]; then
    echo "✅ JARVIS backend is ready!"
    echo "   URL: http://$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip):8010"
    echo "   Health: http://localhost:8010/health"
    echo "   Logs: $LOG_FILE"
    echo "   Screen session: screen -r jarvis"
else
    echo "❌ Backend failed to start within timeout"
    echo "   Check logs: $LOG_FILE"
    exit 1
fi

# Log memory usage
echo "💾 Memory usage:"
free -h

# Log disk usage
echo "💿 Disk usage:"
df -h

echo "✅ Startup complete at: $(date)"
echo "================================"

# Keep script running to show in startup logs
tail -f $LOG_FILE
