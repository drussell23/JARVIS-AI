#!/bin/bash
#
# GCP Startup Script for JARVIS Hybrid Cloud Auto-Deployment
#
# This script runs when a GCP instance is automatically created by the
# hybrid routing system for RAM overflow protection.
#
# Features:
# - Clones JARVIS repository
# - Installs dependencies
# - Sets up Cloud SQL Proxy
# - Starts backend API
# - Registers with local machine (optional)
# - Health monitoring
#

set -e  # Exit on error

echo "============================================"
echo "JARVIS GCP Auto-Deployment Startup Script"
echo "============================================"
echo "Started at: $(date)"
echo ""

# Configuration (can be overridden by instance metadata)
REPO_URL="${REPO_URL:-https://github.com/drussell23/JARVIS-AI-Agent.git}"
BRANCH="${BRANCH:-multi-monitor-support}"
BACKEND_PORT="${BACKEND_PORT:-8010}"
PROJECT_DIR="$HOME/jarvis-backend"

# Check if running on GCP
if [ ! -f /sys/class/dmi/id/product_name ] || ! grep -q "Google" /sys/class/dmi/id/product_name; then
    echo "⚠️  WARNING: This script is designed for GCP instances"
    echo "   Continuing anyway..."
fi

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================================================
# Step 1: Install system dependencies
# ============================================================================
log "📦 Installing system dependencies..."

if command_exists apt-get; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3.10 \
        python3.10-venv \
        python3-pip \
        git \
        curl \
        wget \
        jq \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-dev \
        postgresql-client \
        || log "⚠️  Some packages failed to install"
else
    log "❌ apt-get not found. This script requires Ubuntu/Debian."
    exit 1
fi

log "✅ System dependencies installed"

# ============================================================================
# Step 2: Clone repository
# ============================================================================
log "📥 Cloning JARVIS repository..."

if [ -d "$PROJECT_DIR" ]; then
    log "⚠️  Project directory exists, updating..."
    cd "$PROJECT_DIR"
    git fetch --all
    git reset --hard "origin/$BRANCH"
else
    git clone -b "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

log "✅ Repository cloned/updated"

# ============================================================================
# Step 3: Set up Python virtual environment
# ============================================================================
log "🐍 Setting up Python environment..."

cd "$PROJECT_DIR/backend"

if [ ! -d "venv" ]; then
    python3.10 -m venv venv
fi

source venv/bin/activate

# Install requirements
if [ -f "requirements-cloud.txt" ]; then
    log "📦 Installing cloud requirements..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements-cloud.txt
elif [ -f "requirements.txt" ]; then
    log "📦 Installing standard requirements..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
else
    log "⚠️  No requirements file found"
fi

log "✅ Python environment ready"

# ============================================================================
# Step 4: Set up Cloud SQL Proxy
# ============================================================================
log "☁️  Setting up Cloud SQL Proxy..."

# Install Cloud SQL Proxy
if [ ! -f "$HOME/.local/bin/cloud-sql-proxy" ]; then
    log "📥 Downloading Cloud SQL Proxy..."
    mkdir -p "$HOME/.local/bin"
    curl -o "$HOME/.local/bin/cloud-sql-proxy" \
        https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.2/cloud-sql-proxy.linux.amd64
    chmod +x "$HOME/.local/bin/cloud-sql-proxy"
fi

# Check for GCP database config (should be in instance metadata or mounted)
if [ -f "$HOME/.jarvis/gcp/database_config.json" ]; then
    log "✅ Found database config"

    CONNECTION_NAME=$(jq -r '.cloud_sql.connection_name' "$HOME/.jarvis/gcp/database_config.json")
    DB_PASSWORD=$(jq -r '.cloud_sql.password' "$HOME/.jarvis/gcp/database_config.json")

    # Create .env.gcp file
    cat > "$PROJECT_DIR/backend/.env.gcp" <<EOF
JARVIS_DB_TYPE=cloudsql
JARVIS_DB_CONNECTION_NAME=$CONNECTION_NAME
JARVIS_DB_HOST=127.0.0.1
JARVIS_DB_PORT=5432
JARVIS_DB_NAME=jarvis_learning
JARVIS_DB_USER=jarvis
JARVIS_DB_PASSWORD=$DB_PASSWORD
JARVIS_HYBRID_MODE=true
GCP_INSTANCE=true
EOF

    log "✅ Environment configured for Cloud SQL"

    # Start Cloud SQL Proxy in background
    log "🚀 Starting Cloud SQL Proxy..."
    nohup "$HOME/.local/bin/cloud-sql-proxy" "$CONNECTION_NAME" \
        --port 5432 \
        > "$HOME/cloud-sql-proxy.log" 2>&1 &

    sleep 3

    if pgrep -f cloud-sql-proxy > /dev/null; then
        log "✅ Cloud SQL Proxy running"
    else
        log "⚠️  Cloud SQL Proxy failed to start"
        log "   Check logs: $HOME/cloud-sql-proxy.log"
    fi
else
    log "⚠️  No database config found, will use SQLite"
    cat > "$PROJECT_DIR/backend/.env.gcp" <<EOF
JARVIS_HYBRID_MODE=true
GCP_INSTANCE=true
EOF
fi

# ============================================================================
# Step 5: Start JARVIS backend
# ============================================================================
log "🚀 Starting JARVIS backend API..."

cd "$PROJECT_DIR/backend"

# Load environment
if [ -f ".env.gcp" ]; then
    set -a
    source .env.gcp
    set +a
fi

# Start uvicorn in background
nohup venv/bin/python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    --log-level info \
    > "$HOME/jarvis-backend.log" 2>&1 &

BACKEND_PID=$!
log "✅ Backend started (PID: $BACKEND_PID)"

# ============================================================================
# Step 6: Health check
# ============================================================================
log "🏥 Waiting for backend to be ready..."

MAX_RETRIES=30
RETRY_COUNT=0
HEALTH_OK=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    sleep 2

    if curl -sf "http://localhost:$BACKEND_PORT/health" > /dev/null; then
        HEALTH_OK=true
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    log "   Retry $RETRY_COUNT/$MAX_RETRIES..."
done

if [ "$HEALTH_OK" = true ]; then
    log "✅ Backend health check passed!"

    # Get instance IP
    INSTANCE_IP=$(curl -sf http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H "Metadata-Flavor: Google" || echo "unknown")

    log ""
    log "============================================"
    log "🎉 JARVIS GCP Instance Ready!"
    log "============================================"
    log "Instance IP: $INSTANCE_IP"
    log "Backend API: http://$INSTANCE_IP:$BACKEND_PORT"
    log "Health: http://$INSTANCE_IP:$BACKEND_PORT/health"
    log "Hybrid Status: http://$INSTANCE_IP:$BACKEND_PORT/hybrid/status"
    log ""
    log "Logs:"
    log "- Backend: $HOME/jarvis-backend.log"
    log "- Cloud SQL Proxy: $HOME/cloud-sql-proxy.log"
    log "- Startup: /var/log/syslog (search for 'jarvis')"
    log "============================================"

    # Optionally register with local machine (if webhook URL provided)
    if [ -n "$LOCAL_WEBHOOK_URL" ]; then
        log "📡 Registering with local machine..."
        curl -X POST "$LOCAL_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"instance_ip\": \"$INSTANCE_IP\", \"backend_port\": $BACKEND_PORT, \"status\": \"ready\"}" \
            || log "⚠️  Registration failed"
    fi

else
    log "❌ Backend failed to start"
    log "   Last 50 lines of backend log:"
    tail -50 "$HOME/jarvis-backend.log"
    exit 1
fi

# ============================================================================
# Step 7: Keep running and monitor
# ============================================================================
log "👀 Monitoring backend process..."

# Keep the script running and monitor the backend
while true; do
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        log "❌ Backend process died! Attempting restart..."

        cd "$PROJECT_DIR/backend"
        nohup venv/bin/python -m uvicorn main:app \
            --host 0.0.0.0 \
            --port "$BACKEND_PORT" \
            --log-level info \
            > "$HOME/jarvis-backend.log" 2>&1 &

        BACKEND_PID=$!
        log "✅ Backend restarted (PID: $BACKEND_PID)"
    fi

    sleep 60  # Check every minute
done
