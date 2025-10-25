#!/bin/bash
set -e
echo "🚀 JARVIS GCP Auto-Deployment Starting..."

# Install dependencies
sudo apt-get update -qq
sudo apt-get install -y -qq python3.10 python3.10-venv python3-pip curl jq build-essential postgresql-client

# Download deployment package from Cloud Storage
PROJECT_DIR="$HOME/jarvis-backend"
DEPLOYMENT_BUCKET="gs://jarvis-473803-deployments"

echo "📥 Downloading latest deployment from Cloud Storage..."

# Get latest commit for this branch
LATEST_COMMIT=$(gsutil cat $DEPLOYMENT_BUCKET/latest-multi-monitor-support.txt 2>/dev/null || echo "")

if [ -z "$LATEST_COMMIT" ]; then
    echo "⚠️  No deployment found for branch multi-monitor-support, falling back to git clone..."
    REPO_URL="https://github.com/drussell23/JARVIS-AI-Agent.git"
    if [ -d "$PROJECT_DIR" ]; then
        cd "$PROJECT_DIR" && git fetch --all && git reset --hard origin/multi-monitor-support
    else
        git clone -b multi-monitor-support $REPO_URL "$PROJECT_DIR"
    fi
else
    echo "📦 Using deployment: $LATEST_COMMIT"

    # Download and extract deployment package
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"

    gsutil cp $DEPLOYMENT_BUCKET/jarvis-$LATEST_COMMIT.tar.gz /tmp/jarvis-deployment.tar.gz
    tar -xzf /tmp/jarvis-deployment.tar.gz -C "$PROJECT_DIR"
    rm /tmp/jarvis-deployment.tar.gz

    echo "✅ Deployment package extracted"
fi

# Setup Python environment
cd "$PROJECT_DIR/backend"
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
fi
source venv/bin/activate
pip install --quiet --upgrade pip
if [ -f "requirements-cloud.txt" ]; then
    pip install --quiet -r requirements-cloud.txt
elif [ -f "requirements.txt" ]; then
    pip install --quiet -r requirements.txt
fi

# Setup Cloud SQL Proxy
if [ ! -f "$HOME/.local/bin/cloud-sql-proxy" ]; then
    mkdir -p "$HOME/.local/bin"
    curl -o "$HOME/.local/bin/cloud-sql-proxy" https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.2/cloud-sql-proxy.linux.amd64
    chmod +x "$HOME/.local/bin/cloud-sql-proxy"
fi

# Configure environment
cat > "$PROJECT_DIR/backend/.env.gcp" <<EOF
JARVIS_HYBRID_MODE=true
GCP_INSTANCE=true
JARVIS_DB_TYPE=cloudsql
EOF

# Start Cloud SQL Proxy (if config available)
if [ -f "$HOME/.jarvis/gcp/database_config.json" ]; then
    CONNECTION_NAME=$(jq -r '.cloud_sql.connection_name' "$HOME/.jarvis/gcp/database_config.json")
    nohup "$HOME/.local/bin/cloud-sql-proxy" "$CONNECTION_NAME" --port 5432 > "$HOME/cloud-sql-proxy.log" 2>&1 &
    sleep 2
fi

# Start backend
cd "$PROJECT_DIR/backend"
source .env.gcp
nohup venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8010 --log-level info > "$HOME/jarvis-backend.log" 2>&1 &

# Wait for health check
for i in {1..30}; do
    sleep 2
    if curl -sf http://localhost:8010/health > /dev/null; then
        INSTANCE_IP=$(curl -sf http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H "Metadata-Flavor: Google" || echo "unknown")
        echo "✅ JARVIS Ready at http://$INSTANCE_IP:8010"
        exit 0
    fi
done

echo "❌ Backend failed to start"
tail -50 "$HOME/jarvis-backend.log"
exit 1
