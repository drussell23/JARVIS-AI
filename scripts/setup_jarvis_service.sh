#!/bin/bash
# Setup JARVIS backend as a systemd service with Cloud SQL support

set -e

echo "🔧 Setting up JARVIS Backend systemd service..."

# Create systemd service file
sudo tee /etc/systemd/system/jarvis-backend.service > /dev/null << EOF
[Unit]
Description=JARVIS AI Backend
After=network.target cloud-sql-proxy.service
Requires=cloud-sql-proxy.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/backend/backend
EnvironmentFile=$HOME/backend/backend/.env.gcp
ExecStart=$HOME/backend/backend/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8010
Restart=always
RestartSec=10
StandardOutput=append:$HOME/backend/backend/jarvis.log
StandardError=append:$HOME/backend/backend/jarvis.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable jarvis-backend
sudo systemctl restart jarvis-backend

# Check status
sleep 5
if sudo systemctl is-active --quiet jarvis-backend; then
    echo "✅ JARVIS Backend service is running"
    sudo systemctl status jarvis-backend --no-pager | head -15
else
    echo "❌ JARVIS Backend service failed to start"
    sudo journalctl -u jarvis-backend -n 50 --no-pager
    exit 1
fi
