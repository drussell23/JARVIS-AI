#!/bin/bash
# Setup Cloud SQL Proxy as a systemd service on GCP VM

set -e

echo "🔧 Setting up Cloud SQL Proxy systemd service..."

# Check if database config exists
if [ ! -f ~/.jarvis/gcp/database_config.json ]; then
    echo "❌ Database config not found: ~/.jarvis/gcp/database_config.json"
    exit 1
fi

# Extract connection name
CONNECTION_NAME=$(jq -r '.cloud_sql.connection_name' ~/.jarvis/gcp/database_config.json)
echo "Connection: $CONNECTION_NAME"

# Create systemd service file
sudo tee /etc/systemd/system/cloud-sql-proxy.service > /dev/null << EOF
[Unit]
Description=Cloud SQL Proxy
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=$HOME/.local/bin/cloud-sql-proxy $CONNECTION_NAME --port 5432
Restart=always
RestartSec=10
StandardOutput=append:$HOME/.jarvis/cloud-sql-proxy.log
StandardError=append:$HOME/.jarvis/cloud-sql-proxy.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable cloud-sql-proxy
sudo systemctl restart cloud-sql-proxy

# Check status
sleep 2
if sudo systemctl is-active --quiet cloud-sql-proxy; then
    echo "✅ Cloud SQL Proxy service is running"
    sudo systemctl status cloud-sql-proxy --no-pager
else
    echo "❌ Cloud SQL Proxy service failed to start"
    sudo journalctl -u cloud-sql-proxy -n 20 --no-pager
    exit 1
fi
