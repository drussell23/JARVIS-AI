#!/bin/bash
################################################################################
# JARVIS GCP Centralized Database Setup
# Creates Cloud SQL (PostgreSQL) and Cloud Storage for shared databases
################################################################################

set -e

PROJECT_ID="jarvis-473803"
REGION="us-central1"
ZONE="us-central1-a"

# Cloud SQL settings
SQL_INSTANCE_NAME="jarvis-learning-db"
SQL_DATABASE_NAME="jarvis_learning"
SQL_USER="jarvis"
SQL_PASSWORD="$(openssl rand -base64 32)"  # Generate secure password
SQL_TIER="db-f1-micro"  # Smallest tier (~$7/month)

# Cloud Storage settings
BUCKET_NAME="${PROJECT_ID}-jarvis-chromadb"
BACKUP_BUCKET_NAME="${PROJECT_ID}-jarvis-backups"

echo "================================================================================"
echo "🚀 JARVIS GCP Centralized Database Setup"
echo "================================================================================"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Check if user is authenticated
echo "🔐 Checking GCP authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated. Please run: gcloud auth login"
    exit 1
fi

# Set project
echo "📋 Setting active project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable sqladmin.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable compute.googleapis.com

echo "✅ APIs enabled"
echo ""

################################################################################
# CLOUD SQL SETUP
################################################################################

echo "================================================================================"
echo "📊 Setting up Cloud SQL (PostgreSQL)"
echo "================================================================================"

# Check if instance already exists
if gcloud sql instances describe $SQL_INSTANCE_NAME &>/dev/null; then
    echo "⚠️  Cloud SQL instance '$SQL_INSTANCE_NAME' already exists"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting existing instance..."
        gcloud sql instances delete $SQL_INSTANCE_NAME --quiet
    else
        echo "ℹ️  Using existing instance"
        SQL_EXISTS=true
    fi
fi

if [ "$SQL_EXISTS" != "true" ]; then
    echo "📦 Creating Cloud SQL instance (this takes 5-10 minutes)..."
    gcloud sql instances create $SQL_INSTANCE_NAME \
        --database-version=POSTGRES_15 \
        --tier=$SQL_TIER \
        --region=$REGION \
        --root-password="$SQL_PASSWORD" \
        --storage-type=SSD \
        --storage-size=10GB \
        --storage-auto-increase \
        --backup-start-time=03:00 \
        --enable-bin-log \
        --retained-backups-count=7 \
        --maintenance-window-day=SUN \
        --maintenance-window-hour=4 \
        --network=default \
        --no-assign-ip

    echo "✅ Cloud SQL instance created!"
    echo ""

    # Create database
    echo "🗄️  Creating database..."
    gcloud sql databases create $SQL_DATABASE_NAME \
        --instance=$SQL_INSTANCE_NAME

    # Create user
    echo "👤 Creating database user..."
    gcloud sql users create $SQL_USER \
        --instance=$SQL_INSTANCE_NAME \
        --password="$SQL_PASSWORD"

    echo "✅ Database and user created!"
fi

# Get connection details
echo ""
echo "📋 Cloud SQL Connection Details:"
CONNECTION_NAME=$(gcloud sql instances describe $SQL_INSTANCE_NAME --format="value(connectionName)")
PRIVATE_IP=$(gcloud sql instances describe $SQL_INSTANCE_NAME --format="value(ipAddresses[0].ipAddress)")

echo "   Instance Name: $SQL_INSTANCE_NAME"
echo "   Connection Name: $CONNECTION_NAME"
echo "   Private IP: $PRIVATE_IP"
echo "   Database: $SQL_DATABASE_NAME"
echo "   User: $SQL_USER"
echo ""

################################################################################
# CLOUD STORAGE SETUP
################################################################################

echo "================================================================================"
echo "☁️  Setting up Cloud Storage Buckets"
echo "================================================================================"

# ChromaDB bucket
if gsutil ls -b gs://$BUCKET_NAME &>/dev/null; then
    echo "⚠️  Bucket '$BUCKET_NAME' already exists"
else
    echo "📦 Creating ChromaDB bucket..."
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME

    # Set lifecycle (delete old versions after 30 days)
    cat > /tmp/lifecycle.json << 'EOF'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30, "isLive": false}
      }
    ]
  }
}
EOF
    gsutil lifecycle set /tmp/lifecycle.json gs://$BUCKET_NAME
    rm /tmp/lifecycle.json

    echo "✅ ChromaDB bucket created: gs://$BUCKET_NAME"
fi

# Backup bucket
if gsutil ls -b gs://$BACKUP_BUCKET_NAME &>/dev/null; then
    echo "⚠️  Backup bucket '$BACKUP_BUCKET_NAME' already exists"
else
    echo "📦 Creating backup bucket..."
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BACKUP_BUCKET_NAME

    # Enable versioning
    gsutil versioning set on gs://$BACKUP_BUCKET_NAME

    echo "✅ Backup bucket created: gs://$BACKUP_BUCKET_NAME"
fi

echo ""

################################################################################
# SAVE CREDENTIALS
################################################################################

echo "================================================================================"
echo "🔐 Saving Credentials"
echo "================================================================================"

CREDS_DIR="$HOME/.jarvis/gcp"
mkdir -p $CREDS_DIR

# Save database credentials
cat > $CREDS_DIR/database_config.json << EOF
{
  "cloud_sql": {
    "instance_name": "$SQL_INSTANCE_NAME",
    "connection_name": "$CONNECTION_NAME",
    "private_ip": "$PRIVATE_IP",
    "database": "$SQL_DATABASE_NAME",
    "user": "$SQL_USER",
    "password": "$SQL_PASSWORD",
    "port": 5432
  },
  "cloud_storage": {
    "chromadb_bucket": "$BUCKET_NAME",
    "backup_bucket": "$BACKUP_BUCKET_NAME"
  },
  "project_id": "$PROJECT_ID",
  "region": "$REGION"
}
EOF

chmod 600 $CREDS_DIR/database_config.json

echo "✅ Credentials saved to: $CREDS_DIR/database_config.json"
echo ""

################################################################################
# CREATE .ENV FILE
################################################################################

echo "📝 Creating .env file for local development..."

cat > .env.gcp << EOF
# JARVIS GCP Database Configuration
# Generated: $(date)

# Cloud SQL
JARVIS_DB_TYPE=cloudsql
JARVIS_DB_CONNECTION_NAME=$CONNECTION_NAME
JARVIS_DB_HOST=$PRIVATE_IP
JARVIS_DB_PORT=5432
JARVIS_DB_NAME=$SQL_DATABASE_NAME
JARVIS_DB_USER=$SQL_USER
JARVIS_DB_PASSWORD=$SQL_PASSWORD

# Cloud Storage
JARVIS_CHROMADB_BUCKET=$BUCKET_NAME
JARVIS_BACKUP_BUCKET=$BACKUP_BUCKET_NAME

# GCP Project
GCP_PROJECT_ID=$PROJECT_ID
GCP_REGION=$REGION
EOF

echo "✅ Environment file created: .env.gcp"
echo ""

################################################################################
# CLOUD PROXY SETUP
################################################################################

echo "================================================================================"
echo "🔌 Installing Cloud SQL Proxy (for local development)"
echo "================================================================================"

PROXY_PATH="/usr/local/bin/cloud_sql_proxy"

if [ -f "$PROXY_PATH" ]; then
    echo "✅ Cloud SQL Proxy already installed"
else
    echo "📥 Downloading Cloud SQL Proxy..."

    # Detect OS
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    if [ "$OS" = "darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then
            PLATFORM="darwin.arm64"
        else
            PLATFORM="darwin.amd64"
        fi
    else
        PLATFORM="linux.amd64"
    fi

    curl -o cloud_sql_proxy https://dl.google.com/cloudsql/cloud_sql_proxy.$PLATFORM
    chmod +x cloud_sql_proxy
    sudo mv cloud_sql_proxy $PROXY_PATH

    echo "✅ Cloud SQL Proxy installed"
fi

echo ""

################################################################################
# SUMMARY
################################################################################

echo "================================================================================"
echo "✅ GCP CENTRALIZED DATABASE SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "📊 Cloud SQL (PostgreSQL):"
echo "   Instance: $SQL_INSTANCE_NAME"
echo "   Database: $SQL_DATABASE_NAME"
echo "   Connection: $CONNECTION_NAME"
echo "   Private IP: $PRIVATE_IP"
echo ""
echo "☁️  Cloud Storage:"
echo "   ChromaDB Bucket: gs://$BUCKET_NAME"
echo "   Backup Bucket: gs://$BACKUP_BUCKET_NAME"
echo ""
echo "🔐 Credentials saved to:"
echo "   ~/.jarvis/gcp/database_config.json"
echo "   .env.gcp"
echo ""
echo "================================================================================"
echo "🚀 NEXT STEPS:"
echo "================================================================================"
echo ""
echo "1. Start Cloud SQL Proxy (for local development):"
echo "   cloud_sql_proxy -instances=$CONNECTION_NAME=tcp:5432"
echo ""
echo "2. Test connection:"
echo "   psql -h 127.0.0.1 -U $SQL_USER -d $SQL_DATABASE_NAME"
echo ""
echo "3. Run migration script to move local data to cloud:"
echo "   python scripts/migrate_to_cloud.py"
echo ""
echo "4. Update JARVIS to use cloud databases:"
echo "   export USE_CLOUD_DB=true"
echo ""
echo "================================================================================"
echo "💰 ESTIMATED COST: ~\$7-12/month"
echo "   (Cloud SQL db-f1-micro + Cloud Storage)"
echo "================================================================================"
