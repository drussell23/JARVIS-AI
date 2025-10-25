#!/bin/bash
# Setup Cost Monitoring for JARVIS Hybrid Cloud
# This script configures GCP budget alerts and cron jobs for orphaned VM cleanup

set -e

echo "💰 Setting up JARVIS Cost Monitoring..."

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-jarvis-473803}"
ALERT_EMAIL="${JARVIS_ALERT_EMAIL:-}"
BUDGET_NAME="jarvis-hybrid-cloud-budget"

# Directories
JARVIS_DIR="$HOME/.jarvis"
LOG_DIR="$JARVIS_DIR/logs"
LEARNING_DIR="$JARVIS_DIR/learning"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$LEARNING_DIR"

echo "📁 Created directories:"
echo "   Logs: $LOG_DIR"
echo "   Learning: $LEARNING_DIR"

# ============================================================================
# 1. GCP BUDGET ALERTS
# ============================================================================

echo ""
echo "📊 Setting up GCP Budget Alerts..."

if ! command -v gcloud &> /dev/null; then
    echo "⚠️  gcloud CLI not found - skipping budget setup"
    echo "   Install: https://cloud.google.com/sdk/docs/install"
else
    # Check if billing account is configured
    BILLING_ACCOUNT=$(gcloud billing accounts list --filter="open=true" --format="value(name)" --limit=1 2>/dev/null || echo "")

    if [ -z "$BILLING_ACCOUNT" ]; then
        echo "⚠️  No billing account found - cannot create budget alerts"
        echo "   Configure billing: https://console.cloud.google.com/billing"
    else
        echo "✅ Billing account found: $BILLING_ACCOUNT"

        # Create budget with alerts at $20, $50, $100/month
        echo "Creating budget alerts..."

        # Note: gcloud doesn't have direct budget creation commands
        # Must use Cloud Console or API
        echo "⚠️  Budget alerts must be configured manually via GCP Console:"
        echo ""
        echo "   1. Visit: https://console.cloud.google.com/billing/${BILLING_ACCOUNT#billingAccounts/}/budgets"
        echo "   2. Click 'CREATE BUDGET'"
        echo "   3. Configure:"
        echo "      - Name: $BUDGET_NAME"
        echo "      - Projects: $PROJECT_ID"
        echo "      - Amount: Custom (\$20/month)"
        echo "      - Thresholds: 50%, 90%, 100%"
        if [ -n "$ALERT_EMAIL" ]; then
            echo "      - Email: $ALERT_EMAIL"
        else
            echo "      - Email: (your email)"
        fi
        echo "   4. Repeat for \$50 and \$100 budgets"
        echo ""
        echo "   Documentation: https://cloud.google.com/billing/docs/how-to/budgets"
    fi
fi

# ============================================================================
# 2. CRON JOB FOR ORPHANED VM CLEANUP
# ============================================================================

echo ""
echo "⏰ Setting up cron job for orphaned VM cleanup..."

# Path to cleanup script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup_orphaned_vms.sh"

if [ ! -f "$CLEANUP_SCRIPT" ]; then
    echo "⚠️  Cleanup script not found: $CLEANUP_SCRIPT"
    exit 1
fi

# Make script executable
chmod +x "$CLEANUP_SCRIPT"

# Check if cron job already exists
CRON_CMD="0 */6 * * * $CLEANUP_SCRIPT >> $LOG_DIR/cron_cleanup.log 2>&1"

if crontab -l 2>/dev/null | grep -q "$CLEANUP_SCRIPT"; then
    echo "✅ Cron job already exists"
else
    # Add cron job (every 6 hours)
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✅ Cron job added: Run cleanup every 6 hours"
fi

echo "   Schedule: 0 */6 * * * (every 6 hours)"
echo "   Log: $LOG_DIR/cron_cleanup.log"

# ============================================================================
# 3. INITIALIZE COST TRACKING DATABASE
# ============================================================================

echo ""
echo "💾 Initializing cost tracking database..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "⚠️  Python 3 not found - cannot initialize database"
else
    # Initialize database via backend API
    BACKEND_URL="http://localhost:8010"

    if curl -s "$BACKEND_URL/health" > /dev/null 2>&1; then
        echo "✅ Backend is running"

        # Initialize cost tracking
        RESPONSE=$(curl -s -X POST "$BACKEND_URL/hybrid/initialize")
        echo "✅ Cost tracking initialized: $RESPONSE"

    else
        echo "ℹ️  Backend not running - database will initialize on first startup"
    fi
fi

# ============================================================================
# 4. ENVIRONMENT VARIABLES
# ============================================================================

echo ""
echo "🔧 Environment variable setup..."

ENV_FILE="$HOME/.jarvis/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file..."
    cat > "$ENV_FILE" <<EOF
# JARVIS Environment Configuration

# GCP Configuration
GCP_PROJECT_ID=$PROJECT_ID

# Cost Monitoring
JARVIS_ALERT_EMAIL=${ALERT_EMAIL:-your-email@example.com}

# Add your email for cost alerts
EOF
    echo "✅ Created: $ENV_FILE"
    echo "   ⚠️  Update JARVIS_ALERT_EMAIL with your email address"
else
    echo "✅ .env file exists: $ENV_FILE"
fi

# ============================================================================
# 5. TEST CLEANUP SCRIPT
# ============================================================================

echo ""
echo "🧪 Testing orphaned VM cleanup script..."

if [ -f "$CLEANUP_SCRIPT" ]; then
    echo "Running test (dry run)..."
    bash "$CLEANUP_SCRIPT" || echo "⚠️  Test run completed with warnings (normal if no VMs found)"
    echo "✅ Cleanup script test complete"
else
    echo "⚠️  Cleanup script not found"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=========================================="
echo "✅ Cost Monitoring Setup Complete!"
echo "=========================================="
echo ""
echo "✅ Directories created:"
echo "   - Logs: $LOG_DIR"
echo "   - Learning: $LEARNING_DIR"
echo ""
echo "✅ Cron job configured:"
echo "   - Runs every 6 hours"
echo "   - Checks for orphaned VMs older than 6 hours"
echo "   - Logs to: $LOG_DIR/cron_cleanup.log"
echo ""
echo "⚠️  Manual steps required:"
echo "   1. Configure GCP Budget Alerts (see instructions above)"
echo "   2. Update .env file with your email: $ENV_FILE"
echo "   3. Test: bash $CLEANUP_SCRIPT"
echo ""
echo "📚 Documentation:"
echo "   - Cost API: http://localhost:8010/hybrid/cost"
echo "   - Orphaned VMs: http://localhost:8010/hybrid/orphaned-vms"
echo "   - Status: http://localhost:8010/hybrid/status"
echo ""
echo "💡 Next steps:"
echo "   - Start JARVIS backend to initialize cost tracking database"
echo "   - Monitor costs via API endpoints"
echo "   - Check logs regularly: $LOG_DIR"
echo ""
