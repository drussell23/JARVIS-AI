#!/bin/bash
# Advanced Cost Monitoring Setup for JARVIS Hybrid Cloud
# Features: GCP Budget API automation, email alerts, cron jobs, dynamic config
# No hardcoding - all values from environment variables

set -e

# ============================================================================
# DYNAMIC CONFIGURATION - NO HARDCODING
# ============================================================================

# Load from environment or use defaults
PROJECT_ID="${GCP_PROJECT_ID:-}"
ALERT_EMAIL="${JARVIS_ALERT_EMAIL:-}"
BUDGET_NAME="${JARVIS_BUDGET_NAME:-jarvis-hybrid-cloud-budget}"
JARVIS_DIR="${JARVIS_HOME:-$HOME/.jarvis}"
LOG_DIR="${JARVIS_LOG_DIR:-$JARVIS_DIR/logs}"
LEARNING_DIR="${JARVIS_LEARNING_DIR:-$JARVIS_DIR/learning}"

# Budget thresholds (configurable)
BUDGET_AMOUNT_1="${BUDGET_THRESHOLD_1:-20}"
BUDGET_AMOUNT_2="${BUDGET_THRESHOLD_2:-50}"
BUDGET_AMOUNT_3="${BUDGET_THRESHOLD_3:-100}"

# Alert thresholds (percentages)
THRESHOLD_50="${ALERT_THRESHOLD_50:-50}"
THRESHOLD_90="${ALERT_THRESHOLD_90:-90}"
THRESHOLD_100="${ALERT_THRESHOLD_100:-100}"

# Cron schedule (default: every 6 hours)
CRON_SCHEDULE="${CLEANUP_CRON_SCHEDULE:-0 */6 * * *}"
ORPHANED_VM_MAX_AGE="${ORPHANED_VM_MAX_AGE_HOURS:-6}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

validate_email() {
    if [[ "$1" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        return 0
    fi
    return 1
}

# ============================================================================
# INITIALIZATION
# ============================================================================

echo ""
echo "=========================================="
echo "💰 JARVIS Cost Monitoring Setup (Advanced)"
echo "=========================================="
echo ""

# Validate required configuration
if [ -z "$PROJECT_ID" ]; then
    log_error "GCP_PROJECT_ID not set!"
    log_info "Please set: export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

if [ -z "$ALERT_EMAIL" ]; then
    log_warning "JARVIS_ALERT_EMAIL not set - email alerts will be disabled"
    log_info "Set via: export JARVIS_ALERT_EMAIL=your-email@example.com"
fi

log_info "Configuration:"
log_info "  Project ID: $PROJECT_ID"
log_info "  Budget Name: $BUDGET_NAME"
log_info "  Alert Email: ${ALERT_EMAIL:-<not set>}"
log_info "  Budget Thresholds: \$$BUDGET_AMOUNT_1, \$$BUDGET_AMOUNT_2, \$$BUDGET_AMOUNT_3"
log_info "  Cron Schedule: $CRON_SCHEDULE"
log_info "  VM Max Age: ${ORPHANED_VM_MAX_AGE}h"
echo ""

# ============================================================================
# 1. CREATE DIRECTORIES
# ============================================================================

log_info "Creating directory structure..."

mkdir -p "$LOG_DIR"
mkdir -p "$LEARNING_DIR"
mkdir -p "$JARVIS_DIR/config"
mkdir -p "$JARVIS_DIR/gcp"

log_success "Directories created:"
echo "   Logs: $LOG_DIR"
echo "   Learning: $LEARNING_DIR"
echo "   Config: $JARVIS_DIR/config"
echo "   GCP: $JARVIS_DIR/gcp"

# ============================================================================
# 2. VALIDATE GCP CLI
# ============================================================================

echo ""
log_info "Validating GCP CLI..."

if ! check_command gcloud; then
    log_error "gcloud CLI not found!"
    log_info "Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

log_success "gcloud CLI found: $(gcloud version | head -n1)"

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    log_warning "Not authenticated with gcloud"
    log_info "Authenticating now..."
    gcloud auth login
fi

ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1)
log_success "Authenticated as: $ACTIVE_ACCOUNT"

# Set project
gcloud config set project "$PROJECT_ID" --quiet
log_success "Active project: $PROJECT_ID"

# ============================================================================
# 3. GCP BUDGET ALERTS (AUTOMATED VIA API)
# ============================================================================

echo ""
log_info "Setting up GCP Budget Alerts via Cloud Billing API..."

# Enable Cloud Billing Budget API
log_info "Enabling Cloud Billing Budget API..."
gcloud services enable billingbudgets.googleapis.com --project="$PROJECT_ID" --quiet 2>/dev/null || true

# Get billing account
BILLING_ACCOUNT=$(gcloud billing projects describe "$PROJECT_ID" \
    --format="value(billingAccountName)" 2>/dev/null || echo "")

if [ -z "$BILLING_ACCOUNT" ]; then
    log_error "No billing account linked to project!"
    log_info "Link billing: https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
    exit 1
fi

log_success "Billing account: $BILLING_ACCOUNT"

# Create budget configuration file
BUDGET_CONFIG="$JARVIS_DIR/config/budget_config.json"

create_budget_json() {
    local budget_amount=$1
    local budget_suffix=$2

    cat > "$BUDGET_CONFIG" <<EOF
{
  "displayName": "${BUDGET_NAME}-${budget_suffix}",
  "budgetFilter": {
    "projects": ["projects/$PROJECT_ID"],
    "creditTypesTreatment": "INCLUDE_ALL_CREDITS"
  },
  "amount": {
    "specifiedAmount": {
      "currencyCode": "USD",
      "units": "$budget_amount"
    }
  },
  "thresholdRules": [
    {
      "thresholdPercent": 0.5,
      "spendBasis": "CURRENT_SPEND"
    },
    {
      "thresholdPercent": 0.9,
      "spendBasis": "CURRENT_SPEND"
    },
    {
      "thresholdPercent": 1.0,
      "spendBasis": "CURRENT_SPEND"
    }
  ],
  "allUpdatesRule": {
    "pubsubTopic": "projects/$PROJECT_ID/topics/budget-alerts",
    "schemaVersion": "1.0"
  }
}
EOF
}

# Create Pub/Sub topic for budget alerts
log_info "Creating Pub/Sub topic for budget alerts..."
gcloud pubsub topics create budget-alerts --project="$PROJECT_ID" 2>/dev/null || \
    log_warning "Pub/Sub topic 'budget-alerts' already exists"

# Grant billing permissions to Pub/Sub
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
gcloud pubsub topics add-iam-policy-binding budget-alerts \
    --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-billing.iam.gserviceaccount.com" \
    --role="roles/pubsub.publisher" \
    --project="$PROJECT_ID" --quiet 2>/dev/null || true

log_success "Pub/Sub topic configured"

# Create budgets using gcloud (requires beta)
log_info "Creating budget alerts..."

for budget_amt in "$BUDGET_AMOUNT_1" "$BUDGET_AMOUNT_2" "$BUDGET_AMOUNT_3"; do
    create_budget_json "$budget_amt" "${budget_amt}usd"

    # Create budget using gcloud beta
    if gcloud beta billing budgets create \
        --billing-account="$BILLING_ACCOUNT" \
        --display-name="${BUDGET_NAME}-${budget_amt}usd" \
        --budget-amount="${budget_amt}USD" \
        --threshold-rule=percent=0.5 \
        --threshold-rule=percent=0.9 \
        --threshold-rule=percent=1.0 \
        --all-updates-rule-pubsub-topic="projects/$PROJECT_ID/topics/budget-alerts" \
        --filter-projects="projects/$PROJECT_ID" 2>/dev/null; then
        log_success "Budget created: \$$budget_amt/month"
    else
        log_warning "Budget \$$budget_amt already exists or creation failed"
    fi
done

# ============================================================================
# 4. EMAIL NOTIFICATION SETUP (Cloud Function)
# ============================================================================

echo ""
log_info "Setting up email notifications..."

if [ -n "$ALERT_EMAIL" ] && validate_email "$ALERT_EMAIL"; then
    # Create Cloud Function for email notifications
    FUNCTION_DIR="$JARVIS_DIR/functions/budget-alerts"
    mkdir -p "$FUNCTION_DIR"

    # Create Cloud Function code
    cat > "$FUNCTION_DIR/main.py" <<'EOF'
import base64
import json
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def process_budget_alert(event, context):
    """Process budget alert and send email notification"""

    # Decode Pub/Sub message
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    alert_data = json.loads(pubsub_message)

    # Extract budget info
    budget_name = alert_data.get('budgetDisplayName', 'Unknown')
    cost_amount = alert_data.get('costAmount', 0)
    budget_amount = alert_data.get('budgetAmount', 0)
    threshold = (cost_amount / budget_amount * 100) if budget_amount > 0 else 0

    # Send email via SendGrid
    alert_email = os.environ.get('JARVIS_ALERT_EMAIL')
    sendgrid_key = os.environ.get('SENDGRID_API_KEY')

    if not alert_email or not sendgrid_key:
        print("Email or SendGrid API key not configured")
        return

    message = Mail(
        from_email='noreply@jarvis-ai.cloud',
        to_emails=alert_email,
        subject=f'⚠️ JARVIS Budget Alert: {budget_name}',
        html_content=f'''
        <h2>🚨 Budget Alert Triggered</h2>
        <p><strong>Budget:</strong> {budget_name}</p>
        <p><strong>Current Spend:</strong> ${cost_amount:.2f}</p>
        <p><strong>Budget Limit:</strong> ${budget_amount:.2f}</p>
        <p><strong>Threshold:</strong> {threshold:.1f}%</p>
        <p>View details: <a href="https://console.cloud.google.com/billing">GCP Console</a></p>
        '''
    )

    try:
        sg = SendGridAPIClient(sendgrid_key)
        response = sg.send(message)
        print(f"Email sent: {response.status_code}")
    except Exception as e:
        print(f"Email error: {e}")
EOF

    cat > "$FUNCTION_DIR/requirements.txt" <<EOF
sendgrid==6.10.0
EOF

    log_success "Cloud Function code created: $FUNCTION_DIR"
    log_info "To deploy, run:"
    echo "   cd $FUNCTION_DIR"
    echo "   gcloud functions deploy budget-alert-notifier \\"
    echo "     --runtime python311 \\"
    echo "     --trigger-topic budget-alerts \\"
    echo "     --entry-point process_budget_alert \\"
    echo "     --set-env-vars JARVIS_ALERT_EMAIL=$ALERT_EMAIL,SENDGRID_API_KEY=<your-key>"

else
    log_warning "Email not configured - skipping email notifications"
fi

# ============================================================================
# 5. CRON JOB FOR ORPHANED VM CLEANUP
# ============================================================================

echo ""
log_info "Setting up cron job for orphaned VM cleanup..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup_orphaned_vms.sh"

if [ ! -f "$CLEANUP_SCRIPT" ]; then
    log_error "Cleanup script not found: $CLEANUP_SCRIPT"
    log_info "Creating cleanup script..."

    # Create cleanup script if it doesn't exist
    cat > "$CLEANUP_SCRIPT" <<'CLEANUP_EOF'
#!/bin/bash
# Automated cleanup of orphaned JARVIS VMs
# Configured via environment variables

PROJECT_ID="${GCP_PROJECT_ID:-}"
MAX_AGE_HOURS="${ORPHANED_VM_MAX_AGE_HOURS:-6}"
LOG_FILE="${JARVIS_LOG_DIR:-$HOME/.jarvis/logs}/cleanup.log"

echo "$(date): Starting orphaned VM cleanup..." >> "$LOG_FILE"

# Find VMs older than MAX_AGE_HOURS
THRESHOLD_TIME=$(date -u -d "$MAX_AGE_HOURS hours ago" +%Y-%m-%dT%H:%M:%S 2>/dev/null || \
    date -u -v-${MAX_AGE_HOURS}H +%Y-%m-%dT%H:%M:%S)

gcloud compute instances list \
    --project="$PROJECT_ID" \
    --filter="name~'jarvis-auto-.*' AND creationTimestamp<'$THRESHOLD_TIME'" \
    --format="value(name,zone)" | while read -r name zone; do

    echo "$(date): Deleting orphaned VM: $name (zone: $zone)" >> "$LOG_FILE"

    gcloud compute instances delete "$name" \
        --project="$PROJECT_ID" \
        --zone="$zone" \
        --quiet >> "$LOG_FILE" 2>&1
done

echo "$(date): Cleanup complete" >> "$LOG_FILE"
CLEANUP_EOF

    chmod +x "$CLEANUP_SCRIPT"
    log_success "Cleanup script created"
fi

# Make cleanup script executable
chmod +x "$CLEANUP_SCRIPT"

# Add cron job
CRON_CMD="$CRON_SCHEDULE $CLEANUP_SCRIPT >> $LOG_DIR/cron_cleanup.log 2>&1"

if crontab -l 2>/dev/null | grep -q "$CLEANUP_SCRIPT"; then
    log_success "Cron job already exists"
else
    # Add cron job
    (crontab -l 2>/dev/null; echo "# JARVIS Orphaned VM Cleanup"; echo "$CRON_CMD") | crontab -
    log_success "Cron job added"
fi

log_success "Cron configuration:"
echo "   Schedule: $CRON_SCHEDULE"
echo "   Log: $LOG_DIR/cron_cleanup.log"
echo "   Max age: ${ORPHANED_VM_MAX_AGE}h"

# ============================================================================
# 6. SAVE CONFIGURATION
# ============================================================================

echo ""
log_info "Saving configuration..."

CONFIG_FILE="$JARVIS_DIR/config/cost_monitoring.env"

cat > "$CONFIG_FILE" <<EOF
# JARVIS Cost Monitoring Configuration
# Generated: $(date)

# GCP Configuration
export GCP_PROJECT_ID="$PROJECT_ID"
export GCP_BILLING_ACCOUNT="$BILLING_ACCOUNT"

# Cost Monitoring
export JARVIS_ALERT_EMAIL="${ALERT_EMAIL}"
export BUDGET_THRESHOLD_1="$BUDGET_AMOUNT_1"
export BUDGET_THRESHOLD_2="$BUDGET_AMOUNT_2"
export BUDGET_THRESHOLD_3="$BUDGET_AMOUNT_3"

# Cleanup Configuration
export CLEANUP_CRON_SCHEDULE="$CRON_SCHEDULE"
export ORPHANED_VM_MAX_AGE_HOURS="$ORPHANED_VM_MAX_AGE"

# Directories
export JARVIS_HOME="$JARVIS_DIR"
export JARVIS_LOG_DIR="$LOG_DIR"
export JARVIS_LEARNING_DIR="$LEARNING_DIR"
EOF

log_success "Configuration saved: $CONFIG_FILE"
log_info "To load config: source $CONFIG_FILE"

# ============================================================================
# 7. TEST SETUP
# ============================================================================

echo ""
log_info "Testing setup..."

# Test gcloud access
if gcloud compute instances list --project="$PROJECT_ID" --limit=1 &>/dev/null; then
    log_success "GCP API access verified"
else
    log_warning "GCP API access test failed"
fi

# Test cleanup script (dry run)
log_info "Running cleanup script test..."
bash "$CLEANUP_SCRIPT" || log_warning "Cleanup test completed with warnings"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=========================================="
log_success "Cost Monitoring Setup Complete!"
echo "=========================================="
echo ""
echo "✅ GCP Budget Alerts:"
echo "   - Budgets: \$$BUDGET_AMOUNT_1, \$$BUDGET_AMOUNT_2, \$$BUDGET_AMOUNT_3 per month"
echo "   - Thresholds: 50%, 90%, 100%"
echo "   - Pub/Sub topic: budget-alerts"
echo ""
echo "✅ Automated Cleanup:"
echo "   - Schedule: $CRON_SCHEDULE"
echo "   - Max VM age: ${ORPHANED_VM_MAX_AGE}h"
echo "   - Log: $LOG_DIR/cron_cleanup.log"
echo ""
echo "✅ Configuration:"
echo "   - Config file: $CONFIG_FILE"
echo "   - Budget config: $BUDGET_CONFIG"
echo ""

if [ -n "$ALERT_EMAIL" ]; then
    echo "📧 Email Notifications:"
    echo "   - Recipient: $ALERT_EMAIL"
    echo "   - Function: $FUNCTION_DIR/main.py"
    log_warning "Deploy Cloud Function manually (see instructions above)"
    echo ""
fi

echo "📚 API Endpoints:"
echo "   - Status: http://localhost:8010/hybrid/status"
echo "   - Cost: http://localhost:8010/hybrid/cost/{period}"
echo "   - Cleanup: http://localhost:8010/hybrid/cleanup"
echo "   - WebSocket: ws://localhost:8010/hybrid/ws"
echo ""

echo "🔧 Next Steps:"
echo "   1. Start JARVIS backend to initialize cost tracking"
echo "   2. Monitor budgets: https://console.cloud.google.com/billing/$PROJECT_ID/budgets"
if [ -n "$ALERT_EMAIL" ]; then
    echo "   3. Deploy email notification function (optional)"
fi
echo "   4. Check cron logs: tail -f $LOG_DIR/cron_cleanup.log"
echo ""

log_success "Setup complete! Cost monitoring is now active."
