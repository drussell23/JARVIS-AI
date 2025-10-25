#!/bin/bash
# Cleanup orphaned JARVIS Spot VMs
# Run this daily via cron or manually to cleanup forgotten VMs
# Enhanced with cost tracking and notifications

set -e

PROJECT_ID="${GCP_PROJECT_ID:-jarvis-473803}"
ZONE="us-central1-a"
MAX_AGE_HOURS=6  # Delete VMs older than 6 hours
LOG_FILE="${HOME}/.jarvis/logs/vm_cleanup_$(date +%Y%m%d).log"
COST_DB="${HOME}/.jarvis/learning/cost_tracking.db"

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "🔍 Checking for orphaned JARVIS VMs..."

# Get all jarvis-auto VMs with their creation time
VMS=$(gcloud compute instances list \
  --project="$PROJECT_ID" \
  --filter="name~'jarvis-auto-.*'" \
  --format="csv[no-heading](name,creationTimestamp)" 2>/dev/null || echo "")

if [ -z "$VMS" ]; then
  log "✅ No orphaned VMs found"
  exit 0
fi

# Track orphaned VMs for reporting
ORPHANED_COUNT=0
ORPHANED_VMS_LIST=""

# Current timestamp
NOW=$(date +%s)

# Check each VM
while IFS=, read -r VM_NAME CREATED_AT; do
  # Convert creation time to Unix timestamp
  CREATED_TIMESTAMP=$(date -d "$CREATED_AT" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "$CREATED_AT" +%s 2>/dev/null || echo "0")

  if [ "$CREATED_TIMESTAMP" = "0" ]; then
    echo "⚠️  Could not parse timestamp for $VM_NAME, skipping"
    continue
  fi

  # Calculate age in hours
  AGE_SECONDS=$((NOW - CREATED_TIMESTAMP))
  AGE_HOURS=$((AGE_SECONDS / 3600))

  log "Found VM: $VM_NAME (age: ${AGE_HOURS}h)"

  if [ $AGE_HOURS -ge $MAX_AGE_HOURS ]; then
    log "⚠️  VM is older than ${MAX_AGE_HOURS}h - deleting..."

    gcloud compute instances delete "$VM_NAME" \
      --zone="$ZONE" \
      --project="$PROJECT_ID" \
      --quiet

    log "✅ Deleted orphaned VM: $VM_NAME"

    # Track for cost reporting
    ORPHANED_COUNT=$((ORPHANED_COUNT + 1))
    ORPHANED_VMS_LIST="${ORPHANED_VMS_LIST}${VM_NAME},"

    # Record in cost tracking database (if Python/sqlite3 available)
    if command -v python3 &> /dev/null && [ -f "$COST_DB" ]; then
      python3 -c "
import sqlite3
from datetime import datetime

try:
    conn = sqlite3.connect('$COST_DB')
    cursor = conn.cursor()

    # Calculate runtime and cost
    runtime_hours = $AGE_HOURS
    cost = runtime_hours * 0.029  # Spot VM hourly rate

    cursor.execute('''
        INSERT OR REPLACE INTO vm_sessions
        (instance_id, created_at, deleted_at, runtime_hours, estimated_cost, is_orphaned)
        VALUES (?, ?, ?, ?, ?, 1)
    ''', (
        '$VM_NAME',
        datetime.utcfromtimestamp($CREATED_TIMESTAMP).isoformat(),
        datetime.utcnow().isoformat(),
        runtime_hours,
        cost
    ))

    conn.commit()
    conn.close()
    print('💰 Recorded orphaned VM in cost database')
except Exception as e:
    print(f'⚠️  Failed to record in cost database: {e}')
" || log "⚠️  Failed to record orphaned VM in cost database"
    fi
  else
    log "ℹ️  VM is recent (${AGE_HOURS}h), keeping for now"
  fi
done <<< "$VMS"

# Summary and notifications
log "✅ Cleanup complete"

if [ $ORPHANED_COUNT -gt 0 ]; then
  log "📊 Summary: Deleted $ORPHANED_COUNT orphaned VM(s)"
  log "   VMs: ${ORPHANED_VMS_LIST%,}"

  # Send notification (if notification system available)
  if command -v osascript &> /dev/null; then
    osascript -e "display notification \"Deleted $ORPHANED_COUNT orphaned VM(s)\" with title \"JARVIS Cleanup\" sound name \"Purr\""
  fi

  # Send email alert (if mail command available and configured)
  if command -v mail &> /dev/null && [ -n "$JARVIS_ALERT_EMAIL" ]; then
    echo "JARVIS deleted $ORPHANED_COUNT orphaned VM(s): ${ORPHANED_VMS_LIST%,}" | \
      mail -s "JARVIS: Orphaned VMs Cleaned Up" "$JARVIS_ALERT_EMAIL"
  fi
else
  log "✅ No orphaned VMs required cleanup"
fi

# Log to monitoring endpoint (if backend is running)
if command -v curl &> /dev/null; then
  curl -s -X POST http://localhost:8010/hybrid/initialize > /dev/null 2>&1 || true
fi
