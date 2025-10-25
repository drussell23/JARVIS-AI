#!/bin/bash
# Cleanup orphaned JARVIS Spot VMs
# Run this daily via cron or manually to cleanup forgotten VMs

set -e

PROJECT_ID="${GCP_PROJECT_ID:-jarvis-473803}"
ZONE="us-central1-a"
MAX_AGE_HOURS=6  # Delete VMs older than 6 hours

echo "🔍 Checking for orphaned JARVIS VMs..."

# Get all jarvis-auto VMs with their creation time
VMS=$(gcloud compute instances list \
  --project="$PROJECT_ID" \
  --filter="name~'jarvis-auto-.*'" \
  --format="csv[no-heading](name,creationTimestamp)" 2>/dev/null || echo "")

if [ -z "$VMS" ]; then
  echo "✅ No orphaned VMs found"
  exit 0
fi

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

  echo "Found VM: $VM_NAME (age: ${AGE_HOURS}h)"

  if [ $AGE_HOURS -ge $MAX_AGE_HOURS ]; then
    echo "⚠️  VM is older than ${MAX_AGE_HOURS}h - deleting..."

    gcloud compute instances delete "$VM_NAME" \
      --zone="$ZONE" \
      --project="$PROJECT_ID" \
      --quiet

    echo "✅ Deleted orphaned VM: $VM_NAME"
  else
    echo "ℹ️  VM is recent (${AGE_HOURS}h), keeping for now"
  fi
done <<< "$VMS"

echo "✅ Cleanup complete"
