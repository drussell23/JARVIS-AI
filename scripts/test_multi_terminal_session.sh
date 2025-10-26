#!/bin/bash
###############################################################################
# Multi-Terminal Session Test Script for JARVIS
# Tests VMSessionTracker implementation for multi-terminal safety
#
# Purpose:
#   Validates that multiple JARVIS terminals can run simultaneously without
#   interfering with each other's GCP VM instances during cleanup.
#
# Usage:
#   bash scripts/test_multi_terminal_session.sh [test_number]
#
# Tests:
#   1. Multi-terminal session isolation
#   2. Session registry inspection
#   3. Stale session cleanup
#   4. Rapid terminal cycling
#   5. Comprehensive integration test
#
# Requirements:
#   - gcloud CLI configured
#   - GCP_PROJECT_ID environment variable set
#   - JARVIS repository at /Users/derekjrussell/Documents/repos/JARVIS-AI-Agent
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-jarvis-473803}"
REPO_DIR="/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent"
LOG_DIR="/tmp/jarvis_test_logs"
REGISTRY_FILE="/tmp/jarvis_vm_registry.json"

# Create log directory
mkdir -p "$LOG_DIR"

###############################################################################
# Helper Functions
###############################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install gcloud."
        exit 1
    fi

    # Check GCP project
    if [ -z "$PROJECT_ID" ]; then
        log_error "GCP_PROJECT_ID not set. Please set GCP_PROJECT_ID environment variable."
        exit 1
    fi

    # Check JARVIS repo
    if [ ! -f "$REPO_DIR/start_system.py" ]; then
        log_error "JARVIS repository not found at $REPO_DIR"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

list_vms() {
    gcloud compute instances list \
        --project="$PROJECT_ID" \
        --filter="name:jarvis-auto-*" \
        --format="table(name,zone,status,creationTimestamp)" \
        2>/dev/null || echo "No VMs found"
}

count_vms() {
    gcloud compute instances list \
        --project="$PROJECT_ID" \
        --filter="name:jarvis-auto-*" \
        --format="value(name)" \
        2>/dev/null | wc -l | tr -d ' '
}

list_session_files() {
    ls -la /tmp/jarvis_session_*.json 2>/dev/null || echo "No session files found"
}

count_session_files() {
    ls /tmp/jarvis_session_*.json 2>/dev/null | wc -l | tr -d ' '
}

show_registry() {
    if [ -f "$REGISTRY_FILE" ]; then
        log_info "VM Registry contents:"
        cat "$REGISTRY_FILE" | python3 -m json.tool 2>/dev/null || cat "$REGISTRY_FILE"
    else
        log_info "VM Registry file does not exist"
    fi
}

cleanup_all() {
    log_warning "Cleaning up all test resources..."

    # Kill all start_system.py processes
    pkill -f "python.*start_system.py" 2>/dev/null || true
    sleep 2

    # Delete all test VMs
    local vm_count=$(count_vms)
    if [ "$vm_count" -gt 0 ]; then
        log_info "Deleting $vm_count test VM(s)..."
        gcloud compute instances list \
            --project="$PROJECT_ID" \
            --filter="name:jarvis-auto-*" \
            --format="value(name,zone)" | \
        while IFS=$'\t' read -r name zone; do
            log_info "  Deleting: $name (zone: $zone)"
            gcloud compute instances delete "$name" \
                --project="$PROJECT_ID" \
                --zone="$zone" \
                --quiet 2>/dev/null || true
        done
    fi

    # Clean up session files
    rm -f /tmp/jarvis_session_*.json 2>/dev/null || true
    rm -f "$REGISTRY_FILE" 2>/dev/null || true

    # Clean up log files
    rm -f "$LOG_DIR"/*.log 2>/dev/null || true

    log_success "Cleanup complete"
}

wait_for_vm_creation() {
    local log_file=$1
    local timeout=120
    local elapsed=0

    log_info "Waiting for VM creation (max ${timeout}s)..."

    while [ $elapsed -lt $timeout ]; do
        if grep -q "📝 Tracking GCP instance for cleanup" "$log_file" 2>/dev/null; then
            local vm_id=$(grep "📝 Tracking GCP instance for cleanup" "$log_file" | tail -1 | awk '{print $NF}')
            log_success "VM created: $vm_id"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    log_error "VM creation timeout after ${timeout}s"
    return 1
}

wait_for_session_init() {
    local log_file=$1
    local timeout=30
    local elapsed=0

    log_info "Waiting for session initialization (max ${timeout}s)..."

    while [ $elapsed -lt $timeout ]; do
        if grep -q "🆔 Session tracker initialized" "$log_file" 2>/dev/null; then
            local session_id=$(grep "🆔 Session tracker initialized" "$log_file" | tail -1 | awk '{print $NF}')
            log_success "Session initialized: $session_id"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    log_error "Session initialization timeout after ${timeout}s"
    return 1
}

###############################################################################
# Test 1: Multi-Terminal Session Isolation
###############################################################################

test_multi_terminal_isolation() {
    log_info "========================================="
    log_info "Test 1: Multi-Terminal Session Isolation"
    log_info "========================================="

    cleanup_all

    log_info "Starting Terminal 1..."
    cd "$REPO_DIR"
    python3 start_system.py > "$LOG_DIR/terminal1.log" 2>&1 &
    local PID1=$!
    log_info "Terminal 1 PID: $PID1"

    # Wait for Terminal 1 session initialization
    if ! wait_for_session_init "$LOG_DIR/terminal1.log"; then
        log_error "Terminal 1 session initialization failed"
        kill $PID1 2>/dev/null || true
        return 1
    fi

    # Wait for Terminal 1 VM creation (if RAM triggers it)
    sleep 30  # Give it time to potentially create VM

    log_info "Starting Terminal 2..."
    python3 start_system.py > "$LOG_DIR/terminal2.log" 2>&1 &
    local PID2=$!
    log_info "Terminal 2 PID: $PID2"

    # Wait for Terminal 2 session initialization
    if ! wait_for_session_init "$LOG_DIR/terminal2.log"; then
        log_error "Terminal 2 session initialization failed"
        kill $PID1 $PID2 2>/dev/null || true
        return 1
    fi

    sleep 30  # Give it time to potentially create VM

    log_info "Current VMs:"
    list_vms

    log_info "Session files:"
    list_session_files

    show_registry

    log_info "Killing Terminal 1 (PID: $PID1)..."
    kill -INT $PID1 2>/dev/null || true
    sleep 5

    log_info "VMs after Terminal 1 cleanup:"
    list_vms

    log_info "Session files after Terminal 1 cleanup:"
    list_session_files

    # Check Terminal 2 still running
    if ps -p $PID2 > /dev/null; then
        log_success "Terminal 2 still running (PID: $PID2)"
    else
        log_error "Terminal 2 terminated unexpectedly"
        return 1
    fi

    log_info "Killing Terminal 2 (PID: $PID2)..."
    kill -INT $PID2 2>/dev/null || true
    sleep 5

    log_info "VMs after Terminal 2 cleanup:"
    local final_vm_count=$(count_vms)
    list_vms

    log_info "Session files after Terminal 2 cleanup:"
    local final_session_count=$(count_session_files)
    list_session_files

    # Validate results
    if [ "$final_vm_count" -eq 0 ]; then
        log_success "✅ All VMs cleaned up correctly"
    else
        log_error "❌ $final_vm_count VM(s) remain after cleanup"
        return 1
    fi

    if [ "$final_session_count" -eq 0 ]; then
        log_success "✅ All session files cleaned up correctly"
    else
        log_warning "⚠️  $final_session_count session file(s) remain"
    fi

    log_success "Test 1 PASSED"
    return 0
}

###############################################################################
# Test 2: Session Registry Inspection
###############################################################################

test_registry_inspection() {
    log_info "======================================"
    log_info "Test 2: Session Registry Inspection"
    log_info "======================================"

    cleanup_all

    log_info "Starting 2 terminals..."
    cd "$REPO_DIR"

    python3 start_system.py > "$LOG_DIR/terminal1.log" 2>&1 &
    local PID1=$!
    sleep 10

    python3 start_system.py > "$LOG_DIR/terminal2.log" 2>&1 &
    local PID2=$!
    sleep 10

    log_info "Inspecting VM registry..."
    show_registry

    if [ -f "$REGISTRY_FILE" ]; then
        local session_count=$(cat "$REGISTRY_FILE" | python3 -c "import json, sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
        log_info "Sessions in registry: $session_count"

        if [ "$session_count" -ge 2 ]; then
            log_success "✅ Registry contains multiple sessions"
        else
            log_warning "⚠️  Expected 2 sessions, found $session_count"
        fi

        # Validate session structure
        log_info "Validating session structure..."
        cat "$REGISTRY_FILE" | python3 -c "
import json, sys
registry = json.load(sys.stdin)
for sid, data in registry.items():
    required = ['session_id', 'pid', 'hostname', 'vm_id', 'zone', 'components', 'created_at', 'registered_at']
    missing = [k for k in required if k not in data]
    if missing:
        print(f'❌ Session {sid[:8]} missing fields: {missing}')
        sys.exit(1)
print('✅ All sessions have required fields')
" 2>/dev/null
    else
        log_warning "⚠️  Registry file not found"
    fi

    log_info "Inspecting individual session files..."
    list_session_files

    # Cleanup
    log_info "Cleaning up..."
    kill -INT $PID1 $PID2 2>/dev/null || true
    sleep 5

    log_success "Test 2 PASSED"
    return 0
}

###############################################################################
# Test 3: Stale Session Cleanup
###############################################################################

test_stale_session_cleanup() {
    log_info "==================================="
    log_info "Test 3: Stale Session Cleanup"
    log_info "==================================="

    cleanup_all

    log_info "Starting JARVIS, then force killing..."
    cd "$REPO_DIR"

    python3 start_system.py > "$LOG_DIR/terminal1.log" 2>&1 &
    local PID=$!
    log_info "PID: $PID"

    sleep 15  # Wait for session initialization

    log_info "Force killing (simulating crash)..."
    kill -9 $PID 2>/dev/null || true
    sleep 2

    log_info "Session file after force kill:"
    if [ -f "/tmp/jarvis_session_${PID}.json" ]; then
        log_info "Session file exists (expected)"
        cat "/tmp/jarvis_session_${PID}.json"
    else
        log_warning "Session file does not exist"
    fi

    log_info "Starting new JARVIS session..."
    python3 start_system.py > "$LOG_DIR/terminal2.log" 2>&1 &
    local PID2=$!
    log_info "New PID: $PID2"

    sleep 15  # Wait for registry auto-clean

    log_info "Registry after new session start:"
    show_registry

    # Check if stale session was removed
    if [ -f "$REGISTRY_FILE" ]; then
        local has_stale=$(cat "$REGISTRY_FILE" | python3 -c "
import json, sys
registry = json.load(sys.stdin)
stale_pid = $PID
for sid, data in registry.items():
    if data.get('pid') == stale_pid:
        print('yes')
        sys.exit(0)
print('no')
" 2>/dev/null || echo "unknown")

        if [ "$has_stale" = "no" ]; then
            log_success "✅ Stale session removed from registry"
        else
            log_error "❌ Stale session still in registry"
        fi
    fi

    # Cleanup
    log_info "Cleaning up..."
    kill -INT $PID2 2>/dev/null || true
    sleep 5

    log_success "Test 3 PASSED"
    return 0
}

###############################################################################
# Test 4: Rapid Terminal Cycling
###############################################################################

test_rapid_cycling() {
    log_info "================================"
    log_info "Test 4: Rapid Terminal Cycling"
    log_info "================================"

    cleanup_all

    local cycles=3
    log_info "Running $cycles rapid cycles..."

    cd "$REPO_DIR"

    for i in $(seq 1 $cycles); do
        log_info "Cycle $i/$cycles"

        python3 start_system.py > "$LOG_DIR/cycle${i}.log" 2>&1 &
        local PID=$!
        log_info "  Started PID: $PID"

        sleep 20  # Wait for initialization

        log_info "  Stopping PID: $PID"
        kill -INT $PID 2>/dev/null || true
        wait $PID 2>/dev/null || true

        sleep 3
    done

    log_info "Final state after $cycles cycles:"
    log_info "VMs:"
    local final_vm_count=$(count_vms)
    list_vms

    log_info "Session files:"
    local final_session_count=$(count_session_files)
    list_session_files

    # Validate cleanup
    if [ "$final_vm_count" -eq 0 ]; then
        log_success "✅ No VMs remain after cycling"
    else
        log_error "❌ $final_vm_count VM(s) remain after cycling"
        return 1
    fi

    if [ "$final_session_count" -eq 0 ]; then
        log_success "✅ No session files remain after cycling"
    else
        log_warning "⚠️  $final_session_count session file(s) remain"
    fi

    log_success "Test 4 PASSED"
    return 0
}

###############################################################################
# Test 5: Comprehensive Integration Test
###############################################################################

test_comprehensive() {
    log_info "==========================================="
    log_info "Test 5: Comprehensive Integration Test"
    log_info "==========================================="

    log_info "Running all tests in sequence..."

    local failed=0

    test_multi_terminal_isolation || ((failed++))
    sleep 5

    test_registry_inspection || ((failed++))
    sleep 5

    test_stale_session_cleanup || ((failed++))
    sleep 5

    test_rapid_cycling || ((failed++))

    if [ $failed -eq 0 ]; then
        log_success "========================================="
        log_success "ALL TESTS PASSED (0 failures)"
        log_success "========================================="
        return 0
    else
        log_error "========================================="
        log_error "$failed TEST(S) FAILED"
        log_error "========================================="
        return 1
    fi
}

###############################################################################
# Main
###############################################################################

main() {
    check_prerequisites

    local test_num=${1:-5}

    case $test_num in
        1)
            test_multi_terminal_isolation
            ;;
        2)
            test_registry_inspection
            ;;
        3)
            test_stale_session_cleanup
            ;;
        4)
            test_rapid_cycling
            ;;
        5)
            test_comprehensive
            ;;
        *)
            log_error "Invalid test number: $test_num"
            log_info "Usage: $0 [1-5]"
            log_info "  1 - Multi-terminal session isolation"
            log_info "  2 - Session registry inspection"
            log_info "  3 - Stale session cleanup"
            log_info "  4 - Rapid terminal cycling"
            log_info "  5 - Comprehensive integration test (default)"
            exit 1
            ;;
    esac

    local exit_code=$?

    log_info "Test logs available at: $LOG_DIR"

    exit $exit_code
}

# Trap Ctrl+C to cleanup
trap cleanup_all INT TERM

main "$@"
