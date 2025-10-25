#!/bin/bash
# JARVIS Launcher with Robust Cleanup
# Handles terminal close gracefully

set -e

JARVIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$JARVIS_DIR"

# PID file for tracking
PID_FILE="/tmp/jarvis.pid"
LOG_FILE="/tmp/jarvis_launcher.log"

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up JARVIS..." | tee -a "$LOG_FILE"

    if [ -f "$PID_FILE" ]; then
        JARVIS_PID=$(cat "$PID_FILE")
        if ps -p "$JARVIS_PID" > /dev/null 2>&1; then
            echo "Stopping JARVIS (PID: $JARVIS_PID)..." | tee -a "$LOG_FILE"
            kill -TERM "$JARVIS_PID" 2>/dev/null || true

            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! ps -p "$JARVIS_PID" > /dev/null 2>&1; then
                    echo "✅ JARVIS stopped gracefully" | tee -a "$LOG_FILE"
                    break
                fi
                sleep 1
            done

            # Force kill if still running
            if ps -p "$JARVIS_PID" > /dev/null 2>&1; then
                echo "⚠️  Force killing JARVIS..." | tee -a "$LOG_FILE"
                kill -9 "$JARVIS_PID" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi

    # Clean up any orphaned processes
    pkill -f "start_system.py" 2>/dev/null || true

    echo "✅ Cleanup complete" | tee -a "$LOG_FILE"
}

# Trap signals
trap cleanup EXIT INT TERM HUP

echo "🚀 Starting JARVIS..." | tee "$LOG_FILE"
echo "📝 Logs: $LOG_FILE"
echo ""

# Start JARVIS
python start_system.py &
JARVIS_PID=$!
echo $JARVIS_PID > "$PID_FILE"

echo "JARVIS PID: $JARVIS_PID" | tee -a "$LOG_FILE"
echo ""
echo "Press Ctrl+C to stop, or close this terminal"
echo ""

# Wait for JARVIS
wait $JARVIS_PID 2>/dev/null || true

# If we reach here, JARVIS exited naturally
rm -f "$PID_FILE"
echo "JARVIS exited" | tee -a "$LOG_FILE"
