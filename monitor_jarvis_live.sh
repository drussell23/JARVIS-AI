#!/bin/bash
# Monitor JARVIS for display clicking in real-time

echo "🔍 Monitoring JARVIS for 'living room tv' commands..."
echo "=================================================="
echo ""
echo "Watching for:"
echo "  - Control Center clicks at (1236, 12)"
echo "  - Screen Mirroring clicks at (1396, 177)"
echo "  - Living Room TV clicks at (1223, 115)"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "=================================================="
echo ""

# Find the latest JARVIS log file or monitor stdout
tail -f /tmp/jarvis_output.log 2>/dev/null | grep -E --line-buffered "living room|Living Room|control.center|screen.mirroring|ADAPTIVE|CLICKING|coordinates|1236|1396|1223|DISPLAY MONITOR|Click" &

# Also monitor any Python processes running main.py
while true; do
    PID=$(ps aux | grep "python.*main.py" | grep -v grep | awk '{print $2}' | head -1)
    if [ ! -z "$PID" ]; then
        echo "📡 Found JARVIS process: PID=$PID"
        # Try to capture output from the process
        sudo dtruss -p $PID 2>&1 | grep -E "living room|control center" 2>/dev/null &
        break
    fi
    sleep 2
done

wait