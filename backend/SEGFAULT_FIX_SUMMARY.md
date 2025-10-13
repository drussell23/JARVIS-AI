# JARVIS Segfault & Process Cleanup Fix Summary

## Problem Identified
- JARVIS was experiencing segfaults (exit code -11)
- Leaked semaphores and shared memory were accumulating
- Stuck processes were not being cleaned up properly
- Old JARVIS instances were lingering after crashes

## Solutions Implemented

### 1. Enhanced Process Cleanup Manager (`backend/process_cleanup_manager.py`)

#### New Functions Added:
- **`emergency_cleanup_all_jarvis()`**: Aggressive cleanup of all JARVIS processes
- **`check_for_segfault_recovery()`**: Detects crashes and performs recovery

#### Key Improvements:
- Reduced stuck process detection to 10 minutes (from 1 hour)
- More aggressive IPC cleanup (1 minute threshold)
- Better JARVIS process pattern detection
- Includes frontend port 3000 in cleanup
- Force-kills processes on all JARVIS ports
- Removes stale PID files and code state
- Handles leaked semaphores and shared memory

### 2. Test & Cleanup Tools

#### `backend/test_cleanup.py`
- Diagnostic tool to check system state
- Automatic cleanup with `--auto` flag
- Shows CPU, memory, and process status

#### `start_jarvis_clean.sh`
- Wrapper script for clean JARVIS startup
- Runs cleanup before starting JARVIS
- Ensures no stuck processes remain

## Usage

### Emergency Cleanup (When JARVIS is stuck)
```bash
cd backend
python -c "from process_cleanup_manager import emergency_cleanup; emergency_cleanup(force=True)"
```

### Test System State
```bash
cd backend
python test_cleanup.py
```

### Automatic Cleanup & Test
```bash
cd backend
python test_cleanup.py --auto
```

### Clean Start JARVIS
```bash
./start_jarvis_clean.sh
```

## What Gets Cleaned

1. **Processes:**
   - Python processes running `main.py`
   - Node.js frontend processes
   - WebSocket server processes
   - Any process with JARVIS patterns

2. **Ports:**
   - 3000 (Frontend)
   - 8000, 8001, 8010 (Backend)
   - 8080, 8765, 5000 (Additional services)

3. **System Resources:**
   - Leaked semaphores (IPC)
   - Shared memory segments
   - Stale PID files
   - Code state files

## Detection Patterns

The cleanup manager looks for:
- Process names/commands containing: `jarvis`, `main.py`, `voice_unlock`, `websocket_server`
- Python processes in JARVIS directory
- Node processes for frontend
- Processes using JARVIS ports
- Zombie and stuck processes

## Recovery Behavior

When a segfault is detected:
1. All JARVIS processes are terminated
2. Ports are forcefully freed
3. IPC resources are cleaned
4. State files are removed
5. System is prepared for fresh start

## Prevention Tips

1. **Use the clean start script**: `./start_jarvis_clean.sh`
2. **Monitor memory usage**: Keep below 65% system memory
3. **Regular cleanup**: Run `python backend/test_cleanup.py` periodically
4. **Code changes**: Cleanup manager auto-detects and handles code changes

## Troubleshooting

If JARVIS still crashes:
1. Run emergency cleanup: `python -c "from process_cleanup_manager import emergency_cleanup; emergency_cleanup(force=True)"`
2. Check for leaked semaphores: `ipcs -s`
3. Clean them manually: `ipcs -s | grep $USER | awk '{print $2}' | xargs -n1 ipcrm -s`
4. Restart terminal/shell
5. Use clean start script

## Success Indicators

✅ No segfaults during startup
✅ Clean port availability
✅ No leaked semaphores warning
✅ Memory usage below 65%
✅ CPU usage reasonable
✅ No zombie processes