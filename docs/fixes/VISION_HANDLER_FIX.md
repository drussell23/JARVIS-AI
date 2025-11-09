# Vision Handler Fix - Multiple Process Detection Enhancement

## Issue
JARVIS vision commands ("can you see my screen?") were hanging indefinitely due to:
1. Anthropic API key not being retrieved from GCP Secret Manager
2. Multiple backend processes running simultaneously when using `python start_system.py --restart`

## Root Cause
- **API Key Issue**: Vision handler was using `os.getenv("ANTHROPIC_API_KEY")` instead of SecretManager
- **Multiple Processes**: Process detection in `start_system.py` wasn't catching all JARVIS backend processes, particularly those bound to ports 8010, 8000, and 3000

## Solution

### 1. SecretManager Integration (Files: `vision_command_handler.py`, `jarvis_voice_api.py`)
- Added `get_anthropic_key()` from `core.secret_manager` as primary API key source
- Implemented fallback chain: SecretManager → App State → Environment
- Added comprehensive error handling with user-friendly messages
- Added 30-second timeout protection using `asyncio.wait_for()`

### 2. Enhanced Process Detection (File: `start_system.py`)
Added **Strategy 3: Port-based detection** to complement existing strategies:

#### Three Detection Strategies:
1. **Strategy 1: psutil enumeration** - Scans all processes for JARVIS-related names
2. **Strategy 2: ps command verification** - Uses `ps aux | grep` for additional verification
3. **Strategy 3: Port-based detection** *(NEW)* - Uses `lsof` to find processes on ports 8010, 8000, 3000

#### Strategy 3 Implementation:
```python
# Strategy 3: Check for processes using JARVIS ports (8010, 8000, 3000)
for port in [8010, 8000, 3000]:
    lsof_result = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True,
        text=True,
        timeout=2
    )
    if lsof_result.stdout.strip():
        pids = lsof_result.stdout.strip().split("\n")
        # Process each PID found on this port
        # Add to jarvis_processes list if not already found
```

## Verification

### API Key Retrieval:
```
✅ Retrieved 'anthropic-api-key' from GCP Secret Manager
```

### Process Detection:
When running `python start_system.py --restart`, all three strategies execute:
- Strategy 1 finds processes by name
- Strategy 2 verifies with ps command
- Strategy 3 catches any processes bound to JARVIS ports

## Usage

### Normal Restart:
```bash
python start_system.py --restart
```

This now properly detects and terminates:
- Main JARVIS processes
- Backend API servers (port 8010)
- Frontend servers (port 3000)
- Any orphaned processes still bound to JARVIS ports

### Manual Cleanup (if needed):
```bash
# Find processes on JARVIS ports
lsof -ti :8010
lsof -ti :8000
lsof -ti :3000

# Kill specific PID
kill -9 <PID>
```

## Prevention

The enhanced 3-strategy detection ensures:
- No multiple backend processes when using `--restart`
- All JARVIS-related processes are properly terminated before starting fresh
- Port conflicts are avoided
- Clean startup every time

## Files Modified
1. `/backend/api/vision_command_handler.py` - SecretManager integration
2. `/backend/api/jarvis_voice_api.py` - Timeout protection and error handling
3. `/start_system.py` - Strategy 3 port-based detection
