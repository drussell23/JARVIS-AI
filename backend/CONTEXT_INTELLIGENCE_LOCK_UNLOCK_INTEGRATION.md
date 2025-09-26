# Context Intelligence Lock/Unlock Integration

## Overview

The Context Intelligence System now includes both screen locking and unlocking capabilities, integrating seamlessly with the existing Voice Unlock infrastructure when available.

## Architecture

### Lock/Unlock Manager (`unlock_manager.py`)
The `UnlockManager` class has been enhanced to handle both locking and unlocking:

```python
# Lock screen
success, message = await unlock_manager.lock_screen("User command")

# Unlock screen 
success, message = await unlock_manager.unlock_screen("Execute command", context)
```

### Integration Hierarchy

1. **Voice Unlock Connector** (if available)
   - Uses WebSocket connection to Voice Unlock daemon
   - Provides most reliable lock/unlock functionality
   - Falls back to other methods if not connected

2. **AppleScript Method**
   - Primary fallback for locking: `Cmd+Ctrl+Q`
   - Primary method for unlocking (with stored password)
   - Works reliably on most macOS systems

3. **PMSet Method** 
   - Final fallback for locking: `pmset displaysleepnow`
   - Puts display to sleep, effectively locking

## How It Works

### When User Says "Lock My Screen"

1. **Command Detection**
   - `unified_command_executor.py` detects lock phrases
   - Routes to `_handle_lock_screen()` method

2. **Lock Execution**
   ```python
   # Tries methods in order:
   1. Voice Unlock connector (if connected)
   2. AppleScript (Cmd+Ctrl+Q)
   3. PMSet (display sleep)
   ```

3. **Feedback**
   - Success: "Screen locked successfully"
   - Failure: Detailed error message

### When Screen is Locked and User Gives Command

1. **Context Detection**
   - `screen_state.py` detects locked screen
   - Command queued in `command_queue.py`

2. **Policy Evaluation**
   - `policy_engine.py` decides if auto-unlock allowed
   - Low sensitivity commands unlock automatically

3. **Unlock Execution**
   - `unlock_manager.py` attempts unlock
   - Uses stored password from Keychain
   - Falls back through multiple methods

## Configuration

### Store Password (One-Time Setup)
```python
from context_intelligence.core.unlock_manager import get_unlock_manager
manager = get_unlock_manager()
manager.store_password("your_password")
```

### Enable Voice Unlock (Optional)
```bash
# Start Voice Unlock WebSocket server
cd voice_unlock/objc/server
python websocket_server.py
```

## Supported Commands

### Lock Commands
- "lock my screen"
- "lock screen"
- "lock the screen"
- "lock my mac"
- "lock mac"
- "lock the mac"
- "lock computer"
- "lock the computer"
- "lock my computer"

### Unlock (Automatic)
Happens automatically when:
- Screen is locked
- User gives command requiring screen
- Policy allows auto-unlock

## Testing

### Test Lock Functionality
```bash
python test_lock_screen_integration.py
```

### Test Full Context Flow
```bash
# Lock screen manually or via command
# Then say: "JARVIS, open Safari and search for dogs"
```

## Integration Status

✅ **Working Without Voice Unlock Daemon**
- AppleScript lock/unlock
- PMSet display sleep
- Keychain password storage
- All Context Intelligence features

✅ **Enhanced With Voice Unlock Daemon**
- More reliable lock/unlock
- Additional biometric options
- Voice-based authentication
- Proximity detection (if configured)

## Error Handling

The system gracefully handles:
- Missing Voice Unlock daemon
- No stored password
- Permission issues
- Timeout scenarios
- Failed unlock attempts

Each fallback method is tried in sequence, ensuring maximum reliability.

## Why Files Appear Gray in IDE

The Voice Unlock integration files may appear gray because:
1. They're imported dynamically only when needed
2. The import is wrapped in try/except blocks
3. IDE static analysis doesn't trace dynamic imports

This is normal and doesn't affect functionality. The integration works whether Voice Unlock is available or not.