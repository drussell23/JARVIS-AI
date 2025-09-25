# Enhanced Process Cleanup Manager

## Overview
The Process Cleanup Manager has been significantly enhanced to ensure that JARVIS always runs with the latest code. It now detects code changes automatically and terminates old instances before starting new ones.

## Key Features

### 1. **Code Change Detection**
- Calculates SHA-256 hash of critical JARVIS files
- Compares with saved state from previous runs
- Detects any modifications to core files
- Monitored files include:
  - `main.py`
  - `api/jarvis_voice_api.py`
  - `api/unified_command_processor.py`
  - `api/voice_unlock_integration.py`
  - `voice/jarvis_voice.py`
  - `voice/macos_voice.py`
  - `engines/voice_engine.py`

### 2. **Automatic Old Instance Cleanup**
- When code changes are detected, all old JARVIS instances are terminated
- Graceful termination attempted first (5 second timeout)
- Force kill if graceful termination fails
- Cleans up related processes (voice_unlock, websocket_server, etc.)

### 3. **Single Instance Enforcement**
- Ensures only one JARVIS instance runs per port
- Checks if another instance is using the target port
- If code has changed, terminates the old instance
- Prevents port conflicts and confusion

### 4. **Integration with Main Startup**
The process cleanup is now integrated into `main.py`:

```python
# At startup, before loading components
from process_cleanup_manager import ensure_fresh_jarvis_instance, cleanup_system_for_jarvis

# Check and clean up old instances
if not ensure_fresh_jarvis_instance():
    raise RuntimeError("Port conflict - another JARVIS instance is running")

# Run full system cleanup
cleanup_report = await cleanup_system_for_jarvis(dry_run=False)
```

### 5. **State Persistence**
- Code state saved to `~/.jarvis/code_state.json`
- Includes:
  - Code hash
  - Last update timestamp
  - Process ID
- State saved on shutdown for next startup comparison

## How It Works

### Startup Flow
1. JARVIS starts up
2. Process cleanup manager checks for code changes
3. If changes detected:
   - Find all old JARVIS processes
   - Terminate them (gracefully, then forcefully)
   - Clean up orphaned ports
   - Save new code state
4. Ensure single instance on target port
5. Continue with normal startup

### Code Change Detection Algorithm
```python
def _calculate_code_hash(self) -> str:
    """Calculate hash of critical JARVIS files"""
    hasher = hashlib.sha256()
    
    for file_path in self.config['critical_files']:
        full_path = self.backend_path / file_path
        if full_path.exists():
            # Hash file contents
            with open(full_path, 'rb') as f:
                hasher.update(f.read())
            # Include modification time
            hasher.update(str(full_path.stat().st_mtime).encode())
    
    return hasher.hexdigest()
```

## Benefits

### No More Old Instance Issues
- **Problem**: Previously, old JARVIS instances would keep running after code updates
- **Solution**: Automatic detection and cleanup ensures only fresh code runs
- **Result**: Audio fixes, new features, and bug fixes take effect immediately

### Cleaner Development Workflow
1. Make code changes
2. Start JARVIS
3. Old instances automatically cleaned up
4. Fresh instance starts with new code
5. No manual process killing required

### Memory and Resource Management
- Prevents resource leaks from multiple instances
- Cleans up zombie processes
- Monitors system health
- Provides recommendations for optimization

## Testing

Run the test script to verify functionality:
```bash
python test_process_cleanup.py
```

This will:
- Show current JARVIS processes
- Check for code changes
- Test cleanup functionality
- Verify single instance enforcement
- Demonstrate code change simulation

## Configuration

The cleanup manager can be configured in `process_cleanup_manager.py`:

```python
'jarvis_patterns': [
    'jarvis', 'main.py', 'jarvis_backend', 'jarvis_voice',
    'voice_unlock', 'websocket_server', 'jarvis-ai-agent',
    'unified_command_processor', 'resource_manager'
],
'jarvis_port_patterns': [8000, 8001, 8010, 8080, 8765, 5000],
```

Add patterns or ports as needed for your setup.

## Troubleshooting

### Permission Errors
- The manager handles permission errors gracefully
- Falls back to `lsof` command for port checking
- Continues operation even if some checks fail

### Code State Not Updating
- Check `~/.jarvis/code_state.json`
- Delete the file to force fresh state
- Ensure write permissions to `~/.jarvis/`

### Old Instances Not Cleaned
- Check process patterns match your setup
- Verify file paths in `critical_files`
- Run with debug logging enabled

## Future Enhancements

1. **Automatic Restart**: Restart JARVIS automatically when code changes detected
2. **Hot Reload**: Reload specific modules without full restart
3. **Version Tracking**: Track code versions and rollback capability
4. **Multi-Instance Support**: Support multiple JARVIS instances on different ports
5. **Remote Cleanup**: Clean up JARVIS instances on remote machines

## Conclusion

The enhanced Process Cleanup Manager ensures that JARVIS always runs with the latest code, eliminating the frustration of old instances running outdated code. This is especially critical for fixes like the audio system using Daniel's British voice - now these changes take effect immediately upon restart.