# Context Intelligence Integration Guide

## Overview

The Context Intelligence System is now fully integrated with JARVIS to handle the scenario where commands are issued while the screen is locked. This guide explains how the integration works and how to use it.

## Integration Architecture

```
User Voice Command
        ↓
jarvis_voice_api.py
        ↓
enhanced_context_wrapper.py (NEW - replaces simple_context_handler_enhanced.py)
        ↓
Context Intelligence System
        ├── Screen State Detection
        ├── Command Queue (if locked)
        ├── Policy Engine
        ├── Unlock Manager
        └── Command Execution → unified_command_processor.py
```

## Key Integration Points

### 1. **Drop-in Replacement**
The new system provides `enhanced_context_wrapper.py` which is a drop-in replacement for `simple_context_handler_enhanced.py`:

```python
# Old import (in jarvis_voice_api.py)
from .simple_context_handler_enhanced import wrap_with_enhanced_context

# New import (automatic with apply_context_intelligence_integration.py)
from context_intelligence.integrations.enhanced_context_wrapper import wrap_with_enhanced_context
```

### 2. **WebSocket Integration**
Real-time updates are sent through WebSocket connections:
- Progress updates during execution
- Voice feedback messages
- State change notifications

### 3. **Command Execution**
Commands are executed through the existing `unified_command_processor.py` via the new `unified_command_executor.py`.

## Example Flow: "Open Safari and search for dogs" (Screen Locked)

### 1. **Detection Phase**
```python
# User says: "JARVIS, open Safari and search for dogs"

# System detects screen is locked
screen_state = await screen_detector.get_screen_state()
# Returns: ScreenState.LOCKED with 95% confidence
```

### 2. **Queue Phase**
```python
# Command is queued with context
queued_command = await command_queue.enqueue(
    command_text="open Safari and search for dogs",
    intent={
        "type": "open_app",
        "action": "open", 
        "target": "Safari",
        "parameters": {"search": "dogs"}
    },
    requires_screen=True
)
```

### 3. **Feedback Phase**
```python
# User receives immediate feedback
"I see your screen is locked. I'll unlock it now to open Safari and search for dogs."
```

### 4. **Policy Evaluation**
```python
# Policy engine decides if auto-unlock is allowed
decision = await policy_engine.evaluate_unlock_request(
    command="open Safari and search for dogs",
    sensitivity=CommandSensitivity.LOW  # Browser search is low sensitivity
)
# Returns: PolicyDecision.AUTO_UNLOCK
```

### 5. **Unlock Phase**
```python
# Unlock manager attempts to unlock
success = await unlock_manager.unlock_screen(
    reason="User command: open Safari and search for dogs"
)
# Tries multiple methods: AppleScript → Accessibility → Voice Unlock
```

### 6. **Execution Phase**
```python
# After successful unlock, command is executed
result = await unified_command_executor.execute_command(
    "open Safari and search for dogs"
)
# Opens Safari and performs search
```

### 7. **Completion Feedback**
```python
# User receives completion message
"I've successfully unlocked your screen, opened Safari, and searched for dogs."
```

## Quick Start

### 1. Apply Integration
```bash
cd backend
python apply_context_intelligence_integration.py
```

### 2. Test Integration
```bash
# Simple test
python test_integration.py

# Full scenario test
python test_full_context_flow.py

# Example usage
python example_context_usage.py
```

### 3. Configure Policies
Edit `~/.jarvis/unlock_policies.json` to customize auto-unlock rules:

```json
{
  "rules": [
    {
      "rule_id": "browser_auto_unlock",
      "name": "Auto-unlock for browser commands",
      "conditions": {
        "sensitivity": ["public", "low"],
        "is_quiet_hours": false
      },
      "decision": "auto_unlock"
    }
  ]
}
```

### 4. Store Password (Required for Auto-Unlock)
```python
from context_intelligence.core.unlock_manager import get_unlock_manager
manager = get_unlock_manager()
manager.store_password("your_password")  # Stored securely in macOS Keychain
```

## Configuration

### Context Manager Settings
- `max_execution_time`: 300 seconds (5 minutes)
- `state_timeout`: 60 seconds per state
- `enable_auto_unlock`: true
- `verbose_feedback`: true

### Screen Detection Settings
- Cache TTL: 500ms
- Detection methods: Quartz → IORegistry → PMSet → Process
- Confidence threshold: 0.8

### Queue Settings
- Maximum capacity: 50 commands
- Persistence: `~/.jarvis/command_queue.json`
- Auto-cleanup: 24 hours

## Monitoring & Debugging

### View System Status
```python
from context_intelligence.integrations.jarvis_integration import handle_system_status
status = await handle_system_status()
```

### View Queue Status
```python
from context_intelligence.integrations.jarvis_integration import handle_queue_status
queue = await handle_queue_status()
```

### View Audit Trail
```python
from context_intelligence.core.policy_engine import get_policy_engine
engine = get_policy_engine()
audit = engine.get_audit_trail(hours=24)
```

## Troubleshooting

### Issue: Commands not queuing when locked
- Check screen detection: `await screen_detector.get_detection_stats()`
- Verify detection accuracy and method reliability

### Issue: Auto-unlock not working
- Verify password is stored: `unlock_manager.has_stored_password()`
- Check available methods: `unlock_manager.get_statistics()`
- Review policy decisions in audit log

### Issue: Commands failing after unlock
- Check unified processor integration
- Verify WebSocket connection is maintained
- Review execution logs

## Benefits

1. **Seamless Experience**: Commands work regardless of screen state
2. **Security**: Policy-based unlock decisions with audit trail  
3. **Transparency**: Clear feedback at every step
4. **Reliability**: Multiple detection/unlock methods with fallbacks
5. **Performance**: <500ms state detection, priority queue ordering

## Future Enhancements

- Machine learning for command patterns
- Biometric unlock integration
- Multi-device synchronization
- Advanced scheduling policies

---

The Context Intelligence System transforms JARVIS into a truly intelligent assistant that understands and adapts to system state, providing a seamless experience even when the screen is locked.