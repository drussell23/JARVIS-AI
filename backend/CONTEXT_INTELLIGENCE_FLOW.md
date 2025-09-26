# Context Intelligence System - Complete Flow

## How All Core Modules Work Together

When a user says **"JARVIS, open Safari and search for dogs"** while the screen is locked:

```
USER VOICE COMMAND
       │
       ▼
jarvis_voice_api.py (WebSocket handler - line 1169)
       │
       ├─► Imports: context_intelligence.integrations.enhanced_context_wrapper
       │
       ▼
enhanced_context_wrapper.py (EnhancedContextIntelligenceHandler)
       │
       ├─► Creates/Uses ALL Core Modules:
       │
       ├─► 1. context_manager.py (get_context_manager)
       │      • Central orchestration hub
       │      • Manages entire flow
       │      • State machine for execution
       │
       ├─► 2. screen_state.py (via context_manager.screen_detector)
       │      • Detects screen is LOCKED
       │      • Uses Quartz API (95% confidence)
       │      • Returns: ScreenState.LOCKED
       │
       ├─► 3. command_queue.py (via context_manager.command_queue)
       │      • Enqueues command with metadata
       │      • Priority: NORMAL
       │      • Persistent storage in ~/.jarvis/command_queue.json
       │
       ├─► 4. policy_engine.py (via context_manager.policy_engine)
       │      • Evaluates: "open Safari" = LOW sensitivity
       │      • Decision: AUTO_UNLOCK allowed
       │      • Logs audit trail
       │
       ├─► 5. feedback_manager.py (sends user feedback)
       │      • "I see your screen is locked. I'll unlock it now..."
       │      • Sent via WebSocket to frontend
       │
       ├─► 6. unlock_manager.py (via context_manager.unlock_manager)
       │      • Retrieves password from Keychain
       │      • Executes AppleScript to type password
       │      • Verifies unlock success
       │
       ├─► 7. context_manager.py (state transitions)
       │      • IDLE → CHECKING_PREREQUISITES
       │      • → AWAITING_UNLOCK → UNLOCKING
       │      • → EXECUTING → COMPLETED
       │
       ├─► 8. unified_command_executor.py (actual execution)
       │      • Calls unified_command_processor.py
       │      • Opens Safari
       │      • Performs search for "dogs"
       │
       └─► 9. feedback_manager.py (completion)
              • "I've successfully opened Safari and searched for dogs"
```

## File-by-File Execution Flow

### 1. **Entry Point**: `jarvis_voice_api.py`
```python
# Line 1169: This is where we switch to new system
from context_intelligence.integrations.enhanced_context_wrapper import (
    wrap_with_enhanced_context,
)
# Line 1174: Creates our handler
context_handler = wrap_with_enhanced_context(processor)
# Line 1180: Processes command
result = await context_handler.process_with_context(command_text, websocket)
```

### 2. **Integration Layer**: `enhanced_context_wrapper.py`
```python
# Initializes all core components
self.context_manager = get_context_manager()      # ← context_manager.py
self.feedback_manager = get_feedback_manager()    # ← feedback_manager.py
self.jarvis_integration = get_jarvis_integration() # ← jarvis_integration.py
```

### 3. **Core Orchestration**: `context_manager.py`
```python
# Creates all detection/execution components
self.screen_detector = get_screen_state_detector()  # ← screen_state.py
self.command_queue = get_command_queue()           # ← command_queue.py
self.policy_engine = get_policy_engine()           # ← policy_engine.py
self.unlock_manager = get_unlock_manager()         # ← unlock_manager.py
```

### 4. **Execution Flow in ContextManager**:

```python
# Step 1: Detect screen state (screen_state.py)
screen_state = await self.screen_detector.get_screen_state()
# Returns: ScreenState.LOCKED

# Step 2: Queue command (command_queue.py)
queued_command = await self.command_queue.enqueue(...)

# Step 3: Policy decision (policy_engine.py)
decision, reason = await self.policy_engine.evaluate_unlock_request(...)
# Returns: PolicyDecision.AUTO_UNLOCK

# Step 4: Send feedback (feedback_manager.py)
await self.feedback_manager.send_contextual_feedback("screen_locked", "open Safari")

# Step 5: Unlock screen (unlock_manager.py)
success, msg = await self.unlock_manager.unlock_screen(...)

# Step 6: Execute command (unified_command_executor.py)
result = await executor.execute_command(...)

# Step 7: Send completion (feedback_manager.py)
await self.feedback_manager.send_contextual_feedback("command_complete", ...)
```

## Core Module Responsibilities

| Module | File | Purpose | When Called |
|--------|------|---------|-------------|
| **ScreenState** | `screen_state.py` | Detects if screen is locked | Before every command |
| **CommandQueue** | `command_queue.py` | Stores commands when locked | When screen is locked |
| **PolicyEngine** | `policy_engine.py` | Decides if auto-unlock allowed | Before unlocking |
| **UnlockManager** | `unlock_manager.py` | Performs actual unlock | When policy approves |
| **ContextManager** | `context_manager.py` | Orchestrates everything | Always active |
| **FeedbackManager** | `feedback_manager.py` | User communication | Throughout process |

## WebSocket Communication Flow

```
Frontend → jarvis_voice_api.py
         → enhanced_context_wrapper.py
         → feedback_manager.py
         → WebSocket: "I see your screen is locked..."
         
         → context_manager.py (orchestrates)
         → screen_state.py (detects)
         → command_queue.py (queues)
         → policy_engine.py (decides)
         → unlock_manager.py (unlocks)
         → unified_command_executor.py (executes)
         
         → feedback_manager.py
         → WebSocket: "I've successfully opened Safari..."
         → Frontend (speaks to user)
```

## Verification

To verify all modules are being used:

1. **Check logs**: Each module logs when it's called
2. **Check persistence files**:
   - `~/.jarvis/command_queue.json` - Queued commands
   - `~/.jarvis/unlock_audit.jsonl` - Unlock attempts
   - `~/.jarvis/context_intelligence_state.json` - System state
3. **Run test**: `python test_context_integration_live.py`

## Summary

**YES, all the core files ARE being used!** The integration happens through:

1. `jarvis_voice_api.py` imports `enhanced_context_wrapper.py`
2. `enhanced_context_wrapper.py` creates the `ContextManager`
3. `ContextManager` creates and uses ALL core modules
4. Each module performs its specific role in the flow

The system is fully integrated and operational! 🎉