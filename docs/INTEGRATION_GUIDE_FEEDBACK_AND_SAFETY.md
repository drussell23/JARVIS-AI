# Integration Guide: Feedback Learning & Command Safety

**Quick Start:** How to integrate the new feedback learning and command safety systems into existing JARVIS components.

---

## Quick Integration Checklist

- [ ] Add feedback loop to `main.py` initialization
- [ ] Integrate with `async_pipeline.py` for command routing
- [ ] Update `follow_up_plugin.py` to use terminal intelligence (✅ Done)
- [ ] Add safety warnings to any command execution paths
- [ ] Configure storage paths in environment

---

## Integration Point 1: Main Initialization

**File:** `backend/main.py`

### Add to startup sequence

```python
# Around line 50-100 (during component initialization)

async def initialize_jarvis():
    """Initialize JARVIS with all components."""

    # ... existing initialization ...

    # Initialize feedback learning loop
    from backend.core.learning.feedback_loop import FeedbackLearningLoop
    from pathlib import Path

    feedback_storage = Path.home() / ".jarvis" / "learning" / "feedback.json"
    feedback_loop = FeedbackLearningLoop(storage_path=feedback_storage)

    logger.info(f"[MAIN] Initialized feedback learning loop")

    # Initialize command safety classifier
    from backend.system_control.command_safety import get_command_classifier

    command_classifier = get_command_classifier()
    logger.info(f"[MAIN] Initialized command safety classifier")

    # Store in global context for access by other components
    app.state.feedback_loop = feedback_loop
    app.state.command_classifier = command_classifier

    # ... rest of initialization ...
```

---

## Integration Point 2: Async Pipeline

**File:** `backend/core/async_pipeline.py`

### Add feedback tracking to command processing

```python
# Around the command processing section (line 500-600)

async def process_command(self, context: PipelineContext) -> PipelineContext:
    """Process command through pipeline."""

    # ... existing processing ...

    # Record feedback if this was in response to a notification
    if context.metadata.get('is_followup_response'):
        from backend.core.learning.feedback_loop import (
            NotificationPattern,
            UserResponse,
            get_feedback_loop,
        )

        feedback_loop = get_feedback_loop()

        # Determine response type from user input
        user_text = context.text.lower()
        if any(word in user_text for word in ['yes', 'sure', 'ok', 'details']):
            response_type = UserResponse.ENGAGED
        elif any(word in user_text for word in ['no', 'nope', 'dismiss', 'skip']):
            response_type = UserResponse.DISMISSED
        else:
            response_type = UserResponse.ENGAGED  # Default

        # Record feedback
        await feedback_loop.record_feedback(
            pattern=context.metadata.get('notification_pattern', NotificationPattern.OTHER),
            response=response_type,
            notification_text=context.metadata.get('notification_text', ''),
            context={
                'window_type': context.metadata.get('window_type', 'unknown'),
                'intent': context.intent,
            },
            time_to_respond=context.metadata.get('time_to_respond', 0.0),
        )

    return context
```

---

## Integration Point 3: Proactive Vision Intelligence

**File:** `backend/vision/proactive_vision_intelligence.py`

### Wrap with feedback-aware layer

```python
# In your main startup/initialization code

from backend.vision.intelligence.feedback_aware_vision import (
    create_feedback_aware_vision,
)

# Replace direct ProactiveVisionIntelligence usage with feedback-aware version
async def create_vision_system():
    """Create vision system with feedback learning."""

    # Create feedback-aware vision (wraps ProactiveVisionIntelligence)
    feedback_aware_vision = await create_feedback_aware_vision(
        vision_analyzer=your_claude_vision_analyzer,
        notification_callback=your_notification_callback,
        storage_path=Path.home() / ".jarvis" / "learning" / "feedback.json",
    )

    # Start monitoring
    await feedback_aware_vision.start_monitoring()

    return feedback_aware_vision
```

---

## Integration Point 4: Terminal Command Execution

**File:** Any file that executes terminal commands (e.g., `backend/system_control/*.py`)

### Add safety checks before execution

```python
from backend.system_control.command_safety import get_command_classifier, SafetyTier

async def execute_terminal_command(command: str, auto_execute: bool = False):
    """Execute terminal command with safety checks."""

    # Classify command
    classifier = get_command_classifier()
    classification = classifier.classify(command)

    # Safety gates
    if classification.tier == SafetyTier.RED:
        if not auto_execute:
            # Always require explicit confirmation for RED tier
            logger.warning(f"[SAFETY] Blocked RED tier command: {command}")
            return {
                'success': False,
                'error': f'Command "{command}" is potentially destructive',
                'reasoning': classification.reasoning,
                'suggested_alternative': classification.suggested_alternative,
                'requires_confirmation': True,
            }

    elif classification.tier == SafetyTier.YELLOW:
        if not auto_execute:
            # Require confirmation for YELLOW tier
            logger.info(f"[SAFETY] Requesting confirmation for: {command}")
            return {
                'success': False,
                'requires_confirmation': True,
                'impact': classification.reasoning,
                'dry_run_available': classification.dry_run_available,
            }

    # GREEN tier or confirmed - execute
    logger.info(f"[SAFETY] Executing {classification.tier.value} tier command: {command}")

    # ... actual execution logic ...

    return {'success': True, 'output': output}
```

---

## Integration Point 5: Voice Command Handler

**File:** `backend/voice/*.py` (wherever voice commands are processed)

### Add command safety to voice-triggered commands

```python
async def handle_voice_command(command_text: str):
    """Handle voice command with safety classification."""

    # Extract command from natural language
    # e.g., "run npm install" -> "npm install"
    extracted_command = extract_shell_command(command_text)

    if extracted_command:
        # Classify before execution
        from backend.system_control.command_safety import get_command_classifier

        classifier = get_command_classifier()
        classification = classifier.classify(extracted_command)

        # Provide voice feedback about safety
        if classification.tier.value == 'red':
            await speak(
                f"Warning: That command is potentially destructive. "
                f"I need explicit confirmation to run: {extracted_command}"
            )
            return

        elif classification.tier.value == 'yellow':
            await speak(
                f"This command will {classification.reasoning.lower()}. "
                f"Should I proceed?"
            )
            # Wait for confirmation
            return

        # GREEN tier - safe to execute
        await speak(f"Running: {extracted_command}")
        # ... execute ...
```

---

## Integration Point 6: WebSocket API

**File:** `backend/api/*.py` (WebSocket endpoints)

### Expose feedback insights via API

```python
from fastapi import WebSocket

@app.websocket("/ws/learning-insights")
async def learning_insights_endpoint(websocket: WebSocket):
    """Provide real-time learning insights to frontend."""
    await websocket.accept()

    try:
        from backend.core.learning.feedback_loop import get_feedback_loop

        feedback_loop = get_feedback_loop()

        while True:
            # Send current insights
            insights = {
                'total_events': len(feedback_loop.feedback_history),
                'pattern_stats': {},
                'suppressed_count': len(feedback_loop.suppressed_patterns),
                'timing_insights': {
                    'best_hours': list(feedback_loop.best_hours),
                    'worst_hours': list(feedback_loop.worst_hours),
                },
            }

            # Get per-pattern insights
            from backend.core.learning.feedback_loop import NotificationPattern
            for pattern in NotificationPattern:
                pattern_insights = feedback_loop.get_pattern_insights(pattern)
                if pattern_insights.get('has_data'):
                    insights['pattern_stats'][pattern.value] = pattern_insights

            await websocket.send_json(insights)

            # Update every 5 seconds
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        logger.info("[WS] Learning insights client disconnected")
```

---

## Integration Point 7: Context Bridge

**File:** `backend/core/unified_context_bridge.py`

### Add feedback loop to context bridge

```python
class UnifiedContextBridge:
    """Enhanced with feedback learning."""

    def __init__(self, config=None, context_store=None):
        # ... existing initialization ...

        # Add feedback loop integration
        from backend.core.learning.feedback_loop import get_feedback_loop
        self.feedback_loop = get_feedback_loop()

        logger.info("[BRIDGE] Integrated feedback learning loop")

    async def track_pending_question(
        self,
        question_text: str,
        window_type: str,
        # ... other params ...
    ) -> Optional[str]:
        """Track pending question with feedback awareness."""

        # ... existing tracking logic ...

        # Determine if we should show based on learned patterns
        from backend.core.learning.feedback_loop import NotificationPattern

        # Map window type to pattern
        pattern_map = {
            'terminal': NotificationPattern.TERMINAL_ERROR,
            'browser': NotificationPattern.BROWSER_UPDATE,
            'code': NotificationPattern.CODE_DIAGNOSTIC,
        }
        pattern = pattern_map.get(window_type, NotificationPattern.OTHER)

        # Check with feedback loop
        should_show, adjusted_importance = self.feedback_loop.should_show_notification(
            pattern=pattern,
            base_importance=0.7,
            context={'window_type': window_type},
        )

        if not should_show:
            logger.info(
                f"[BRIDGE] Suppressed notification for {window_type} "
                "(user historically dismisses this pattern)"
            )
            return None

        # ... continue with tracking ...
```

---

## Environment Configuration

**File:** `backend/.env` or `backend/config/*.py`

### Add configuration variables

```bash
# Feedback Learning Configuration
JARVIS_LEARNING_STORAGE_PATH="${HOME}/.jarvis/learning/feedback.json"
JARVIS_MAX_FEEDBACK_HISTORY=1000
JARVIS_MIN_PATTERN_CONFIDENCE=0.75
JARVIS_TIMING_LEARNING_ENABLED=true
JARVIS_FEEDBACK_LOOP_ENABLED=true

# Command Safety Configuration
JARVIS_COMMAND_SAFETY_ENABLED=true
JARVIS_AUTO_EXECUTE_GREEN_COMMANDS=true
JARVIS_REQUIRE_CONFIRMATION_YELLOW=true
JARVIS_BLOCK_RED_COMMANDS=false  # Set to true for strict safety
```

---

## Testing Your Integration

### 1. Unit Tests

```bash
# Test feedback loop integration
pytest backend/tests/test_feedback_learning_and_safety.py -v

# Test with your existing tests
pytest backend/tests/ -k "feedback or safety" -v
```

### 2. Integration Test

```python
# backend/tests/test_integration_feedback.py

async def test_end_to_end_feedback_flow():
    """Test complete feedback flow."""

    # 1. User sees notification
    notification = {
        'pattern': NotificationPattern.TERMINAL_ERROR,
        'text': 'ModuleNotFoundError detected',
    }

    # 2. User responds
    user_response = UserResponse.ENGAGED

    # 3. System records feedback
    feedback_loop = get_feedback_loop()
    await feedback_loop.record_feedback(
        pattern=notification['pattern'],
        response=user_response,
        notification_text=notification['text'],
        context={'window_type': 'terminal'},
        time_to_respond=2.0,
    )

    # 4. Next time, importance should be adjusted
    should_show, adjusted = feedback_loop.should_show_notification(
        pattern=notification['pattern'],
        base_importance=0.7,
        context={'window_type': 'terminal'},
    )

    assert should_show is True  # User engaged, so keep showing
    assert adjusted >= 0.7  # Should maintain or boost importance
```

### 3. Manual Testing

```bash
# Run the demo
python -m backend.examples.demo_feedback_and_safety

# Start JARVIS with feedback learning
python backend/main.py --enable-feedback-learning

# Trigger a terminal error notification
# Respond "yes" or "no"
# Check that pattern is learned:
# → backend/learning/feedback.json
```

---

## Monitoring & Debugging

### Check Learning Data

```python
from backend.core.learning.feedback_loop import get_feedback_loop
import json

loop = get_feedback_loop()

# Export all learned data
data = loop.export_learned_data()
print(json.dumps(data, indent=2))
```

### View Pattern Insights

```python
from backend.core.learning.feedback_loop import NotificationPattern

insights = loop.get_pattern_insights(NotificationPattern.TERMINAL_ERROR)
print(f"Engagement rate: {insights['engagement_rate']:.0%}")
print(f"Recommendation: {insights['recommendation']}")
```

### Check Command Classification

```bash
# Quick command safety check
python -c "
from backend.system_control.command_safety import get_command_classifier
classifier = get_command_classifier()
result = classifier.classify('rm -rf /tmp/test')
print(f'Tier: {result.tier.value}')
print(f'Safe: {result.is_safe}')
print(f'Destructive: {result.is_destructive}')
"
```

---

## Common Patterns

### Pattern 1: Check Before Showing Notification

```python
from backend.core.learning.feedback_loop import get_feedback_loop, NotificationPattern

async def maybe_show_notification(pattern, text, base_importance=0.7):
    """Show notification only if learned patterns allow."""

    loop = get_feedback_loop()
    should_show, adjusted = loop.should_show_notification(
        pattern=pattern,
        base_importance=base_importance,
        context={'source': 'vision_system'},
    )

    if should_show:
        # Show with adjusted importance
        await show_notification(text, importance=adjusted)
    else:
        # Suppressed based on learning
        logger.debug(f"Suppressed: {text}")
```

### Pattern 2: Record User Response

```python
async def on_notification_response(notification_id, user_response_text):
    """Record user's response to notification."""

    from backend.core.learning.feedback_loop import (
        get_feedback_loop,
        UserResponse,
    )

    # Map text to UserResponse enum
    response_map = {
        'yes': UserResponse.ENGAGED,
        'no': UserResponse.DISMISSED,
        'later': UserResponse.DEFERRED,
        'stop': UserResponse.NEGATIVE_FEEDBACK,
    }
    response = response_map.get(user_response_text.lower(), UserResponse.DISMISSED)

    # Record
    loop = get_feedback_loop()
    await loop.record_feedback(
        pattern=notification_metadata['pattern'],
        response=response,
        notification_text=notification_metadata['text'],
        context=notification_metadata['context'],
        time_to_respond=time.time() - notification_metadata['shown_at'],
    )
```

### Pattern 3: Safe Command Execution

```python
async def safe_execute(command: str, require_confirmation: bool = True):
    """Execute command with safety checks."""

    from backend.system_control.command_safety import get_command_classifier, SafetyTier

    classifier = get_command_classifier()
    result = classifier.classify(command)

    # Safety gates
    if result.tier == SafetyTier.RED and require_confirmation:
        return {
            'blocked': True,
            'reason': result.reasoning,
            'alternative': result.suggested_alternative,
        }

    # Execute
    return await execute_command(command)
```

---

## Migration Checklist

If you're adding this to an existing JARVIS installation:

- [ ] **Backup** existing notification data
- [ ] **Test** in development environment first
- [ ] **Gradually enable** feedback learning (start with logging only)
- [ ] **Monitor** for 1 week before full rollout
- [ ] **Document** any custom patterns added
- [ ] **Communicate** to users what's changing

---

## Rollback Plan

If you need to disable feedback learning:

```python
# In main.py or config
FEEDBACK_LEARNING_ENABLED = False

# In code
if app.config.get('FEEDBACK_LEARNING_ENABLED', False):
    # Use feedback-aware system
    ...
else:
    # Use original system
    ...
```

---

## Performance Considerations

### Memory Usage

- Feedback history: ~1KB per event × 1000 max = ~1MB
- Pattern stats: ~0.5KB per pattern × ~50 patterns = ~25KB
- **Total:** < 2MB in memory

### Disk I/O

- Writes on every feedback event (async, non-blocking)
- Reads once on startup
- Uses JSON for human-readability (could optimize to binary if needed)

### CPU Impact

- Classification: ~0.1ms per command (regex matching)
- Feedback recording: ~0.5ms (update stats + write to disk)
- **Negligible** impact on overall system performance

---

## FAQ

**Q: Will feedback learning slow down JARVIS?**
A: No. All operations are async and take <1ms. Disk writes are non-blocking.

**Q: What if user wants to reset learning?**
A: Provide UI option that calls `await loop.reset_learning()` or deletes the JSON file.

**Q: Can multiple JARVIS instances share learning?**
A: Yes, if they point to the same storage path. Use file locking if concurrent access.

**Q: How to add new notification patterns?**
A: Add to `NotificationPattern` enum in `feedback_loop.py`, no other changes needed.

**Q: Does command safety work offline?**
A: Yes! All classification is pattern-based, no external API calls.

---

## Support

If you encounter issues:

1. Check logs: `tail -f backend/logs/jarvis.log | grep -E "(FEEDBACK|SAFETY)"`
2. Verify storage path is writable: `ls -la ~/.jarvis/learning/`
3. Run tests: `pytest backend/tests/test_feedback_learning_and_safety.py -v`
4. Check demo: `python -m backend.examples.demo_feedback_and_safety`

---

**Integration Status:** ✅ Ready
**Backward Compatible:** ✅ Yes
**Breaking Changes:** ❌ None

