# Feedback Learning System

**Adaptive intelligence that learns from user interactions**

---

## What Is This?

The Feedback Learning Loop makes JARVIS **smarter over time** by learning from your engagement and dismissal patterns. It's part of the "Invisible Assistant" UX philosophy: **help when needed, invisible when not**.

### Key Features

- ✅ **Learns from user responses** (engaged/dismissed/deferred/negative)
- ✅ **Adapts notification importance** (0.7x - 1.5x multiplier)
- ✅ **Auto-suppresses ignored patterns** (after 70% dismissal rate)
- ✅ **Timing intelligence** (learns best/worst hours)
- ✅ **Privacy-first** (all learning is local)
- ✅ **Transparent** (export all learned data)
- ✅ **User control** (reset anytime)

---

## Quick Start

### Record Feedback

```python
from backend.core.learning.feedback_loop import (
    get_feedback_loop,
    NotificationPattern,
    UserResponse,
)

loop = get_feedback_loop()

# User engaged with notification
await loop.record_feedback(
    pattern=NotificationPattern.TERMINAL_ERROR,
    response=UserResponse.ENGAGED,
    notification_text="ModuleNotFoundError detected",
    context={"window_type": "terminal"},
    time_to_respond=2.5,
)
```

### Check If Should Show Notification

```python
should_show, adjusted_importance = loop.should_show_notification(
    pattern=NotificationPattern.TERMINAL_ERROR,
    base_importance=0.7,
    context={"window_type": "terminal"},
)

if should_show:
    # Show notification with adjusted importance
    send_notification(text, importance=adjusted_importance)
else:
    # Suppressed - user doesn't care about this
    logger.debug("Notification suppressed by learning")
```

### Get Insights

```python
insights = loop.get_pattern_insights(NotificationPattern.TERMINAL_ERROR)

print(f"Total shown: {insights['total_shown']}")
print(f"Engagement rate: {insights['engagement_rate']:.0%}")
print(f"Recommendation: {insights['recommendation']}")
```

---

## How It Works

### 1. User Response Types

| Response | Meaning | Effect on Future |
|----------|---------|------------------|
| **ENGAGED** | User clicked, asked for details | Boost importance (1.2-1.5x) |
| **DISMISSED** | User ignored or said "no" | Reduce importance (0.7x) |
| **DEFERRED** | User said "not now" | Neutral, ask again later |
| **NEGATIVE_FEEDBACK** | User said "stop showing this" | Suppress completely (0.0x) |

### 2. Pattern Learning

After collecting feedback, the system:

- **Calculates engagement rate**: `engaged / total_shown`
- **Determines if valued**: Engagement ≥ 40% (after 3+ events)
- **Determines if ignored**: Dismissal ≥ 70% (after 5+ events)
- **Adjusts importance multiplier**: 0.0 - 1.5x based on patterns

### 3. Automatic Suppression

Patterns are auto-suppressed if:
- Dismissal rate ≥ 70% AND shown ≥ 5 times
- Negative feedback received ≥ 2 times

### 4. Timing Intelligence

After 50+ feedback events, the system learns:
- **Best hours** for notifications (highest engagement)
- **Worst hours** to avoid (highest dismissal)

```python
if loop.is_good_time_to_notify(hour=22):  # 10 PM
    # OK to show
else:
    # User historically dismisses at this hour
    defer_notification()
```

---

## Notification Patterns

```python
class NotificationPattern(str, Enum):
    TERMINAL_ERROR = "terminal_error"           # ModuleNotFoundError, SyntaxError, etc.
    TERMINAL_COMPLETION = "terminal_completion" # Build finished, tests passed
    TERMINAL_WARNING = "terminal_warning"       # Deprecation warnings, low disk space
    BROWSER_UPDATE = "browser_update"           # New version available
    CODE_DIAGNOSTIC = "code_diagnostic"         # Linting errors, type errors
    WORKFLOW_SUGGESTION = "workflow_suggestion" # "You could optimize this..."
    RESOURCE_WARNING = "resource_warning"       # High CPU, low memory
    SECURITY_ALERT = "security_alert"           # Vulnerability detected
    OTHER = "other"                             # Uncategorized
```

---

## Data Storage

All learned data is stored locally in:
```
~/.jarvis/learning/feedback.json
```

### What's Stored

- Last 100 feedback events (pattern, response, timestamp, context)
- Pattern statistics (engagement/dismissal rates, timing)
- Suppressed patterns (user explicitly disabled)
- Timing stats (engagement by hour of day)

### Example Data

```json
{
  "feedback_history": [
    {
      "pattern": "terminal_error",
      "response": "engaged",
      "timestamp": "2025-10-10T14:23:45",
      "notification_text": "ModuleNotFoundError detected",
      "context": {"window_type": "terminal"},
      "time_to_respond": 2.5
    }
  ],
  "pattern_stats": {
    "abc123": {
      "pattern": "terminal_error",
      "total_shown": 10,
      "engaged_count": 8,
      "dismissed_count": 2,
      "engagement_rate": 0.8,
      "dismissal_rate": 0.2
    }
  }
}
```

---

## User Control

### View What JARVIS Learned

```python
data = loop.export_learned_data()

print(f"Total events: {data['total_feedback_events']}")
print(f"Suppressed patterns: {len(data['suppressed_patterns'])}")

# Per-pattern insights
for pattern_hash, stats in data['pattern_stats'].items():
    print(f"{stats['pattern']}: {stats['engagement_rate']:.0%} engaged")
```

### Reset Learning

```python
# Reset all learning
await loop.reset_learning()

# Reset specific pattern only
await loop.reset_learning(pattern=NotificationPattern.WORKFLOW_SUGGESTION)
```

---

## Testing

### Run Automated Tests

```bash
# Unit tests
pytest backend/tests/test_feedback_learning_and_safety.py::TestFeedbackLearningLoop -v

# Live test
python -m backend.examples.live_feedback_test
```

### Manual Testing

See: `docs/MANUAL_TESTING_GUIDE.md`

---

## Integration Examples

### With Proactive Vision

```python
from backend.vision.intelligence.feedback_aware_vision import (
    create_feedback_aware_vision,
)

# Wraps ProactiveVisionIntelligence with learning
feedback_aware = await create_feedback_aware_vision(
    vision_analyzer=your_vision_analyzer,
    notification_callback=your_callback,
)

await feedback_aware.start_monitoring()
# Now all notifications are filtered through learning loop
```

### With Async Pipeline

```python
# In async_pipeline.py command processing

if context.metadata.get('is_followup_response'):
    # Record user's response
    await feedback_loop.record_feedback(
        pattern=context.metadata['notification_pattern'],
        response=classify_user_response(context.text),
        notification_text=context.metadata['notification_text'],
        context={'intent': context.intent},
        time_to_respond=context.metadata['time_to_respond'],
    )
```

---

## Architecture

```
User Sees Notification
        ↓
    Responds (yes/no/later/stop)
        ↓
FeedbackLearningLoop.record_feedback()
        ↓
    Updates pattern stats
        ↓
    Calculates new importance multiplier
        ↓
    Saves to disk
        ↓
Next time: should_show_notification()
        ↓
    Returns (should_show, adjusted_importance)
        ↓
    JARVIS shows or suppresses
```

---

## Performance

- **Memory:** < 2MB (1000 events + stats)
- **CPU:** < 1ms per operation
- **Disk I/O:** Async, non-blocking writes
- **Startup:** Loads all data in < 10ms

---

## Philosophy Alignment

This implements the "Invisible Assistant" UX:

✅ **Intelligence over automation** - Learns, doesn't hardcode
✅ **Context over commands** - Considers patterns, not just text
✅ **Assistance over intrusion** - Reduces noise, respects attention
✅ **Privacy by design** - Local learning, user control
✅ **Trust through transparency** - Export all learned data

---

## FAQ

**Q: Will this slow down JARVIS?**
A: No. All operations are async and take < 1ms.

**Q: What if I want to reset learning?**
A: `await loop.reset_learning()` or delete the JSON file.

**Q: Can I see what JARVIS learned?**
A: Yes! `loop.export_learned_data()` or view the JSON file.

**Q: Does it work offline?**
A: Yes! All learning is local, no external API calls.

**Q: How many events before patterns are learned?**
A: Valued patterns: 3+ events, Ignored patterns: 5+ events

---

## Files

- **Implementation:** `backend/core/learning/feedback_loop.py`
- **Tests:** `backend/tests/test_feedback_learning_and_safety.py`
- **Demo:** `backend/examples/demo_feedback_and_safety.py`
- **Integration:** `backend/vision/intelligence/feedback_aware_vision.py`
- **Docs:** `docs/FEEDBACK_LEARNING_AND_COMMAND_SAFETY.md`

---

**Status:** ✅ Production Ready
**Version:** 1.0
**License:** Same as JARVIS

