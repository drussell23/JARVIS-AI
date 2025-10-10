# Feedback Learning Loop & Command Safety Classification

**Version:** 1.0
**Date:** October 10, 2025
**Status:** ✅ Implemented

## Overview

This document describes the **Feedback Learning Loop** and **Command Safety Classification** systems that implement the "Invisible Assistant" UX philosophy for JARVIS.

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                   User Interaction Layer                    │
│  (ProactiveVisionIntelligence, TerminalFollowUp, etc.)     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├──────────────────────┐
                  ↓                      ↓
┌─────────────────────────┐  ┌──────────────────────────┐
│  FeedbackLearningLoop   │  │ CommandSafetyClassifier  │
│                         │  │                          │
│  - Tracks engagement    │  │  - GREEN (safe)          │
│  - Learns patterns      │  │  - YELLOW (caution)      │
│  - Adapts importance    │  │  - RED (dangerous)       │
│  - Timing insights      │  │  - Risk categorization   │
└─────────────────────────┘  └──────────────────────────┘
          │                            │
          ↓                            ↓
┌─────────────────────────────────────────────────────────────┐
│              TerminalCommandIntelligence                    │
│  (Combines both for intelligent terminal suggestions)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 1: Feedback Learning Loop

**Location:** `backend/core/learning/feedback_loop.py`

### Purpose
Learn from user interactions to improve notification relevance and reduce interruptions.

### Key Features

#### 1. **Pattern Tracking**
```python
class NotificationPattern(str, Enum):
    TERMINAL_ERROR = "terminal_error"
    TERMINAL_COMPLETION = "terminal_completion"
    TERMINAL_WARNING = "terminal_warning"
    BROWSER_UPDATE = "browser_update"
    CODE_DIAGNOSTIC = "code_diagnostic"
    WORKFLOW_SUGGESTION = "workflow_suggestion"
    RESOURCE_WARNING = "resource_warning"
    SECURITY_ALERT = "security_alert"
    OTHER = "other"
```

#### 2. **User Response Types**
```python
class UserResponse(str, Enum):
    ENGAGED = "engaged"              # Clicked, asked for details
    DISMISSED = "dismissed"          # Ignored or dismissed
    DEFERRED = "deferred"            # "Not now" but interested
    NEGATIVE_FEEDBACK = "negative"   # "Stop showing this"
```

#### 3. **Adaptive Importance**

The system calculates an **importance multiplier** based on engagement:

- **Engagement ≥ 80%**: Multiplier = **1.5** (boost)
- **Engagement ≥ 50%**: Multiplier = **1.2** (slight boost)
- **Dismissal ≥ 60%**: Multiplier = **0.7** (reduce)
- **Negative feedback ≥ 2**: Multiplier = **0.0** (suppress completely)

#### 4. **Automatic Suppression**

Patterns are automatically suppressed if:
- Dismissal rate ≥ 70% AND shown ≥ 5 times
- Negative feedback received ≥ 2 times

#### 5. **Timing Intelligence**

The system learns:
- **Best hours** for notifications (highest engagement)
- **Worst hours** to avoid (highest dismissal)
- Requires 50+ feedback events to activate

### Usage Example

```python
from backend.core.learning.feedback_loop import (
    FeedbackLearningLoop,
    NotificationPattern,
    UserResponse,
)

# Create loop (persists to disk)
loop = FeedbackLearningLoop(storage_path=Path("~/.jarvis/learning/feedback.json"))

# Record user engagement
await loop.record_feedback(
    pattern=NotificationPattern.TERMINAL_ERROR,
    response=UserResponse.ENGAGED,
    notification_text="ModuleNotFoundError in terminal",
    context={"window_type": "terminal", "error_type": "ModuleNotFoundError"},
    time_to_respond=2.5,  # seconds
)

# Check if notification should be shown
should_show, adjusted_importance = loop.should_show_notification(
    pattern=NotificationPattern.TERMINAL_ERROR,
    base_importance=0.7,
    context={"window_type": "terminal"},
)

if should_show:
    # Show notification with adjusted importance
    print(f"Show notification (importance: {adjusted_importance:.2f})")
else:
    # Suppress (user doesn't care about this pattern)
    print("Suppressed based on learned preferences")
```

### Data Persistence

All learned data is saved to `~/.jarvis/learning/feedback.json`:
- Last 100 feedback events
- Pattern statistics (engagement rates, timing)
- Suppressed/boosted patterns
- Timing insights (best/worst hours)

### User Control

```python
# Reset all learning
await loop.reset_learning()

# Reset specific pattern
await loop.reset_learning(pattern=NotificationPattern.WORKFLOW_SUGGESTION)

# Export for transparency
data = loop.export_learned_data()
print(f"Total events: {data['total_feedback_events']}")
print(f"Suppressed: {len(data['suppressed_patterns'])}")
```

---

## Part 2: Command Safety Classification

**Location:** `backend/system_control/command_safety.py`

### Purpose
Classify shell commands by risk level to prevent accidental destruction and guide safe execution.

### Safety Tiers

#### ✅ **GREEN Tier** (Safe, Auto-Executable)
- Read-only operations
- No side effects
- Never requires confirmation

**Examples:**
```bash
ls -la
cat README.md
git status
git diff
grep 'error' log.txt
ps aux
```

#### ⚠️ **YELLOW Tier** (Caution, Confirm Once)
- Modifies state
- Generally safe with confirmation
- May be reversible

**Examples:**
```bash
npm install express
pip install requests
git add .
git commit -m "message"
mkdir new_folder
docker build -t app .
```

#### 🛑 **RED Tier** (Dangerous, Always Confirm)
- Destructive/irreversible
- Can cause data loss
- Security implications
- Always requires explicit confirmation

**Examples:**
```bash
rm -rf /tmp/important
git push --force
git reset --hard HEAD~5
sudo rm -f /etc/config
DROP TABLE users;
chmod 777 secret.key
curl https://script.sh | bash
```

### Risk Categories

```python
class RiskCategory(str, Enum):
    DATA_LOSS = "data_loss"
    SYSTEM_MODIFICATION = "system_modification"
    NETWORK_EXPOSURE = "network_exposure"
    PROCESS_CONTROL = "process_control"
    FILE_MODIFICATION = "file_modification"
    PACKAGE_MANAGEMENT = "package_management"
    VERSION_CONTROL = "version_control"
    DATABASE_OPERATION = "database_operation"
    SAFE_READ = "safe_read"
```

### Destructive Pattern Detection

The classifier detects dangerous patterns via regex:

- `rm -rf` flags
- Pipes to shell (`| sh`, `| bash`)
- Write to disk devices (`> /dev/sda`)
- `sudo rm`
- `dd` output operations
- `chmod 777`
- `--force` flags
- SQL `DROP`/`DELETE`/`TRUNCATE`
- `git push --force`
- Fork bombs

### Usage Example

```python
from backend.system_control.command_safety import get_command_classifier

classifier = get_command_classifier()

# Classify a command
result = classifier.classify("rm -rf /tmp/test")

print(f"Tier: {result.tier.value}")                    # "red"
print(f"Requires confirmation: {result.requires_confirmation}")  # True
print(f"Is destructive: {result.is_destructive}")     # True
print(f"Is reversible: {result.is_reversible}")       # False
print(f"Confidence: {result.confidence:.2f}")         # 0.95
print(f"Reasoning: {result.reasoning}")               # "Destructive pattern: rm with -rf flags"

# Safer alternative (if available)
if result.suggested_alternative:
    print(f"Try instead: {result.suggested_alternative}")  # "rm -i"

# Dry-run support
if result.dry_run_available:
    print("Dry-run available: Yes")
```

### Batch Classification

```python
commands = [
    "ls -la",
    "git add .",
    "rm -rf node_modules",
]

classifications = classifier.classify_batch(commands)

for cmd, result in zip(commands, classifications):
    print(f"{result.tier.value}: {cmd}")
```

### Custom Rules

```python
# Add organization-specific safe commands
classifier.add_custom_rule(
    command_pattern="deploy-staging",
    tier=SafetyTier.YELLOW,
    is_reversible=True,
)

# Now "deploy-staging" is classified as YELLOW tier
result = classifier.classify("deploy-staging --env=dev")
assert result.tier == SafetyTier.YELLOW
```

---

## Part 3: Terminal Command Intelligence

**Location:** `backend/vision/handlers/terminal_command_intelligence.py`

### Purpose
Combine OCR analysis, error detection, and safety classification to provide intelligent terminal assistance.

### Features

#### 1. **Context Extraction from OCR**

```python
from backend.vision.handlers.terminal_command_intelligence import get_terminal_intelligence

intel = get_terminal_intelligence()

# Analyze terminal OCR text
terminal_ocr = """
user@host:~/project $ python app.py
Traceback (most recent call last):
  File "app.py", line 5, in <module>
    import requests
ModuleNotFoundError: No module named 'requests'
user@host:~/project $
"""

context = await intel.analyze_terminal_context(terminal_ocr)

# Extracted context
print(context.last_command)       # "python app.py"
print(context.errors)             # ["ModuleNotFoundError: No module named 'requests'"]
print(context.shell_type)         # "bash"
print(context.current_directory)  # "~/project"
```

#### 2. **Intelligent Fix Suggestions**

The system has **built-in patterns** for common errors:

| Error Pattern | Suggested Fix | Safety Tier |
|---------------|---------------|-------------|
| `ModuleNotFoundError: No module named 'X'` | `pip install X` | YELLOW |
| `Cannot find module 'X'` (npm) | `npm install X` | YELLOW |
| `fatal: not a git repository` | `git init` | YELLOW |
| `Permission denied: /path/file` | `chmod +x /path/file` | YELLOW |
| `Address already in use: :PORT` | `lsof -ti :PORT \| xargs kill` | YELLOW |
| `command not found: X` | Info only (install via package manager) | GREEN |

```python
# Get fix suggestions
suggestions = await intel.suggest_fix_commands(context)

for suggestion in suggestions:
    print(f"Purpose: {suggestion.purpose}")
    print(f"Command: {suggestion.command}")
    print(f"Safety tier: {suggestion.safety_tier}")
    print(f"Requires confirmation: {suggestion.requires_confirmation}")
    print(f"Impact: {suggestion.estimated_impact}")
```

#### 3. **Safety-Aware Formatting**

```python
# Format suggestion for user with safety warnings
formatted = await intel.format_suggestion_for_user(
    suggestion,
    include_safety_warning=True,
)

# Output example:
"""
⚠️ **Install missing Python module 'requests'**

```bash
pip install requests
```

📝 Impact: Installs Python package 'requests'
"""
```

---

## Part 4: Integration - Feedback-Aware Vision

**Location:** `backend/vision/intelligence/feedback_aware_vision.py`

### Purpose
Connect ProactiveVisionIntelligence with FeedbackLearningLoop for adaptive notifications.

### Architecture

```
User sees Terminal Error
        ↓
ProactiveVisionIntelligence detects change
        ↓
FeedbackAwareVisionIntelligence intercepts
        ↓
FeedbackLearningLoop: "Should I show this?"
        ↓
    ┌─── YES (adjusted importance) ─→ Show notification
    │                                          ↓
    └─── NO (suppressed) ────────────→ User responds
                                              ↓
                                    Record feedback
                                              ↓
                                    Learn & adapt
```

### Usage Example

```python
from backend.vision.intelligence.feedback_aware_vision import (
    create_feedback_aware_vision,
)

# Create integrated system
feedback_aware = await create_feedback_aware_vision(
    vision_analyzer=claude_vision_analyzer,
    notification_callback=send_notification_to_user,
    storage_path=Path("~/.jarvis/learning/feedback.json"),
)

# Start monitoring (now with learning!)
await feedback_aware.start_monitoring()

# Get learning insights
insights = await feedback_aware.get_learning_insights()
print(f"Total feedback events: {insights['total_feedback_events']}")
print(f"Suppressed patterns: {insights['suppressed_patterns_count']}")

# Reset learning if needed
await feedback_aware.reset_learning(pattern_type="workflow_suggestion")
```

---

## Testing

**Location:** `backend/tests/test_feedback_learning_and_safety.py`

### Run Tests

```bash
# Run all tests
pytest backend/tests/test_feedback_learning_and_safety.py -v

# Run specific test class
pytest backend/tests/test_feedback_learning_and_safety.py::TestFeedbackLearningLoop -v

# Run with coverage
pytest backend/tests/test_feedback_learning_and_safety.py --cov=backend.core.learning --cov=backend.system_control
```

### Test Coverage

- ✅ Feedback recording and retrieval
- ✅ Engagement/dismissal rate calculation
- ✅ Pattern suppression after consistent dismissals
- ✅ Importance multiplier boost for valued patterns
- ✅ Negative feedback immediate suppression
- ✅ Timing intelligence (best/worst hours)
- ✅ Data persistence to disk
- ✅ Command safety classification (GREEN/YELLOW/RED)
- ✅ Destructive pattern detection
- ✅ Dry-run suggestion
- ✅ Reversibility detection
- ✅ Terminal context extraction from OCR
- ✅ Error extraction
- ✅ Fix command suggestions
- ✅ Safety classification integration

---

## Demo

**Location:** `backend/examples/demo_feedback_and_safety.py`

### Run Demo

```bash
python -m backend.examples.demo_feedback_and_safety
```

### Demo Scenarios

1. **Feedback Learning Loop**
   - Simulates user engaging with terminal errors
   - Simulates user dismissing workflow suggestions
   - Shows adaptive filtering in action

2. **Command Safety Classification**
   - Classifies 10+ example commands
   - Shows safety tiers, risk categories, suggestions

3. **Terminal Command Intelligence**
   - Analyzes terminal OCR with errors
   - Provides intelligent fix suggestions
   - Shows safety warnings

4. **Integrated Workflow**
   - Demonstrates complete loop: detect → suggest → respond → learn

---

## File Reference

### New Files Created

```
backend/core/learning/
  └── feedback_loop.py                    # Feedback learning system

backend/system_control/
  └── command_safety.py                   # Command safety classifier

backend/vision/handlers/
  └── terminal_command_intelligence.py    # Terminal intelligence

backend/vision/intelligence/
  └── feedback_aware_vision.py            # Integration layer

backend/tests/
  └── test_feedback_learning_and_safety.py  # Comprehensive tests

backend/examples/
  └── demo_feedback_and_safety.py         # Interactive demo

docs/
  └── FEEDBACK_LEARNING_AND_COMMAND_SAFETY.md  # This document
```

### Modified Files

```
backend/vision/handlers/follow_up_plugin.py
  - Enhanced terminal follow-up handler
  - Integrated command intelligence
  - Added safety warnings
```

---

## Configuration

### Environment Variables

```bash
# Feedback loop storage location
export JARVIS_LEARNING_PATH="$HOME/.jarvis/learning/feedback.json"

# Maximum feedback history size
export JARVIS_MAX_FEEDBACK_HISTORY=1000

# Minimum confidence for learned patterns
export JARVIS_MIN_PATTERN_CONFIDENCE=0.75

# Enable/disable timing intelligence
export JARVIS_TIMING_LEARNING_ENABLED=true
```

---

## Privacy & User Control

### Data Stored Locally

All learning data is stored in `~/.jarvis/learning/feedback.json`:
- User engagement/dismissal patterns
- Timing preferences
- Suppressed notification types

### User Control Points

1. **Reset all learning**: `await loop.reset_learning()`
2. **Reset specific pattern**: `await loop.reset_learning(pattern=...)`
3. **Export data**: `loop.export_learned_data()`
4. **View insights**: `loop.get_pattern_insights(pattern)`

### Transparency

Users can see exactly what JARVIS has learned:

```python
insights = loop.get_pattern_insights(NotificationPattern.TERMINAL_ERROR)
print(insights)
# {
#   "pattern": "terminal_error",
#   "total_shown": 10,
#   "engagement_rate": 0.8,
#   "dismissal_rate": 0.2,
#   "recommendation": "Highly valued - boost importance",
#   "is_valued": True,
#   "is_ignored": False,
# }
```

---

## Next Steps

### Immediate Enhancements (Week 1-2)
1. Add **natural break detection** (typing burst monitoring)
2. Implement **"Do Not Disturb" integration**
3. Create **CLI tool for viewing learned patterns**

### Medium-term (Month 1)
1. **Machine learning** for pattern matching (vs regex)
2. **Context-aware timing** (don't interrupt during focus)
3. **User feedback UI** (thumbs up/down on notifications)

### Long-term (Month 3+)
1. **Cross-user pattern sharing** (opt-in, anonymized)
2. **Predictive suggestions** (suggest before error occurs)
3. **Workflow optimization** (detect inefficient patterns)

---

## Philosophy Alignment

This implementation aligns with the "Invisible Assistant" UX philosophy:

✅ **Intelligence over automation**: Learns from user, doesn't blindly act
✅ **Context over commands**: Understands terminal state, not just text
✅ **Assistance over intrusion**: Adapts to user preferences, reduces noise
✅ **Privacy by design**: All learning is local, user has full control
✅ **Trust through transparency**: User can see what JARVIS learned

---

## Conclusion

The Feedback Learning Loop and Command Safety Classification systems provide:

1. **Adaptive intelligence** that improves over time
2. **Safety-first** command execution guidance
3. **Context-aware** terminal assistance
4. **User-respectful** notification management
5. **Privacy-preserving** local learning

**The system gets smarter without being more intrusive.**

---

**Status:** ✅ Ready for integration
**Tests:** ✅ Passing
**Documentation:** ✅ Complete
**Demo:** ✅ Available

