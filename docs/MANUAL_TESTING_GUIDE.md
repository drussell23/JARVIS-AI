# Manual Testing Guide: Feedback Learning & Command Safety

**How to verify the systems work when running JARVIS**

---

## Quick Test (5 minutes)

### 1. Run the Automated Test Suite

```bash
# From the project root
python -m backend.examples.live_feedback_test
```

**Expected output:**
```
✓ Feedback Loop Basic: PASSED
✓ Command Safety: PASSED
✓ Terminal Intelligence: PASSED
✓ Persistence: PASSED
✓ Integration: PASSED
ℹ Live Jarvis: SKIPPED (JARVIS not running)

Results: 5/6 passed, 0 failed, 1 skipped
✨ SUCCESS! All tests passed!
```

### 2. Run the Interactive Demo

```bash
python -m backend.examples.demo_feedback_and_safety
```

**Expected output:**
- Demo 1: Shows feedback learning in action
- Demo 2: Classifies commands by safety tier
- Demo 3: Analyzes terminal errors and suggests fixes
- Demo 4: Shows complete integrated workflow

---

## Full Manual Test (15 minutes)

### Test 1: Verify Feedback Learning Works

#### Step 1: Check Initial State
```bash
# Check if learning file exists
ls -la ~/.jarvis/learning/feedback.json

# If it exists, view it
cat ~/.jarvis/learning/feedback.json | python -m json.tool | head -20
```

#### Step 2: Record Some Feedback
```python
# Open Python REPL
python3

from backend.core.learning.feedback_loop import (
    FeedbackLearningLoop,
    NotificationPattern,
    UserResponse,
)
from pathlib import Path
import asyncio

async def test():
    loop = FeedbackLearningLoop(storage_path=Path.home() / ".jarvis/learning/feedback.json")

    # Record 5 engagements
    for i in range(5):
        await loop.record_feedback(
            pattern=NotificationPattern.TERMINAL_ERROR,
            response=UserResponse.ENGAGED,
            notification_text=f"Test error {i+1}",
            context={"window_type": "terminal"},
            time_to_respond=2.0,
        )

    # Check if pattern is valued
    insights = loop.get_pattern_insights(NotificationPattern.TERMINAL_ERROR)
    print(f"\nPattern Insights:")
    print(f"  Total shown: {insights['total_shown']}")
    print(f"  Engagement rate: {insights['engagement_rate']:.0%}")
    print(f"  Is valued: {insights['is_valued']}")
    print(f"  Recommendation: {insights['recommendation']}")

asyncio.run(test())
```

**Expected output:**
```
Pattern Insights:
  Total shown: 5
  Engagement rate: 100%
  Is valued: True
  Recommendation: Highly valued - boost importance
```

#### Step 3: Verify Data Persisted
```bash
# Check file was updated
cat ~/.jarvis/learning/feedback.json | python -m json.tool | grep -A 5 "terminal_error"
```

**Expected:** Should show stats for terminal_error pattern with engagement_count=5

---

### Test 2: Verify Command Safety Classification

#### Run Classification Tests
```python
python3

from backend.system_control.command_safety import get_command_classifier

classifier = get_command_classifier()

# Test safe command
result = classifier.classify("ls -la")
print(f"ls -la: {result.tier.value} (expected: green)")
print(f"  Is safe: {result.is_safe}")
print(f"  Requires confirmation: {result.requires_confirmation}")
print()

# Test caution command
result = classifier.classify("npm install express")
print(f"npm install express: {result.tier.value} (expected: yellow)")
print(f"  Is safe: {result.is_safe}")
print(f"  Requires confirmation: {result.requires_confirmation}")
print()

# Test dangerous command
result = classifier.classify("rm -rf /tmp/test")
print(f"rm -rf /tmp/test: {result.tier.value} (expected: red)")
print(f"  Is destructive: {result.is_destructive}")
print(f"  Requires confirmation: {result.requires_confirmation}")
print(f"  Safer alternative: {result.suggested_alternative}")
```

**Expected output:**
```
ls -la: green (expected: green)
  Is safe: True
  Requires confirmation: False

npm install express: yellow (expected: yellow)
  Is safe: False
  Requires confirmation: True

rm -rf /tmp/test: red (expected: red)
  Is destructive: True
  Requires confirmation: True
  Safer alternative: rm -i
```

---

### Test 3: Verify Terminal Intelligence

#### Simulate Terminal Error Analysis
```python
python3

from backend.vision.handlers.terminal_command_intelligence import get_terminal_intelligence
import asyncio

async def test():
    intel = get_terminal_intelligence()

    # Simulate terminal with ModuleNotFoundError
    terminal_ocr = """
    user@host:~/project $ python app.py
    Traceback (most recent call last):
      File "app.py", line 5, in <module>
        import requests
    ModuleNotFoundError: No module named 'requests'
    user@host:~/project $
    """

    # Analyze context
    context = await intel.analyze_terminal_context(terminal_ocr)
    print(f"Last command: {context.last_command}")
    print(f"Errors found: {len(context.errors)}")
    for err in context.errors:
        print(f"  • {err}")
    print()

    # Get fix suggestions
    suggestions = await intel.suggest_fix_commands(context)
    print(f"Suggestions: {len(suggestions)}")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion.purpose}")
        print(f"   Command: {suggestion.command}")
        print(f"   Safety tier: {suggestion.safety_tier}")
        print(f"   Impact: {suggestion.estimated_impact}")

asyncio.run(test())
```

**Expected output:**
```
Last command: python app.py
Errors found: 1
  • ModuleNotFoundError: No module named 'requests'

Suggestions: 1

1. Install missing Python module 'requests'
   Command: pip install requests
   Safety tier: yellow
   Impact: Installs Python package 'requests'
```

---

## Test with Live JARVIS

### Prerequisites
1. JARVIS must be running
2. Vision system must be enabled
3. Terminal monitoring must be active

### Scenario 1: Terminal Error Detection & Fix

#### Step 1: Start JARVIS
```bash
# In terminal 1
cd /path/to/JARVIS-AI-Agent
python backend/main.py
```

#### Step 2: Trigger a Terminal Error
```bash
# In terminal 2 (make sure JARVIS can see it)
cd /tmp
python -c "import nonexistent_module"
```

**Expected JARVIS behavior:**
1. JARVIS detects the terminal error via OCR
2. Creates a pending question context
3. May proactively say: "I noticed an error in your terminal. Would you like me to analyze it?"

#### Step 3: Respond to JARVIS
```
You: "yes" or "what does it say?"
```

**Expected JARVIS response:**
```
I found a ModuleNotFoundError in your terminal.

Recommended fix:
⚠️ Install missing Python module 'nonexistent_module'

```bash
pip install nonexistent_module
```

📝 Impact: Installs Python package 'nonexistent_module'

Would you like me to help you resolve this?
```

#### Step 4: Check Feedback Was Recorded
```bash
# Check the feedback file
cat ~/.jarvis/learning/feedback.json | python -m json.tool | tail -30
```

**Expected:** Should see a new feedback event with:
- `pattern: "terminal_error"`
- `response: "engaged"` (if you said yes)
- Recent timestamp

---

### Scenario 2: Learning from Dismissals

#### Repeat Error Detection 5+ Times

1. Trigger terminal error
2. JARVIS asks if you want help
3. You say: "no" or "not now"
4. Repeat 5-7 times

#### Check Learning Effect
```python
python3

from backend.core.learning.feedback_loop import FeedbackLearningLoop, NotificationPattern
from pathlib import Path
import asyncio

async def check():
    loop = FeedbackLearningLoop(storage_path=Path.home() / ".jarvis/learning/feedback.json")
    insights = loop.get_pattern_insights(NotificationPattern.TERMINAL_ERROR)

    print(f"Terminal Error Pattern:")
    print(f"  Total shown: {insights['total_shown']}")
    print(f"  Dismissal rate: {insights['dismissal_rate']:.0%}")
    print(f"  Is ignored: {insights['is_ignored']}")

    # Check if it would suppress
    should_show, importance = loop.should_show_notification(
        pattern=NotificationPattern.TERMINAL_ERROR,
        base_importance=0.7,
    )
    print(f"\nWould show next time: {should_show}")
    print(f"Adjusted importance: {importance:.2f}")

asyncio.run(check())
```

**Expected after 5+ dismissals:**
```
Terminal Error Pattern:
  Total shown: 7
  Dismissal rate: 100%
  Is ignored: True

Would show next time: False
Adjusted importance: 0.00
```

---

### Scenario 3: Command Safety in Action

If JARVIS ever suggests running a command:

#### Monitor Safety Classification
```bash
# Watch JARVIS logs
tail -f backend/logs/jarvis.log | grep -E "(SAFETY|COMMAND)"
```

**Expected log entries when command is classified:**
```
[COMMAND-SAFETY] Classifying: pip install requests
[COMMAND-SAFETY] Tier: YELLOW, requires_confirmation: True
[TERMINAL-CMD-INTEL] Generated suggestion: pip install requests (YELLOW)
```

#### Check for Safety Warnings

If JARVIS suggests a RED tier command, you should see:
```
⚠️ Warning: This command is potentially destructive!
Impact: [description of what it will do]
```

---

## Verification Checklist

Use this to verify everything works:

### Core Functionality
- [ ] Feedback loop records events
- [ ] Data persists to `~/.jarvis/learning/feedback.json`
- [ ] Pattern stats calculate correctly
- [ ] Engagement rate tracking works
- [ ] Dismissal suppression works (after 5+ dismissals)
- [ ] Importance multiplier adjusts (boost/reduce)

### Command Safety
- [ ] GREEN commands classified correctly (ls, cat, git status)
- [ ] YELLOW commands classified correctly (npm install, git add)
- [ ] RED commands classified correctly (rm -rf, git push --force)
- [ ] Destructive patterns detected (pipes to shell, force flags)
- [ ] Safer alternatives suggested when available

### Terminal Intelligence
- [ ] Extracts commands from terminal OCR
- [ ] Detects errors (ModuleNotFoundError, etc.)
- [ ] Generates fix suggestions
- [ ] Suggestions include safety tier
- [ ] Formatting includes emojis and warnings

### Integration
- [ ] Follow-up handler uses terminal intelligence
- [ ] Safety warnings appear in suggestions
- [ ] Feedback recorded when user responds
- [ ] Learning data accessible via API (if implemented)

---

## Debugging

### Issue: No feedback file created

**Check:**
```bash
# Verify directory exists
ls -la ~/.jarvis/learning/

# Create if missing
mkdir -p ~/.jarvis/learning/

# Run test again
python -m backend.examples.live_feedback_test
```

### Issue: JARVIS doesn't detect terminal errors

**Check:**
1. Is vision system enabled?
2. Is terminal monitoring active?
3. Is OCR working?

```bash
# Check JARVIS logs
tail -f backend/logs/jarvis.log | grep -i "vision\|ocr\|terminal"
```

### Issue: Feedback not persisting

**Check file permissions:**
```bash
ls -la ~/.jarvis/learning/feedback.json
chmod 644 ~/.jarvis/learning/feedback.json
```

### Issue: Command classification seems wrong

**Test classifier directly:**
```python
from backend.system_control.command_safety import get_command_classifier

classifier = get_command_classifier()
result = classifier.classify("YOUR_COMMAND_HERE")

print(f"Tier: {result.tier.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Reasoning: {result.reasoning}")
print(f"Risk categories: {[r.value for r in result.risk_categories]}")
```

---

## Success Criteria

✅ **System is working correctly if:**

1. **Feedback loop:**
   - Events are recorded
   - File persists at `~/.jarvis/learning/feedback.json`
   - Stats update correctly
   - Patterns are suppressed after consistent dismissals

2. **Command safety:**
   - Read-only commands are GREEN
   - Modifying commands are YELLOW
   - Destructive commands are RED
   - Reasoning makes sense

3. **Terminal intelligence:**
   - Errors are detected from OCR
   - Fix suggestions are relevant
   - Safety tiers are included
   - Formatting is user-friendly

4. **Integration:**
   - Live test passes (5/6 or 6/6)
   - Demo runs without errors
   - Feedback persists across JARVIS restarts

---

## Next Steps After Testing

Once verified working:

1. **Monitor in production:**
   ```bash
   tail -f ~/.jarvis/learning/feedback.json
   ```

2. **Check learning insights:**
   ```python
   from backend.core.learning.feedback_loop import get_feedback_loop
   loop = get_feedback_loop()
   print(loop.export_learned_data())
   ```

3. **Reset if needed:**
   ```bash
   rm ~/.jarvis/learning/feedback.json
   # Or selectively reset patterns via API
   ```

4. **Integrate with Privacy Dashboard** (when ready)

---

**Test Status:** Ready to run
**Estimated Time:** 15-20 minutes for full manual test
**Automation:** Live test script handles most verification

