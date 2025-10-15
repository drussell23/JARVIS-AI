# JARVIS Vision-Multispace Intelligence: Edge Cases & Scenarios

**Version:** 1.0
**Last Updated:** 2025-10-14
**Status:** Production Reference Guide

---

## Table of Contents

1. [User Query Patterns & Scenarios](#user-query-patterns--scenarios)
2. [Edge Cases & System States](#edge-cases--system-states)
3. [Error Handling Matrix](#error-handling-matrix)
4. [Query Complexity Levels](#query-complexity-levels)
5. [Response Strategies](#response-strategies)
6. [Multi-Monitor Scenarios](#multi-monitor-scenarios)
7. [Temporal & State-Based Scenarios](#temporal--state-based-scenarios)
8. [Performance & Rate Limiting](#performance--rate-limiting)
9. [Security & Privacy Considerations](#security--privacy-considerations)

---

## 1. User Query Patterns & Scenarios

### 1.1 Direct Space Queries

**Clear & Unambiguous:**
```
✅ "What's in space 3?"
✅ "Show me space 1"
✅ "What errors are in space 5?"
✅ "Read space 2"
```

**JARVIS Response:**
- Capture specified space
- Extract text via Claude Vision
- Return OCR results
- Handle gracefully if space doesn't exist

---

### 1.2 Ambiguous/Contextual Queries

**Missing Space Number:**
```
❓ "What's on that screen?"
❓ "What's the error?"
❓ "What IDE am I using?"
❓ "What's happening?"
```

**JARVIS Strategy:**
- Assume **current/active space** (from yabai query)
- Or ask for clarification: `"Which space? (Currently on Space 2)"`
- Default to Space 1 if active space detection fails

---

**Pronoun References:**
```
User: "What's in space 3?"
JARVIS: [Returns OCR results]
User: "What about space 5?"
JARVIS: [Returns space 5 results]
User: "Compare them"
       ^^^^
       Refers to spaces 3 & 5
```

**JARVIS Strategy:**
- Track conversation context (last 2-3 spaces queried)
- Resolve pronouns ("it", "that", "them") to spaces
- Maintain session state in memory

---

### 1.3 Multi-Space Queries

**Comparison:**
```
❓ "Compare space 3 and space 5"
❓ "Which space has the error?"
❓ "Find the terminal across all spaces"
❓ "What's different between space 1 and space 2?"
```

**JARVIS Strategy:**
- Capture all specified spaces in parallel
- Run Claude Vision analysis on each
- Synthesize comparison results
- Return unified response

**Example Response:**
```
Space 3: VS Code, Python file with TypeError on line 421
Space 5: Browser showing documentation
Difference: Space 3 has an active error, Space 5 is reference material
```

---

### 1.4 Temporal Queries

**Time-Based:**
```
❓ "What changed in space 3?"
❓ "Has the error been fixed?"
❓ "What's new in the last 5 minutes?"
❓ "When did this error first appear?"
```

**Current Limitation:**
- ❌ No time-series tracking (v1.0)
- ⚠️ Returns: "Temporal tracking not yet implemented"

**Future (v2.0):**
- ✅ Cache screenshots with timestamps
- ✅ Diff images to detect changes
- ✅ Track error appearance/resolution

---

### 1.5 Predictive/Analytical Queries

**High-Level Analysis:**
```
❓ "Am I making progress?"
❓ "What should I work on next?"
❓ "Are there any potential bugs?"
❓ "Explain what this code does"
```

**JARVIS Strategy:**
- Capture relevant spaces
- Use Claude Vision with analytical prompts
- Provide high-level insights
- Require semantic understanding (v2.0 feature)

---

### 1.6 Action-Oriented Queries

**Requests for Action:**
```
❓ "Fix the error in space 3"
❓ "Switch to space 5"
❓ "Close the browser in space 2"
❓ "Run the tests"
```

**JARVIS Strategy (v1.0):**
- Vision is **read-only**
- Return: `"I can see [error], but cannot execute actions yet"`
- Suggest manual steps

**JARVIS Strategy (v2.0):**
- Integrate with action APIs (yabai, AppleScript)
- Execute safe commands with user confirmation
- Autonomous execution for trusted actions

---

## 2. Edge Cases & System States

### 2.1 Space-Related Edge Cases

| Edge Case | Detection | JARVIS Response |
|-----------|-----------|-----------------|
| **Space doesn't exist** | `yabai -m query --spaces` returns no match | `"Space 10 doesn't exist. You have 6 spaces."` |
| **Empty space** | No windows in space | `"Space 3 is empty (no windows)."` |
| **Space with only minimized windows** | All windows minimized | `"Space 4 has minimized windows only. Cannot capture."` |
| **Space mid-transition** | User switching spaces during capture | Retry with 500ms delay |
| **Fullscreen app** | Single fullscreen window | Capture works normally |
| **Split view** | Multiple windows side-by-side | Capture entire space (both windows) |

---

### 2.2 Window Capture Edge Cases

| Edge Case | Cause | Handling |
|-----------|-------|----------|
| **Invalid window ID** | Window closed mid-capture | Fallback to next window in space |
| **Permission denied** | Screen recording disabled | `"Enable Screen Recording in System Settings > Privacy"` |
| **Window off-screen** | Window partially/fully outside display bounds | CoreGraphics clips to visible area |
| **Transparent windows** | Overlay/HUD windows | Capture underlying content |
| **4K/5K displays** | Very large screenshots | Resize to 2560px max width before sending to Claude |

---

### 2.3 System State Edge Cases

| State | Detection | Response |
|-------|-----------|----------|
| **Yabai not running** | `yabai -m query` fails | `"Yabai not detected. Install: brew install koekeishiya/formulae/yabai"` |
| **Yabai crashed** | Command hangs/timeout | Restart yabai: `brew services restart yabai` |
| **Display sleep** | Screen off, no capture possible | `"Display is sleeping. Wake to use vision."` |
| **Screen locked** | Login screen active | `"Screen is locked. Unlock to capture."` |
| **No displays** | Headless/SSH session | `"No displays detected. Vision requires GUI session."` |

---

### 2.4 API & Network Edge Cases

| Edge Case | Cause | Fallback Strategy |
|-----------|-------|-------------------|
| **Claude API timeout** | Network issues | Retry 3x with exponential backoff (1s, 2s, 4s) |
| **Rate limit (429)** | Too many requests | Wait & retry, use cached results if available |
| **Invalid API key** | Expired/wrong key | `"Claude API key invalid. Check .env"` |
| **Image too large** | Screenshot >5MB | Resize to max 2560px width, compress to JPEG 85% |
| **Network offline** | No internet | `"Offline. Vision requires internet for Claude API."` |

---

## 3. Error Handling Matrix

### 3.1 Graceful Degradation Strategy

```
Priority 1: Try primary method
   ↓ (fails)
Priority 2: Try fallback method
   ↓ (fails)
Priority 3: Return partial results + warning
   ↓ (fails)
Priority 4: Return user-friendly error message
```

---

### 3.2 Capture Fallbacks

```python
# Primary: Capture specific window
try:
    capture_window(window_id)
except:
    # Fallback 1: Capture entire space
    try:
        capture_space(space_id)
    except:
        # Fallback 2: Use cached screenshot (if <60s old)
        try:
            use_cached_screenshot(space_id)
        except:
            # Fallback 3: Return error
            return "Unable to capture Space {space_id}"
```

---

### 3.3 OCR Fallbacks

```python
# Primary: Claude Vision API
try:
    claude_vision_ocr(image)
except RateLimitError:
    # Fallback 1: Use cached OCR (if <5min old)
    return cached_ocr_results(image_hash)
except NetworkError:
    # Fallback 2: Local OCR (Tesseract)
    return tesseract_ocr(image)
except:
    # Fallback 3: Return image metadata only
    return f"Image: {width}x{height}, {window_title}"
```

---

## 4. Query Complexity Levels

### Level 1: Simple (Single Space, Single Question)

**Examples:**
- "What's in space 3?"
- "Show me space 1"

**Processing:**
1. Parse space number
2. Capture space
3. Run OCR
4. Return results

**Latency:** 2-4s
**API Calls:** 1 (Claude Vision)

---

### Level 2: Medium (Multiple Spaces or Context)

**Examples:**
- "Compare space 3 and space 5"
- "Which space has the terminal?"

**Processing:**
1. Parse multiple spaces
2. Capture in parallel
3. Run OCR on each
4. Synthesize comparison

**Latency:** 3-6s
**API Calls:** 2-6 (depending on spaces)

---

### Level 3: Complex (Temporal, Predictive, Cross-Space)

**Examples:**
- "What changed in the last 5 minutes?"
- "Find all errors across all spaces"
- "Am I making progress?"

**Processing:**
1. Query all spaces (1-10+)
2. Capture each
3. Run OCR + analysis
4. Apply temporal/semantic logic
5. Synthesize high-level answer

**Latency:** 10-30s
**API Calls:** 5-15+
**Requires:** v2.0 features (caching, session memory)

---

## 5. Response Strategies

### 5.1 Clear & Actionable

**Good:**
```
✅ "Space 3 has a TypeError on line 421 in test_vision.py.
   The error is: 'NoneType' object has no attribute 'get'"
```

**Bad:**
```
❌ "There's an error."
❌ "I see some text in a code editor."
```

---

### 5.2 Context-Aware

**User Context Matters:**

```
Query: "What's the error?"
Context: User just asked about Space 3

Response: "The error in Space 3 is a TypeError on line 421."
          (No need to re-ask which space)
```

---

### 5.3 Proactive Suggestions

**Offer Next Steps:**

```
Query: "What's in space 5?"
Response: "Space 5 shows Chrome with error documentation for NoneType.
           Would you like me to compare this with the error in Space 3?"
```

---

### 5.4 Confidence Levels

**Express Uncertainty:**

```
High Confidence:
✅ "Space 3 has 15 visible lines of Python code."

Medium Confidence:
⚠️ "Space 3 appears to have a syntax error, though the text is partially obscured."

Low Confidence:
❓ "Space 3 may contain a terminal, but the resolution is too low to confirm."
```

---

## 6. Multi-Monitor Scenarios

### 6.1 Current Limitation (v1.0)

- ❌ Assumes single display
- ❌ Doesn't map spaces to monitors
- ❌ Can't distinguish "left monitor" vs "right monitor"

---

### 6.2 User Queries (Multi-Monitor)

```
❓ "What's on my second monitor?"
❓ "Show me all my displays"
❓ "Which monitor has the terminal?"
❓ "Move space 3 to the left monitor"
```

**v1.0 Response:**
```
"Multi-monitor detection not yet supported.
 I can see Space 3, but cannot identify which monitor it's on."
```

---

### 6.3 v2.0 Multi-Monitor Support

**Implementation:**

```python
# Detect all displays
displays = CGGetActiveDisplayList()
# Returns: [Display1, Display2, Display3]

# Map spaces to displays
for space in yabai_spaces:
    space['display_id'] = get_display_for_space(space['id'])

# Query: "What's on my left monitor?"
left_display = get_display_by_position('left')
spaces_on_left = [s for s in spaces if s['display_id'] == left_display]
```

**New Capabilities:**
- "What's on my right monitor?" → Capture all spaces on right display
- "Move this to my main monitor" → Yabai move command
- "Compare left and right monitors" → Multi-space capture & comparison

---

## 7. Temporal & State-Based Scenarios

### 7.1 Change Detection

**User Queries:**
```
❓ "What changed since I last asked?"
❓ "Did the error get fixed?"
❓ "Has the build finished?"
```

**v2.0 Implementation:**

```python
# Cache screenshots with timestamps
cache = {
    'space_3': {
        'screenshot': image_bytes,
        'timestamp': 1697234567,
        'hash': 'abc123...'
    }
}

# Detect changes
new_screenshot = capture_space(3)
if new_screenshot.hash != cache['space_3']['hash']:
    diff = image_diff(cache['space_3']['screenshot'], new_screenshot)
    return f"Changed: {diff.description}"
else:
    return "No changes detected."
```

---

### 7.2 Proactive Monitoring

**Autonomous Alerts:**

```python
# Every 10 seconds
while monitoring:
    for space in critical_spaces:
        screenshot = capture_space(space)

        # Detect new errors
        if "error" in ocr_text and "error" not in last_ocr_text:
            alert_user(f"New error detected in Space {space}")

        # Detect build completion
        if "Build succeeded" in ocr_text:
            alert_user(f"Build completed in Space {space}")
```

**User Experience:**
```
[JARVIS, unprompted]: "Sir, a new error appeared in Space 3, line 422."
```

---

### 7.3 Session Memory

**Cross-Session Learning:**

```python
# Session 1 (Monday)
User: "What's the error in space 3?"
JARVIS: "TypeError on line 421"
User: "I fixed it by adding a null check."
# Store: error_type="TypeError", solution="null check"

# Session 2 (Wednesday)
User: "What's the error in space 5?"
JARVIS: "TypeError on line 89. Similar to Monday's error.
         You fixed that by adding a null check. Try that here?"
```

---

## 8. Performance & Rate Limiting

### 8.1 Claude API Limits

| Tier | RPM | TPM | Daily Limit |
|------|-----|-----|-------------|
| Free | 5 | 20k | 1000 requests |
| Pro | 50 | 100k | 10k requests |
| Team | 100 | 200k | 50k requests |

---

### 8.2 Cost Optimization

**Without Caching:**
- 10 queries/session × 1 image/query = 10 API calls
- ~$0.10/call = **$1.00/session**

**With Smart Caching (v2.0):**
- 10 queries/session × 30% cache hit rate = 7 API calls
- ~$0.10/call = **$0.70/session** (30% savings)

**With Aggressive Caching:**
- 10 queries/session × 60% cache hit rate = 4 API calls
- ~$0.10/call = **$0.40/session** (60% savings)

---

### 8.3 Rate Limit Handling

**Strategy:**

```python
import time
from functools import wraps

def rate_limited(max_per_minute=50):
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed

            if wait_time > 0:
                time.sleep(wait_time)

            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limited(max_per_minute=50)
def call_claude_vision(image):
    # API call here
    pass
```

---

### 8.4 Parallelization

**Sequential (Slow):**
```python
# 4 spaces × 3s/space = 12 seconds total
for space in [1, 2, 3, 4]:
    capture_and_ocr(space)
```

**Parallel (Fast):**
```python
# 4 spaces in parallel = 3 seconds total
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(capture_and_ocr, s) for s in [1, 2, 3, 4]]
    results = [f.result() for f in futures]
```

**Speedup:** 4x faster for multi-space queries

---

## 9. Security & Privacy Considerations

### 9.1 Sensitive Data in Screenshots

**Potential Exposure:**
- Passwords visible in terminals
- API keys in .env files
- Personal messages in Slack/email
- Financial data in spreadsheets
- Health records in browser

---

### 9.2 Mitigation Strategies

**1. Redaction (v2.0):**
```python
def redact_sensitive_data(image):
    # OCR to find patterns
    text = ocr(image)

    # Detect sensitive patterns
    patterns = [
        r'password\s*[:=]\s*\S+',  # Passwords
        r'sk-[a-zA-Z0-9]{32,}',     # OpenAI keys
        r'\d{4}-\d{4}-\d{4}-\d{4}', # Credit cards
    ]

    # Black out matches
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            bbox = get_bounding_box(match)
            image = black_out_region(image, bbox)

    return image
```

**2. User Consent:**
```python
# First time capturing a space
if not user_consented(space_id):
    preview = capture_space(space_id)
    show_preview(preview)

    consent = ask_user(f"Allow JARVIS to read Space {space_id}?")
    if consent:
        store_consent(space_id)
    else:
        return "User denied consent for Space {space_id}"
```

**3. Local-Only Mode:**
```python
# .env
JARVIS_VISION_MODE=local  # Never send to Claude API

# Use local OCR (Tesseract) instead
def ocr_image(image):
    if os.getenv('JARVIS_VISION_MODE') == 'local':
        return tesseract.image_to_string(image)
    else:
        return claude_vision_api(image)
```

---

### 9.3 Data Retention

**Current (v1.0):**
- Screenshots stored temporarily in `/tmp/jarvis_vision/`
- Deleted after processing
- No persistent storage

**Future (v2.0 with caching):**
- Screenshots cached for 30-60 seconds
- OCR results cached for 5 minutes
- Automatic expiration/cleanup
- Option to disable caching: `JARVIS_VISION_CACHE=false`

---

### 9.4 Audit Logging

**Track Vision Usage:**

```python
# Log every capture
import logging

logging.info({
    'timestamp': '2025-10-14T15:30:42Z',
    'space_id': 3,
    'query': 'What errors are visible?',
    'user': 'derek',
    'api_call': True,
    'cache_hit': False
})
```

**Benefits:**
- Debugging
- Cost tracking
- Security audits
- Privacy compliance

---

## Summary: Edge Case Coverage

### ✅ Well Handled (v1.0)
- Single space queries
- Basic error detection
- Simple OCR
- Yabai integration
- Permission errors

### ⚠️ Partially Handled
- Multi-space queries (works but slow)
- Rate limiting (manual backoff)
- Large images (resize but not optimal)

### ❌ Not Yet Handled (Requires v2.0)
- Multi-monitor detection
- Temporal tracking
- Change detection
- Session memory
- Predictive analysis
- Sensitive data redaction
- Proactive monitoring
- Autonomous actions

---

## Next Steps

1. **Read this document** before handling user queries
2. **Reference edge case matrix** when encountering errors
3. **Implement missing features** from roadmap
4. **Update this doc** as new edge cases are discovered

---

**Document Maintainer:** Derek Russell
**JARVIS Version:** 1.0 (vision-multispace-improvements branch)
**Last Test Date:** 2025-10-14
