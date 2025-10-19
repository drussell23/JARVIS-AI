# Edge Case Handling System - Complete Documentation

**Version:** 1.2
**Last Updated:** 2025-10-19
**Status:** ✅ Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Space-Related Edge Cases](#space-related-edge-cases)
3. [Window Capture Edge Cases](#window-capture-edge-cases)
4. [System State Edge Cases](#system-state-edge-cases)
5. [API & Network Edge Cases](#api--network-edge-cases)
6. [Integration Points](#integration-points)
7. [Usage Examples](#usage-examples)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The Edge Case Handling System provides comprehensive, robust handling for all macOS space, window capture, system state, and API/network edge cases. It ensures JARVIS can gracefully handle failures, provide helpful error messages, and automatically retry or fallback when needed.

### Key Features

✅ **Fully Async** - All operations use `asyncio` for non-blocking execution
✅ **Dynamic** - No hardcoded space IDs, window IDs, or image sizes
✅ **Robust** - Retry logic with exponential backoff and automatic fallback
✅ **Natural Language Responses** - User-friendly error messages
✅ **Comprehensive Metadata** - Detailed information about what happened
✅ **Zero Dependencies** - Uses native macOS tools (yabai, screencapture, sips)
✅ **Auto-Recovery** - Automatic service restart and recovery attempts
✅ **System Health Monitoring** - Continuous health checks for critical services
✅ **Network Detection** - Real-time network connectivity monitoring
✅ **Image Optimization** - Automatic image resizing and compression for API limits
✅ **Circuit Breaker** - Prevents API overload with intelligent request throttling

### Architecture

```
User Request
    ↓
Intent Analyzer
    ↓
┌──────────────────────────────────────────────────┐
│  Edge Case Validation                            │
│  ├── SystemStateManager (system health)          │
│  ├── SpaceStateManager (space validation)        │
│  ├── WindowCaptureManager (window capture)       │
│  └── APINetworkManager (API/network readiness)   │
└──────────────────────────────────────────────────┘
    ↓
Action Execution / Vision Processing
    ↓
Response with Metadata
```

---

## Space-Related Edge Cases

### SpaceStateManager

**Location:** `backend/context_intelligence/managers/space_state_manager.py`

Handles all space-related validation and edge cases before operations.

### Supported Edge Cases

| Edge Case | Detection | JARVIS Response |
|-----------|-----------|-----------------|
| **Space doesn't exist** | `yabai -m query --spaces` returns no match | `"Space 10 doesn't exist. You have 6 spaces."` |
| **Empty space** | No windows in space | `"Space 3 is empty (no windows)."` |
| **Minimized-only windows** | All windows minimized | `"Space 4 has 2 minimized window(s) only (Safari, Terminal). Cannot capture."` |
| **Space mid-transition** | User switching spaces during capture | Retry with 500ms delay (exponential backoff) |
| **Fullscreen app** | Single fullscreen window | `"Space 5 has Chrome in fullscreen. Capture will work normally."` |
| **Split view** | Multiple windows side-by-side | `"Space 2 has split view (VSCode and Terminal). Entire space will be captured."` |

### Components

#### 1. SpaceValidator

Validates space existence and properties.

```python
from context_intelligence.managers import get_space_state_manager

manager = get_space_state_manager()

# Validate space exists
exists, max_space_id = await manager.validator.validate_space_exists(space_id=10)
if not exists:
    print(f"Space 10 doesn't exist. You have {max_space_id} spaces.")
```

#### 2. SpaceTransitionHandler

Handles retry logic for transient states.

```python
# Wait for space to stabilize
stable = await manager.transition_handler.wait_for_stable_state(
    space_id=3,
    validator=manager.validator,
    timeout=5.0
)
```

#### 3. SpaceStateManager (Main)

Main coordinator handling all edge cases.

```python
# Get comprehensive space state
state_info = await manager.get_space_state(space_id=3)

print(f"Space {state_info.space_id}:")
print(f"  State: {state_info.state.value}")
print(f"  Windows: {state_info.window_count}")
print(f"  Visible: {state_info.visible_window_count}")
print(f"  Minimized: {state_info.minimized_window_count}")
print(f"  Apps: {', '.join(state_info.applications)}")
```

### Integration Example

**In action_executor.py:**

```python
# Before executing yabai command
if self.validate_spaces and self.space_manager:
    space_id = self._extract_space_id(step.command)
    if space_id is not None:
        edge_case_result = await self.space_manager.handle_edge_case(space_id)

        if edge_case_result.edge_case == "not_exist":
            return StepResult(
                success=False,
                error=edge_case_result.message  # "Space 10 doesn't exist. You have 6 spaces."
            )
```

---

## Window Capture Edge Cases

### WindowCaptureManager

**Location:** `backend/context_intelligence/managers/window_capture_manager.py`

Handles all window capture failures and edge cases with automatic retry and fallback.

### Supported Edge Cases

| Edge Case | Cause | Handling |
|-----------|-------|----------|
| **Invalid window ID** | Window closed mid-capture | Fallback to next window in space |
| **Permission denied** | Screen recording disabled | `"Enable Screen Recording in System Settings > Privacy & Security > Screen Recording"` |
| **Window off-screen** | Window partially/fully outside display bounds | CoreGraphics clips to visible area |
| **Transparent windows** | Overlay/HUD windows | Capture underlying content, flag in metadata |
| **4K/5K displays** | Very large screenshots | Resize to 2560px max width before sending to Claude |

### Components

#### 1. PermissionChecker

Checks macOS screen recording permissions.

```python
from context_intelligence.managers import get_window_capture_manager

manager = get_window_capture_manager()

# Check permissions (cached for 60s)
has_permission, message = await manager.permission_checker.check_screen_recording_permission()

if not has_permission:
    print(message)  # "Enable Screen Recording in System Settings..."
```

#### 2. WindowValidator

Validates window state before capture.

```python
# Validate window
is_valid, window_info, message = await manager.window_validator.validate_window(window_id=12345)

if is_valid:
    print(f"Window: {window_info.app} - {window_info.title}")
    print(f"State: {window_info.state.value}")
    print(f"Bounds: {window_info.bounds.width}x{window_info.bounds.height}")
    print(f"On screen: {window_info.bounds.is_on_screen}")
    print(f"Visible area: {window_info.bounds.visible_area_ratio * 100:.1f}%")
```

#### 3. ImageProcessor

Processes captured images for edge cases.

```python
# Process image (resize if needed)
success, processed_path, original_size, final_size = await manager.image_processor.process_image(
    image_path="/tmp/screenshot.png"
)

if original_size[0] > 2560:
    print(f"Resized from {original_size[0]}x{original_size[1]} to {final_size[0]}x{final_size[1]}")
```

#### 4. CaptureRetryHandler

Retry logic with fallback windows.

```python
# Retry with fallback
result, used_window_id = await manager.retry_handler.retry_with_fallback(
    capture_func=some_capture_function,
    window_id=12345,
    fallback_windows=[12346, 12347, 12348]
)

if result.success and result.fallback_window_id:
    print(f"Used fallback window {result.fallback_window_id}")
```

#### 5. WindowCaptureManager (Main)

Main coordinator for robust window capture.

```python
# Capture window with full edge case handling
result = await manager.capture_window(
    window_id=12345,
    space_id=3,  # For fallback window discovery
    use_fallback=True
)

if result.success:
    print(f"✅ Captured: {result.image_path}")
    print(f"   Status: {result.status.value}")
    print(f"   Original: {result.original_size}")
    print(f"   Final: {result.resized_size}")
    print(f"   Message: {result.message}")

    # Check metadata
    if result.metadata.get("off_screen"):
        print("   ⚠️ Window was partially off-screen")
    if result.metadata.get("transparent"):
        print("   ⚠️ Window has transparency")
    if result.metadata.get("resized"):
        print("   ⚠️ Image was resized (4K/5K display)")
else:
    print(f"❌ Failed: {result.error}")
```

---

## System State Edge Cases

### SystemStateManager

**Location:** `backend/context_intelligence/managers/system_state_manager.py`

Handles all system-level health checks and edge cases before operations.

### Supported Edge Cases

| State | Detection | Response |
|-------|-----------|----------|
| **Yabai not running** | `yabai -m query` fails | `"Yabai not detected. Install: brew install koekeishiya/formulae/yabai"` |
| **Yabai crashed** | Command hangs/timeout | `"Yabai crashed or hung. Restart: brew services restart yabai"` |
| **Display sleep** | Screen off, no capture possible | `"Display is sleeping. Wake to use vision."` |
| **Screen locked** | Login screen active | `"Screen is locked. Unlock to capture."` |
| **No displays** | Headless/SSH session | `"No displays detected. Vision requires GUI session."` |

### Components

#### 1. YabaiHealthChecker

Monitors yabai service health and detects issues.

```python
from context_intelligence.managers import get_system_state_manager

manager = get_system_state_manager()

# Check yabai status
yabai_status = await manager.yabai_checker.check_yabai_status()

if yabai_status.state.value == "not_installed":
    print(yabai_status.message)  # "Yabai not detected. Install: brew install..."
    print(f"Recovery: {yabai_status.recovery_command}")
elif yabai_status.state.value == "timeout":
    print(yabai_status.message)  # "Yabai crashed or hung. Restart: brew services..."
    print(f"Can recover: {yabai_status.can_recover}")
```

#### 2. DisplayStateDetector

Detects display state (awake, sleeping, locked, headless).

```python
# Check display state
display_status = await manager.display_detector.check_display_state()

if display_status.state.value == "locked":
    print(display_status.message)  # "Screen is locked. Unlock to capture."
elif display_status.state.value == "sleeping":
    print(display_status.message)  # "Display is sleeping. Wake to use vision."
elif display_status.state.value == "no_displays":
    print(display_status.message)  # "No displays detected. Vision requires GUI session."
    print(f"Headless: {display_status.is_headless}")
```

#### 3. SystemRecoveryHandler

Attempts automatic recovery from system state issues.

```python
# Enable auto-recovery
manager = initialize_system_state_manager(auto_recover=True)

# Check system state (will auto-recover if possible)
state_info = await manager.check_system_state()

if state_info.health.value == "healthy":
    print("✅ System healthy!")
else:
    print(f"System: {state_info.health.value}")
    for suggestion in state_info.recovery_suggestions:
        print(f"  - {suggestion}")
```

#### 4. SystemStateManager (Main)

Main coordinator for system health monitoring.

```python
# Comprehensive system check
state_info = await manager.check_system_state()

print(f"Health: {state_info.health.value}")
print(f"Can use vision: {state_info.can_use_vision}")
print(f"Can use spaces: {state_info.can_use_spaces}")

print("\nChecks passed:")
for check in state_info.checks_passed:
    print(f"  ✅ {check}")

print("\nChecks failed:")
for check in state_info.checks_failed:
    print(f"  ❌ {check}")

print("\nWarnings:")
for warning in state_info.warnings:
    print(f"  ⚠️ {warning}")
```

### Auto-Recovery Example

```python
# Initialize with auto-recovery enabled
manager = initialize_system_state_manager(
    auto_recover=True,  # Enable automatic recovery
    yabai_timeout=5.0,   # Timeout for yabai commands
    cache_ttl=5.0        # Cache TTL for health checks
)

# Check system state - will attempt recovery if needed
state_info = await manager.check_system_state()

if state_info.yabai_status.state.value == "running":
    print("✅ Yabai running (auto-recovered if needed)")
else:
    print(f"Yabai status: {state_info.yabai_status.state.value}")
    if state_info.yabai_status.can_recover:
        print(f"Can recover with: {state_info.yabai_status.recovery_command}")
```

### Wait for Healthy State

```python
# Wait for system to become healthy (useful after recovery attempts)
became_healthy, final_state = await manager.wait_for_healthy_state(
    timeout=30.0,         # Maximum time to wait
    check_interval=2.0    # Time between checks
)

if became_healthy:
    print("✅ System is now healthy!")
else:
    print(f"❌ System did not become healthy: {final_state.health.value}")
    for failure in final_state.checks_failed:
        print(f"  - {failure}")
```

### Integration Example

**In multi_space_capture_engine.py:**

```python
# Check system health before capture
is_healthy, health_message, state_info = await engine.check_system_health()

if not is_healthy:
    logger.error(f"System health check failed: {health_message}")
    return SpaceCaptureResult(
        screenshots={},
        metadata={},
        success=False,
        errors={-1: health_message}  # System-level error
    )
```

**In action_executor.py:**

```python
# Check system health before yabai command
if self.check_system_health and self.system_state_manager:
    system_state = await self.system_state_manager.check_system_state()

    if not system_state.can_use_spaces:
        return StepResult(
            success=False,
            error=system_state.yabai_status.message  # Helpful error message
        )
```

---

## API & Network Edge Cases

### APINetworkManager

**Location:** `backend/context_intelligence/managers/api_network_manager.py`

Handles all Claude API and network-related edge cases before making API calls.

### Supported Edge Cases

| Edge Case | Cause | Handling |
|-----------|-------|----------|
| **Claude API timeout** | Network issues, slow response | Retry 3x with exponential backoff (1s, 2s, 4s) |
| **Rate limit (429)** | Too many requests | Wait & retry, use cached results if available |
| **Invalid API key** | Expired/wrong key | `"Claude API key invalid. Check .env"` |
| **Image too large** | Screenshot >5MB | Resize to max 2560px width, compress to JPEG 85% |
| **Network offline** | No internet | `"Offline. Vision requires internet for Claude API."` |

### Components

#### 1. APIHealthChecker

Monitors Claude API health and detects issues.

```python
from context_intelligence.managers import get_api_network_manager

manager = get_api_network_manager()

# Check API status
api_status = await manager.api_health_checker.check_api_status()

if api_status.state.value == "invalid_key":
    print(api_status.message)  # "Claude API key invalid. Check .env file."
elif api_status.state.value == "rate_limited":
    print(f"Rate limited. Wait {api_status.retry_after_seconds}s")
elif api_status.state.value == "available":
    print("API ready for calls")
```

**Features:**
- API key format validation
- Rate limit detection (429 responses)
- Circuit breaker pattern (opens after 5 consecutive failures)
- Automatic rate limit tracking

#### 2. NetworkDetector

Detects network connectivity in real-time.

```python
# Check network status
network_status = await manager.network_detector.check_network_status()

if network_status.state.value == "offline":
    print(network_status.message)  # "Offline. Vision requires internet for Claude API."
elif network_status.state.value == "online":
    print(f"Online (latency: {network_status.latency_ms:.1f}ms)")
elif network_status.state.value == "degraded":
    print(f"Slow connection (latency: {network_status.latency_ms:.1f}ms)")
```

**Features:**
- Ping-based connectivity test (Cloudflare DNS 1.1.1.1)
- Latency measurement
- Connection quality assessment (online vs degraded)
- 5-second cache for status checks

#### 3. ImageOptimizer

Optimizes images for Claude API size limits.

```python
# Optimize image before sending to API
opt_result = await manager.image_optimizer.optimize_image(
    image_path="/tmp/screenshot.png"
)

if opt_result.success:
    print(f"✅ {opt_result.message}")
    print(f"   Original: {opt_result.original_size_bytes // 1024}KB")
    print(f"   Optimized: {opt_result.optimized_size_bytes // 1024}KB")
    print(f"   Reduction: {opt_result.size_reduction_percent:.1f}%")

    # Use optimized image
    image_to_send = opt_result.optimized_path
```

**Features:**
- Automatic resize to max 2560px width (configurable)
- JPEG compression at 85% quality (configurable)
- PNG → JPEG conversion for smaller size
- 5MB size limit enforcement
- Uses native macOS `sips` tool (no PIL dependency)

#### 4. RetryHandler

Handles retry logic with exponential backoff.

```python
# Execute API call with retry
async def make_api_call():
    return await client.messages.create(...)

retry_result = await manager.retry_handler.retry_with_backoff(
    make_api_call,
    cache_key="analysis_123"  # Optional caching
)

if retry_result.success:
    print(f"✅ Success after {retry_result.attempts} attempt(s)")
    print(f"   Total delay: {retry_result.total_delay:.1f}s")
    result = retry_result.result
else:
    print(f"❌ Failed after {retry_result.attempts} attempts")
    print(f"   Error: {retry_result.final_error}")
```

**Features:**
- Exponential backoff (1s, 2s, 4s, 8s, ...)
- Configurable max retries (default: 3)
- Result caching with TTL (default: 5 minutes)
- Automatic cache key generation

#### 5. APINetworkManager (Main)

Main coordinator for all API/network edge cases.

```python
from context_intelligence.managers import initialize_api_network_manager

# Initialize manager
manager = initialize_api_network_manager(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_retries=3,
    initial_retry_delay=1.0,
    max_image_width=2560,
    max_image_size_mb=5.0
)

# Check readiness before API call
is_ready, message, status_info = await manager.check_ready_for_api_call()

if not is_ready:
    print(f"❌ Not ready: {message}")
    # Handle specific issues
    if "network" in status_info:
        print(f"   Network: {status_info['network'].state.value}")
    if "api" in status_info:
        print(f"   API: {status_info['api'].state.value}")
else:
    print("✅ Ready for API call")
    # Proceed with API call
```

**Comprehensive API call with all edge cases:**

```python
# Execute API call with full edge case handling
async def my_api_call(prompt, image_path):
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "path", "path": image_path}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    return response

# Let manager handle all edge cases
result = await manager.execute_api_call_with_retry(
    my_api_call,
    prompt="Analyze this screenshot",
    optimize_image="/tmp/large_screenshot.png",  # Will optimize before call
    cache_key="screenshot_analysis_123"  # Cache successful result
)

if result.success:
    print(f"✅ API call succeeded")
    print(f"   Attempts: {result.attempts}")
    print(f"   Total delay: {result.total_delay:.1f}s")
    response = result.result
else:
    print(f"❌ API call failed: {result.final_error}")
    # Helpful error message with specific issue
```

### Integration Examples

**In claude_streamer.py:**

```python
# Check API/Network readiness before streaming
if self._api_network_manager:
    is_ready, message, status_info = await self._api_network_manager.check_ready_for_api_call()

    if not is_ready:
        # Return helpful error to user
        yield f"\n⚠️  {message}\n"
        return

    # Proceed with streaming
    async for chunk in self._stream_with_model(...):
        yield chunk
```

**In claude_vision_analyzer_main.py:**

```python
# Initialize with API/Network manager
self.api_network_manager = initialize_api_network_manager(
    api_key=api_key,
    max_image_width=self.config.max_image_dimension,
    max_image_size_mb=5.0
)

# Before making vision API call
if self.api_network_manager:
    # Optimize image first
    opt_result = await self.api_network_manager.image_optimizer.optimize_image(image_path)

    if opt_result.success:
        # Use optimized image
        image_to_send = opt_result.optimized_path
```

### Wait for Ready State

```python
# Wait for system to become ready (e.g., after network outage)
became_ready, message = await manager.wait_for_ready(timeout=60.0)

if became_ready:
    print("✅ System is now ready for API calls")
    # Proceed with API calls
else:
    print(f"❌ Timeout: {message}")
    # Still not ready after 60s
```

### Circuit Breaker Example

```python
# Circuit breaker prevents overload after failures
for i in range(10):
    api_status = await manager.api_health_checker.check_api_status()

    if api_status.state.value == "unavailable" and api_status.metadata.get("circuit_breaker") == "open":
        print(f"Circuit breaker open. Wait {api_status.retry_after_seconds}s")
        await asyncio.sleep(api_status.retry_after_seconds)
        continue

    # Make API call
    try:
        result = await make_api_call()
        manager.api_health_checker.record_success()  # Reset circuit breaker
    except Exception as e:
        manager.api_health_checker.record_failure()  # Increment failure count
```

---

## Integration Points

### 1. Temporal Query Handler

**File:** `backend/context_intelligence/handlers/temporal_query_handler.py`

**Integration:** ScreenshotManager now uses WindowCaptureManager

```python
# Before (old)
screenshot = pyautogui.screenshot()

# After (new)
screenshot = await self.window_capture_manager.capture_window(
    window_id=window_id,
    space_id=space_id,
    use_fallback=True
)
```

**Benefits:**
- Permission checking before capture
- 4K/5K automatic resizing
- Window fallback when primary fails
- Off-screen window handling

### 2. Multi-Space Capture Engine

**File:** `backend/vision/multi_space_capture_engine.py`

**Integration:** `_capture_with_cg_windows()` uses WindowCaptureManager first

```python
# Try WindowCaptureManager first
if WINDOW_CAPTURE_AVAILABLE:
    capture_manager = get_window_capture_manager()
    capture_result = await capture_manager.capture_window(
        window_id=window_id,
        space_id=space_id,
        use_fallback=True
    )

    if capture_result.success:
        img = Image.open(capture_result.image_path)
        screenshot = np.array(img)
        return screenshot

# Fallback to CGWindowCapture
screenshot = CGWindowCapture.capture_window_by_id(window_id)
```

**Benefits:**
- Robust window validation
- Automatic retry with exponential backoff
- Edge case logging for debugging
- Metadata about capture quality

### 3. Reliable Screenshot Capture

**File:** `backend/vision/reliable_screenshot_capture.py`

**Integration:** Added as first method in fallback hierarchy

```python
# Method priority
methods = [
    ('window_capture_manager', ...),  # ← NEW (first choice)
    ('quartz_composite', ...),
    ('quartz_windows', ...),
    ('appkit_screen', ...),
    ('screencapture_cli', ...),
    ('window_server', ...)
]
```

**Benefits:**
- WindowCaptureManager as first choice
- Graceful degradation to other methods
- Comprehensive error handling
- Metadata about edge cases encountered

### 4. Action Executor

**File:** `backend/context_intelligence/executors/action_executor.py`

**Integration:** Validates spaces before yabai commands

```python
# Validate space before execution
if self.validate_spaces and self.space_manager:
    space_id = self._extract_space_id(step.command)
    if space_id is not None:
        edge_case_result = await self.space_manager.handle_edge_case(space_id)

        # Handle edge cases
        if edge_case_result.edge_case == "not_exist":
            return StepResult(success=False, error=edge_case_result.message)
```

### 5. Multi-Space Query Handler

**File:** `backend/context_intelligence/handlers/multi_space_query_handler.py`

**Integration:** Validates spaces before parallel analysis

```python
# Validate space before analysis
edge_case_result = await self.space_manager.handle_edge_case(space_id)

if edge_case_result.edge_case == "not_exist":
    return SpaceAnalysisResult(
        space_id=space_id,
        success=False,
        content_summary=edge_case_result.message
    )
```

---

## Usage Examples

### Example 1: Capture Window with All Edge Case Handling

```python
from context_intelligence.managers import get_window_capture_manager

async def capture_with_edge_cases():
    manager = get_window_capture_manager()

    # Capture with automatic handling
    result = await manager.capture_window(
        window_id=12345,
        space_id=3,
        use_fallback=True
    )

    # Check result
    if result.success:
        print(f"✅ Success!")
        print(f"   Image: {result.image_path}")
        print(f"   Size: {result.resized_size}")

        # Handle edge cases
        if result.status.value == "image_too_large":
            print(f"   Resized from {result.original_size}")

        if result.status.value == "fallback_used":
            print(f"   Used fallback window {result.fallback_window_id}")

    else:
        print(f"❌ Failed: {result.error}")

        # Provide helpful guidance
        if result.status.value == "permission_denied":
            print("   Please enable screen recording permissions")
        elif result.status.value == "window_not_found":
            print("   Window may have closed")
```

### Example 2: Validate Space Before Operation

```python
from context_intelligence.managers import get_space_state_manager

async def validate_before_operation(space_id: int):
    manager = get_space_state_manager()

    # Get comprehensive state
    state_info = await manager.get_space_state(space_id)

    # Check if operation should proceed
    if not state_info.exists:
        print(state_info.error_message)  # "Space 10 doesn't exist. You have 6 spaces."
        return False

    if state_info.state.value == "empty":
        print(f"Space {space_id} is empty")
        return False

    if state_info.state.value == "minimized_only":
        apps = ", ".join(state_info.applications[:2])
        print(f"Space {space_id} has only minimized windows ({apps})")
        return False

    # Space is ready!
    print(f"✅ Space {space_id} ready: {state_info.window_count} windows")
    return True
```

### Example 3: Handle Edge Case in Action

```python
from context_intelligence.managers import get_space_state_manager

async def switch_to_space_safely(space_id: int):
    manager = get_space_state_manager()

    # Validate and handle edge case
    edge_case_result = await manager.handle_edge_case(space_id)

    if edge_case_result.edge_case == "not_exist":
        return {
            "success": False,
            "message": edge_case_result.message
        }

    if edge_case_result.edge_case == "transitioning":
        if edge_case_result.success:
            # Transition completed, space is now stable
            return await execute_switch(space_id)
        else:
            # Transition timed out
            return {
                "success": False,
                "message": "Space is transitioning, please try again"
            }

    # Space is ready
    return await execute_switch(space_id)
```

### Example 4: Capture with Fallback Chain

```python
from context_intelligence.managers import get_window_capture_manager

async def capture_from_space(space_id: int):
    """Capture any window from a space with automatic fallback"""

    manager = get_window_capture_manager()

    # Get windows in space (using yabai)
    import subprocess
    import json

    result = subprocess.run(
        ["yabai", "-m", "query", "--windows"],
        capture_output=True,
        text=True
    )

    windows = json.loads(result.stdout)
    space_windows = [w for w in windows if w.get("space") == space_id]

    if not space_windows:
        return {"success": False, "error": f"No windows in space {space_id}"}

    # Try primary window with automatic fallback
    primary_id = space_windows[0]["id"]

    capture_result = await manager.capture_window(
        window_id=primary_id,
        space_id=space_id,
        use_fallback=True  # Will try other windows if primary fails
    )

    return {
        "success": capture_result.success,
        "image_path": capture_result.image_path if capture_result.success else None,
        "message": capture_result.message,
        "fallback_used": capture_result.fallback_window_id is not None
    }
```

---

## API Reference

### SpaceStateManager

```python
from context_intelligence.managers import get_space_state_manager, initialize_space_state_manager

# Get singleton instance
manager = get_space_state_manager()

# Or initialize with custom settings
manager = initialize_space_state_manager(
    max_retry=3,              # Maximum retries for operations
    retry_delay=0.5,          # Initial retry delay in seconds
    transition_timeout=5.0    # Max time to wait for transitions
)

# Get space state
state_info: SpaceStateInfo = await manager.get_space_state(space_id: int)

# Handle edge case
edge_case_result: EdgeCaseResult = await manager.handle_edge_case(space_id: int)

# Validate and prepare for capture
should_capture, message, state_info = await manager.validate_and_prepare_capture(space_id: int)
```

#### SpaceStateInfo

```python
@dataclass
class SpaceStateInfo:
    space_id: int
    state: SpaceState  # ACTIVE, EMPTY, MINIMIZED_ONLY, FULLSCREEN, SPLIT_VIEW, etc.
    exists: bool
    window_count: int
    visible_window_count: int
    minimized_window_count: int
    windows: List[WindowInfo]
    applications: List[str]
    is_current: bool
    is_fullscreen: bool
    display_id: int
    error_message: Optional[str]
    detection_time: float
```

#### EdgeCaseResult

```python
@dataclass
class EdgeCaseResult:
    success: bool
    space_id: int
    edge_case: str  # "not_exist", "empty", "minimized_only", "transitioning", etc.
    message: str
    state_info: Optional[SpaceStateInfo]
    retry_count: int
    action_taken: Optional[str]
```

### WindowCaptureManager

```python
from context_intelligence.managers import get_window_capture_manager, initialize_window_capture_manager

# Get singleton instance
manager = get_window_capture_manager()

# Or initialize with custom settings
manager = initialize_window_capture_manager(
    max_retry=3,           # Maximum capture retries
    retry_delay=0.3,       # Delay between retries
    max_image_width=2560   # Maximum width before resizing
)

# Capture window
capture_result: CaptureResult = await manager.capture_window(
    window_id: int,
    output_path: Optional[str] = None,
    space_id: Optional[int] = None,
    use_fallback: bool = True
)
```

#### CaptureResult

```python
@dataclass
class CaptureResult:
    status: CaptureStatus  # SUCCESS, WINDOW_NOT_FOUND, PERMISSION_DENIED, etc.
    success: bool
    image_path: Optional[str]
    window_id: Optional[int]
    fallback_window_id: Optional[int]
    original_size: Tuple[int, int]
    resized_size: Tuple[int, int]
    message: str
    error: Optional[str]
    retry_count: int
    metadata: Dict[str, Any]
```

#### CaptureStatus

```python
class CaptureStatus(Enum):
    SUCCESS = "success"
    WINDOW_NOT_FOUND = "window_not_found"
    PERMISSION_DENIED = "permission_denied"
    WINDOW_OFF_SCREEN = "window_off_screen"
    WINDOW_TRANSPARENT = "window_transparent"
    IMAGE_TOO_LARGE = "image_too_large"
    CAPTURE_FAILED = "capture_failed"
    FALLBACK_USED = "fallback_used"
```

### SystemStateManager

```python
from context_intelligence.managers import get_system_state_manager, initialize_system_state_manager

# Get singleton instance
manager = get_system_state_manager()

# Or initialize with custom settings
manager = initialize_system_state_manager(
    auto_recover=True,      # Enable automatic recovery
    yabai_timeout=5.0,      # Timeout for yabai commands
    cache_ttl=5.0           # Cache TTL for system state checks
)

# Check system state
state_info: SystemStateInfo = await manager.check_system_state(use_cache: bool = True)

# Wait for system to become healthy
became_healthy, final_state = await manager.wait_for_healthy_state(
    timeout: float = 30.0,
    check_interval: float = 2.0
)

# Force cache refresh
await manager.refresh_state()
```

#### SystemStateInfo

```python
@dataclass
class SystemStateInfo:
    health: SystemHealth  # HEALTHY, DEGRADED, UNHEALTHY
    can_use_vision: bool
    can_use_spaces: bool
    yabai_status: YabaiStatus
    display_status: DisplayStatus
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str]
    recovery_suggestions: List[str]
    check_time: float
```

#### YabaiStatus

```python
@dataclass
class YabaiStatus:
    state: YabaiState  # RUNNING, NOT_INSTALLED, CRASHED, TIMEOUT
    is_running: bool
    can_recover: bool
    message: str
    recovery_command: Optional[str]
    metadata: Dict[str, Any]
```

#### DisplayStatus

```python
@dataclass
class DisplayStatus:
    state: DisplayState  # AWAKE, SLEEPING, LOCKED, NO_DISPLAYS
    is_available: bool
    is_headless: bool
    message: str
    metadata: Dict[str, Any]
```

#### SystemHealth

```python
class SystemHealth(Enum):
    HEALTHY = "healthy"        # All systems operational
    DEGRADED = "degraded"      # Some warnings but functional
    UNHEALTHY = "unhealthy"    # Critical issues, cannot operate
```

#### YabaiState

```python
class YabaiState(Enum):
    RUNNING = "running"             # Yabai is running normally
    NOT_INSTALLED = "not_installed" # Yabai not found
    CRASHED = "crashed"             # Yabai process crashed
    TIMEOUT = "timeout"             # Yabai command timed out
```

#### DisplayState

```python
class DisplayState(Enum):
    AWAKE = "awake"             # Display is on and unlocked
    SLEEPING = "sleeping"       # Display is asleep
    LOCKED = "locked"           # Screen is locked
    NO_DISPLAYS = "no_displays" # No displays detected (headless)
```

### APINetworkManager

```python
from context_intelligence.managers import get_api_network_manager, initialize_api_network_manager

# Get singleton instance
manager = get_api_network_manager()

# Or initialize with custom settings
manager = initialize_api_network_manager(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_retries=3,              # Maximum retry attempts
    initial_retry_delay=1.0,    # Initial retry delay in seconds
    max_image_width=2560,       # Maximum image width before resizing
    max_image_size_mb=5.0       # Maximum image size before compression
)

# Check readiness for API call
is_ready, message, status_info = await manager.check_ready_for_api_call()

# Execute API call with full edge case handling
result = await manager.execute_api_call_with_retry(
    func=api_function,
    optimize_image="/path/to/image.png",  # Optional image optimization
    cache_key="unique_cache_key"          # Optional caching
)

# Wait for system to become ready
became_ready, message = await manager.wait_for_ready(timeout=60.0)
```

#### APIStatus

```python
@dataclass
class APIStatus:
    state: APIState  # AVAILABLE, RATE_LIMITED, INVALID_KEY, TIMEOUT, UNAVAILABLE
    is_available: bool
    can_retry: bool
    message: str
    rate_limit_reset: Optional[datetime]
    retry_after_seconds: Optional[int]
    last_success: Optional[datetime]
    consecutive_failures: int
    metadata: Dict[str, Any]
```

#### NetworkStatus

```python
@dataclass
class NetworkStatus:
    state: NetworkState  # ONLINE, OFFLINE, DEGRADED
    is_online: bool
    latency_ms: Optional[float]
    message: str
    last_check: datetime
    metadata: Dict[str, Any]
```

#### ImageOptimizationResult

```python
@dataclass
class ImageOptimizationResult:
    status: ImageOptimizationStatus  # ALREADY_OPTIMIZED, RESIZED, COMPRESSED, CONVERTED, FAILED
    success: bool
    original_path: str
    optimized_path: str
    original_size_bytes: int
    optimized_size_bytes: int
    original_dimensions: Tuple[int, int]
    optimized_dimensions: Tuple[int, int]
    format_changed: bool
    message: str
    metadata: Dict[str, Any]

    @property
    def size_reduction_percent(self) -> float:
        """Calculate size reduction percentage"""
```

#### RetryResult

```python
@dataclass
class RetryResult:
    success: bool
    attempts: int
    total_delay: float
    final_error: Optional[str]
    result: Any  # The actual result if successful
    metadata: Dict[str, Any]
```

#### APIState

```python
class APIState(Enum):
    AVAILABLE = "available"          # API is available and working
    RATE_LIMITED = "rate_limited"    # Hit rate limit (429)
    INVALID_KEY = "invalid_key"      # API key is invalid/expired
    TIMEOUT = "timeout"              # Request timed out
    UNAVAILABLE = "unavailable"      # API is down/unreachable
```

#### NetworkState

```python
class NetworkState(Enum):
    ONLINE = "online"      # Connected to internet
    OFFLINE = "offline"    # No internet connection
    DEGRADED = "degraded"  # Slow/unstable connection
```

---

## Troubleshooting

### Common Issues

#### 1. Permission Denied Errors

**Symptom:** `CaptureStatus.PERMISSION_DENIED`

**Solution:**
1. Open System Settings
2. Go to Privacy & Security > Screen Recording
3. Enable screen recording for Python/Terminal/your app
4. Restart JARVIS

**Check permissions:**
```python
manager = get_window_capture_manager()
has_permission, message = await manager.permission_checker.check_screen_recording_permission()
print(message)
```

#### 2. Space Doesn't Exist

**Symptom:** `"Space 10 doesn't exist. You have 6 spaces."`

**Solution:** Use a valid space ID (1-6 in this example)

**Check available spaces:**
```python
manager = get_space_state_manager()
state_info = await manager.get_space_state(space_id=1)  # Try space 1
```

#### 3. All Windows Minimized

**Symptom:** `SpaceState.MINIMIZED_ONLY`

**Solution:** Un-minimize a window or switch to a different space

**Detect this condition:**
```python
edge_case_result = await manager.handle_edge_case(space_id=3)
if edge_case_result.edge_case == "minimized_only":
    print("All windows are minimized, cannot capture")
```

#### 4. 4K/5K Images Too Large

**Symptom:** `CaptureStatus.IMAGE_TOO_LARGE`

**Solution:** Already handled automatically! Images are resized to 2560px max width.

**Check if resized:**
```python
if result.original_size != result.resized_size:
    print(f"Resized from {result.original_size} to {result.resized_size}")
```

#### 5. Window Closed Mid-Capture

**Symptom:** `CaptureStatus.WINDOW_NOT_FOUND`

**Solution:** Use `use_fallback=True` to automatically try other windows

```python
# Automatic fallback to other windows in the same space
result = await manager.capture_window(
    window_id=12345,
    space_id=3,
    use_fallback=True  # ← This handles it!
)
```

#### 6. Yabai Not Running

**Symptom:** `YabaiState.NOT_INSTALLED` or `YabaiState.CRASHED`

**Solution:**

For not installed:
```bash
brew install koekeishiya/formulae/yabai
brew services start yabai
```

For crashed:
```bash
brew services restart yabai
```

**Auto-recovery enabled:**
```python
# Initialize with auto-recovery
manager = initialize_system_state_manager(auto_recover=True)

# Will automatically attempt recovery
state_info = await manager.check_system_state()
if state_info.yabai_status.state.value == "running":
    print("✅ Yabai recovered automatically")
```

#### 7. Display Sleeping or Locked

**Symptom:** `DisplayState.SLEEPING` or `DisplayState.LOCKED`

**Solution:** Wake the display or unlock the screen

**Detect this condition:**
```python
manager = get_system_state_manager()
state_info = await manager.check_system_state()

if state_info.display_status.state.value == "locked":
    print("Screen is locked. Unlock to continue.")
elif state_info.display_status.state.value == "sleeping":
    print("Display is sleeping. Wake to continue.")
```

#### 8. Headless Session (No Displays)

**Symptom:** `DisplayState.NO_DISPLAYS`

**Solution:** Vision features require a GUI session with displays attached

**Detect this condition:**
```python
state_info = await manager.check_system_state()

if state_info.display_status.is_headless:
    print("No displays detected. Vision requires GUI session.")
    print("Cannot use: screenshots, window capture, space management")
```

### Debugging

Enable detailed logging:

```python
import logging

# Set log level
logging.getLogger("context_intelligence.managers").setLevel(logging.DEBUG)
logging.getLogger("vision").setLevel(logging.DEBUG)

# Or for specific components
logging.getLogger("context_intelligence.managers.space_state_manager").setLevel(logging.DEBUG)
logging.getLogger("context_intelligence.managers.window_capture_manager").setLevel(logging.DEBUG)
logging.getLogger("context_intelligence.managers.system_state_manager").setLevel(logging.DEBUG)
```

Check metadata for detailed information:

```python
result = await manager.capture_window(window_id=12345, space_id=3)

print(f"Status: {result.status.value}")
print(f"Metadata: {result.metadata}")

# Example metadata:
# {
#   "window_info": {"app": "Chrome", "title": "GitHub", "state": "normal"},
#   "off_screen": False,
#   "transparent": False,
#   "resized": True,
#   "fallback_used": False
# }
```

---

## Performance Considerations

### Caching

- **Permission checks:** Cached for 60 seconds
- **Space state:** Not cached (always fresh)
- **Window validation:** Not cached (always fresh)
- **System state:** Cached with configurable TTL (default 5 seconds)

### Async Operations

All operations are async for non-blocking execution:

```python
# Good - concurrent operations
results = await asyncio.gather(
    manager.get_space_state(1),
    manager.get_space_state(2),
    manager.get_space_state(3)
)

# Bad - sequential blocking
for space_id in [1, 2, 3]:
    result = await manager.get_space_state(space_id)  # Blocks
```

### Retry Timing

- Initial retry delay: 300-500ms
- Exponential backoff: 2x per retry
- Max retries: 3 (configurable)
- Transition timeout: 5s (configurable)

---

## Version History

### v1.2 (2025-10-19)
- ✅ APINetworkManager with 5 API & network edge cases
- ✅ APIHealthChecker for Claude API status and circuit breaker
- ✅ NetworkDetector for real-time connectivity monitoring
- ✅ ImageOptimizer for automatic image resizing and compression
- ✅ RetryHandler with exponential backoff and result caching
- ✅ Integration with claude_streamer.py (readiness checks)
- ✅ Integration with claude_vision_analyzer_main.py (image optimization)
- ✅ Enhanced documentation with API/Network API reference
- ✅ New troubleshooting guides for API/network issues

### v1.1 (2025-10-19)
- ✅ SystemStateManager with 5 system state edge cases
- ✅ YabaiHealthChecker for yabai service monitoring
- ✅ DisplayStateDetector for display state detection
- ✅ SystemRecoveryHandler with automatic recovery
- ✅ Integration with multi_space_capture_engine (health checks)
- ✅ Integration with action_executor (health checks)
- ✅ Enhanced documentation with system state API reference
- ✅ New troubleshooting guides for system state issues

### v1.0 (2025-10-19)
- ✅ Initial release
- ✅ SpaceStateManager with 6 edge cases
- ✅ WindowCaptureManager with 5 edge cases
- ✅ Integration with temporal_query_handler
- ✅ Integration with multi_space_capture_engine
- ✅ Integration with reliable_screenshot_capture
- ✅ Integration with action_executor
- ✅ Integration with multi_space_query_handler

---

## Support

For issues or questions:
- Check logs: `logging.getLogger("context_intelligence.managers")`
- Review metadata in results for detailed diagnostics
- Enable DEBUG logging for step-by-step execution traces

---

**End of Documentation**
