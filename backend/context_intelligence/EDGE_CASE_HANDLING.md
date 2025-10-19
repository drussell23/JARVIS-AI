# Edge Case Handling System - Complete Documentation

**Version:** 1.3
**Last Updated:** 2025-10-19
**Status:** ✅ Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Space-Related Edge Cases](#space-related-edge-cases)
3. [Window Capture Edge Cases](#window-capture-edge-cases)
4. [System State Edge Cases](#system-state-edge-cases)
5. [API & Network Edge Cases](#api--network-edge-cases)
6. [Error Handling Matrix](#error-handling-matrix)
7. [Capture Fallbacks](#capture-fallbacks)
8. [OCR Fallbacks](#ocr-fallbacks)
9. [Integration Points](#integration-points)
10. [Usage Examples](#usage-examples)
11. [API Reference](#api-reference)
12. [Troubleshooting](#troubleshooting)

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

## Error Handling Matrix

### Overview

**Location:** `backend/context_intelligence/managers/error_handling_matrix.py`

The Error Handling Matrix provides comprehensive error handling with graceful degradation following a priority-based strategy:

```
Priority 1: Try primary method
   ↓ (fails)
Priority 2: Try fallback method
   ↓ (fails)
Priority 3: Return partial results + warning
   ↓ (fails)
Priority 4: Return user-friendly error message
```

### Components

#### 1. FallbackChain

Defines the execution order for methods with priority levels.

```python
from context_intelligence.managers import FallbackChain, ExecutionPriority

# Create a fallback chain
chain = FallbackChain("screenshot_capture")

# Add methods in priority order
chain.add_primary(capture_with_cg, timeout=5.0)
chain.add_fallback(capture_with_screencapture, timeout=10.0)
chain.add_secondary(capture_with_pyautogui, timeout=15.0)
chain.add_last_resort(capture_with_cli, timeout=20.0)
```

**Priority Levels:**
- `PRIMARY (1)`: Main method (fastest/best quality)
- `FALLBACK (2)`: First fallback
- `SECONDARY (3)`: Second fallback
- `TERTIARY (4)`: Third fallback
- `LAST_RESORT (5)`: Final attempt

#### 2. ErrorHandlingMatrix (Main)

Executes fallback chains with graceful degradation.

```python
from context_intelligence.managers import (
    initialize_error_handling_matrix,
    FallbackChain
)

# Initialize matrix
matrix = initialize_error_handling_matrix(
    default_timeout=30.0,
    aggregation_strategy="first_success",  # or "best_result", "merge", "union"
    recovery_strategy="continue"            # or "retry", "abort"
)

# Create fallback chain
chain = FallbackChain("api_call")
chain.add_primary(call_primary_api, timeout=5.0)
chain.add_fallback(call_backup_api, timeout=10.0)
chain.add_last_resort(use_cached_result, timeout=1.0)

# Execute with graceful degradation
report = await matrix.execute_chain(chain, stop_on_success=True)

if report.success:
    print(f"✅ {report.message}")
    result = report.final_result
else:
    print(f"❌ {report.message}")
    for error in report.errors:
        print(f"  • {error}")
```

#### 3. PartialResultAggregator

Aggregates partial results when multiple methods succeed.

**Strategies:**
- `first_success`: Use first successful result
- `best_result`: Use best result based on quality metric
- `merge`: Merge all results (for dicts)
- `union`: Union of all results (for lists/sets)

```python
# Example: Merge results from multiple sources
chain = FallbackChain("data_fetch")
chain.add_primary(fetch_from_primary_db)
chain.add_fallback(fetch_from_backup_db)
chain.add_secondary(fetch_from_cache)

# Initialize with merge strategy
matrix = initialize_error_handling_matrix(aggregation_strategy="merge")

# Execute - will merge results from all successful sources
report = await matrix.execute_chain(chain, collect_partial=True)

if report.success:
    # Get merged result
    merged_data = report.final_result
    print(f"Merged data from {len(report.methods_attempted)} sources")
```

#### 4. ErrorRecoveryStrategy

Defines how to recover from errors.

**Strategies:**
- `continue`: Continue to next method (default)
- `retry`: Retry the same method (with max_retries)
- `abort`: Stop execution immediately

```python
# Initialize with retry strategy
matrix = initialize_error_handling_matrix(
    recovery_strategy="retry",
    max_retries=2
)

# Methods will be retried up to 2 times before moving to fallback
```

#### 5. ErrorMessageGenerator

Generates user-friendly error messages with suggestions.

```python
from context_intelligence.managers import ErrorMessageGenerator

# Generate message from execution report
message = ErrorMessageGenerator.generate_message(
    report,
    include_technical=True,    # Include technical details
    include_suggestions=True    # Include actionable suggestions
)

print(message)
```

**Example output:**
```
❌ screenshot_capture failed

Errors encountered:
  • capture_with_cg: Timeout after 5.2s
  • capture_with_screencapture: Permission denied

💡 Suggestions:
  • Check system permissions in Settings > Privacy & Security
  • Try increasing the timeout value

🔧 Technical Details:
  • Attempted 3 method(s)
  • Duration: 15.3s
  ✅ capture_with_cli: success
  ❌ capture_with_cg: timeout
  ❌ capture_with_screencapture: failed
```

### Execution Flow

```
1. Check cache (if applicable)
        ↓
2. Execute Primary Method (Priority 1)
        ↓
   Success? → Stop and return result
        ↓ No
3. Apply Recovery Strategy
   • Continue → Try Fallback
   • Retry → Retry Primary (up to max_retries)
   • Abort → Stop and return error
        ↓
4. Execute Fallback Method (Priority 2)
        ↓
   Success? → Stop and return result
        ↓ No
5. Execute Secondary/Tertiary (Priority 3-4)
        ↓
6. Execute Last Resort (Priority 5)
        ↓
7. Aggregate Partial Results (if any)
        ↓
8. Generate User-Friendly Error Message
```

### Result Quality Levels

The matrix determines result quality based on which methods succeeded:

| Quality | Condition | Meaning |
|---------|-----------|---------|
| **FULL** | Primary succeeded | Best result, no degradation |
| **DEGRADED** | Fallback succeeded | Good result, used fallback |
| **PARTIAL** | Multiple methods succeeded | Partial data, some methods failed |
| **MINIMAL** | Only last resort succeeded | Minimal result, most methods failed |
| **FAILED** | All methods failed | No result available |

### Integration Example: Screenshot Capture

```python
# In reliable_screenshot_capture.py
async def capture_space_with_matrix(self, space_id: int) -> ScreenshotResult:
    """Capture screenshot using Error Handling Matrix"""

    # Build fallback chain
    chain = FallbackChain(f"capture_space_{space_id}")

    chain.add_primary(capture_with_cg, timeout=5.0)
    chain.add_fallback(capture_with_screencapture, timeout=8.0)
    chain.add_secondary(capture_with_quartz, timeout=10.0)
    chain.add_last_resort(capture_with_cli, timeout=15.0)

    # Execute with matrix
    report = await self.error_matrix.execute_chain(chain, stop_on_success=True)

    if report.success:
        logger.info(f"✅ Captured space {space_id} - {report.message}")
        return report.final_result
    else:
        # Generate helpful error message
        error_msg = ErrorMessageGenerator.generate_message(
            report,
            include_technical=True,
            include_suggestions=True
        )
        logger.error(f"❌ Failed to capture space {space_id}:\n{error_msg}")

        return ScreenshotResult(
            success=False,
            error=error_msg,
            metadata={"execution_report": report}
        )
```

### Convenience Method

For simple fallback scenarios, use `execute_with_fallbacks`:

```python
# Execute with simple fallback list
report = await matrix.execute_with_fallbacks(
    operation_name="fetch_data",
    primary=fetch_from_api,
    fallbacks=[fetch_from_cache, fetch_from_file],
    arg1="value1",
    kwarg1="value2"
)
```

### Benefits

✅ **Automatic Fallback** - Seamless transition between methods
✅ **Partial Results** - Return something useful even when primary fails
✅ **User-Friendly Errors** - Helpful messages with actionable suggestions
✅ **Detailed Reporting** - Know exactly what happened
✅ **Configurable Recovery** - Retry, continue, or abort strategies
✅ **Async Support** - Fully async with timeout support

---

## Capture Fallbacks

### Overview

**Location:** `backend/context_intelligence/managers/capture_strategy_manager.py`

The Capture Strategy Manager implements intelligent screenshot capture with a 4-step fallback sequence:

```
1. Primary: Capture specific window
       ↓ (fails)
2. Fallback 1: Capture entire space
       ↓ (fails)
3. Fallback 2: Use cached screenshot (if <60s old)
       ↓ (fails)
4. Fallback 3: Return user-friendly error message
```

Uses the Error Handling Matrix for graceful degradation and cache management with TTL.

### Components

#### 1. CaptureCache

Manages cached screenshots with time-based expiration.

```python
from context_intelligence.managers import CaptureCache, CachedCapture
from datetime import datetime

# Create cache
cache = CaptureCache(default_ttl=60.0, max_entries=100)

# Store capture
cached = CachedCapture(
    image=screenshot_data,
    window_id=12345,
    space_id=3,
    timestamp=datetime.now(),
    method="window_capture",
    metadata={"source": "primary"}
)
cache.store(cached)

# Retrieve by space
cached_capture = cache.get_by_space(space_id=3, max_age=60.0)
if cached_capture:
    print(f"Cache hit! Age: {cached_capture.age_seconds():.1f}s")
    image = cached_capture.image

# Retrieve by window
cached_capture = cache.get_by_window(window_id=12345, max_age=60.0)

# Get stats
stats = cache.get_stats()
print(f"Space entries: {stats['space_entries']}")
print(f"Window entries: {stats['window_entries']}")

# Clear cache
cache.clear()
```

**Features:**
- Time-based expiration (configurable TTL)
- Space-based caching
- Window-based caching
- Automatic cleanup when max_entries exceeded
- Cache validity checking

#### 2. CachedCapture

Data class representing a cached screenshot.

```python
@dataclass
class CachedCapture:
    image: Any                    # Image data (PIL, numpy, etc.)
    window_id: Optional[int]      # Window ID (if window capture)
    space_id: int                 # Space ID
    timestamp: datetime           # Capture time
    method: str                   # Capture method used
    metadata: Dict[str, Any]      # Additional metadata

    def is_valid(self, max_age_seconds: float = 60.0) -> bool:
        """Check if cache is still valid"""

    def age_seconds(self) -> float:
        """Get age in seconds"""
```

#### 3. CaptureStrategyManager (Main)

Main coordinator for intelligent capture with fallbacks.

```python
from context_intelligence.managers import (
    initialize_capture_strategy_manager,
    get_capture_strategy_manager
)

# Initialize manager
manager = initialize_capture_strategy_manager(
    cache_ttl=60.0,              # Cache time-to-live in seconds
    max_cache_entries=100,       # Maximum cache entries
    enable_error_matrix=True     # Use Error Handling Matrix
)

# Or get existing instance
manager = get_capture_strategy_manager()

# Capture with full fallback chain
success, image, message = await manager.capture_with_fallbacks(
    space_id=3,
    window_id=12345,                    # Optional: specific window
    window_capture_func=capture_window, # Async function to capture window
    space_capture_func=capture_space,   # Async function to capture space
    cache_max_age=60.0                  # Optional: max cache age
)

if success:
    print(f"✅ {message}")
    # Use image
    process_screenshot(image)
else:
    print(f"❌ {message}")

# Get cache stats
stats = manager.get_cache_stats()
print(f"Cache entries: {stats['total_entries']}")

# Clear cache
manager.clear_cache()
```

### Fallback Sequence

#### Step 1: Capture Specific Window (Primary)

If `window_id` is provided, attempts to capture the specific window.

```python
async def capture_window(window_id: int, space_id: int):
    """Capture specific window"""
    # Your window capture logic
    return screenshot_image

# Capture with window fallback
success, image, message = await manager.capture_with_fallbacks(
    space_id=3,
    window_id=12345,
    window_capture_func=capture_window,
    space_capture_func=capture_space
)
```

**Behavior:**
- Timeout: 10 seconds
- Priority: PRIMARY
- On success: Caches result and returns immediately
- On failure: Proceeds to Step 2

#### Step 2: Capture Entire Space (Fallback 1)

If window capture fails, captures the entire space.

```python
async def capture_space(space_id: int):
    """Capture entire space"""
    # Your space capture logic
    return screenshot_image

# Capture with space fallback
success, image, message = await manager.capture_with_fallbacks(
    space_id=3,
    space_capture_func=capture_space
)
```

**Behavior:**
- Timeout: 15 seconds
- Priority: FALLBACK
- On success: Caches result and returns immediately
- On failure: Proceeds to Step 3

#### Step 3: Use Cached Screenshot (Fallback 2)

If both capture methods fail, uses cached screenshot if available and valid.

```python
# Cache is checked automatically
success, image, message = await manager.capture_with_fallbacks(
    space_id=3,
    window_id=12345,
    window_capture_func=capture_window,
    space_capture_func=capture_space,
    cache_max_age=60.0  # Use cache if <60s old
)

if success and "cached" in message.lower():
    print(f"Using cached capture: {message}")
```

**Behavior:**
- Timeout: 1 second (fast lookup)
- Priority: SECONDARY
- Checks window cache first (if window_id provided)
- Then checks space cache
- Returns cached image if valid
- On failure: Proceeds to Step 4

#### Step 4: Return Error Message (Fallback 3)

If all methods fail, generates user-friendly error message.

```python
success, image, message = await manager.capture_with_fallbacks(
    space_id=3,
    window_id=12345,
    window_capture_func=capture_window,
    space_capture_func=capture_space
)

if not success:
    # User-friendly error message
    print(f"❌ {message}")
    # Example: "Unable to capture Space 3"
```

**Behavior:**
- Generated by ErrorMessageGenerator
- Includes helpful suggestions if available
- Provides context about what failed

### Integration Example

**In multi_space_capture_engine.py:**

```python
from context_intelligence.managers import (
    get_capture_strategy_manager,
    initialize_capture_strategy_manager
)

class MultiSpaceCaptureEngine:
    def __init__(self):
        # Initialize Capture Strategy Manager
        self.capture_strategy_manager = get_capture_strategy_manager()
        if not self.capture_strategy_manager:
            self.capture_strategy_manager = initialize_capture_strategy_manager(
                cache_ttl=60.0,
                max_cache_entries=100,
                enable_error_matrix=True
            )
        logger.info("✅ Capture Strategy Manager initialized")

    async def capture_space_intelligent(self, space_id: int, window_id: Optional[int] = None):
        """Capture with intelligent fallbacks"""

        if not self.capture_strategy_manager:
            # Fallback to legacy capture
            return await self._legacy_capture(space_id)

        # Use intelligent capture with fallbacks
        success, image, message = await self.capture_strategy_manager.capture_with_fallbacks(
            space_id=space_id,
            window_id=window_id,
            window_capture_func=self._capture_window,
            space_capture_func=self._capture_space,
            cache_max_age=60.0
        )

        if success:
            logger.info(f"✅ Capture successful: {message}")
            return image
        else:
            logger.error(f"❌ Capture failed: {message}")
            return None

    async def _capture_window(self, window_id: int, space_id: int):
        """Capture specific window"""
        # Implementation using WindowCaptureManager or CGWindowCapture
        return screenshot_image

    async def _capture_space(self, space_id: int):
        """Capture entire space"""
        # Implementation using screencapture or other method
        return screenshot_image
```

### Usage Examples

#### Example 1: Simple Window Capture with Fallbacks

```python
from context_intelligence.managers import get_capture_strategy_manager

async def capture_window_smart(window_id: int, space_id: int):
    manager = get_capture_strategy_manager()

    # Capture with automatic fallbacks
    success, image, message = await manager.capture_with_fallbacks(
        space_id=space_id,
        window_id=window_id,
        window_capture_func=my_window_capture,
        space_capture_func=my_space_capture
    )

    if success:
        print(f"✅ {message}")
        return image
    else:
        print(f"❌ {message}")
        return None
```

#### Example 2: Space Capture with Cache

```python
from context_intelligence.managers import initialize_capture_strategy_manager

async def capture_space_cached(space_id: int):
    # Initialize with 30-second cache
    manager = initialize_capture_strategy_manager(
        cache_ttl=30.0,
        max_cache_entries=50
    )

    # First call: Captures and caches
    success, image1, msg1 = await manager.capture_with_fallbacks(
        space_id=space_id,
        space_capture_func=my_space_capture,
        cache_max_age=30.0
    )
    print(f"First call: {msg1}")  # "Captured space 3"

    # Second call within 30s: Uses cache
    success, image2, msg2 = await manager.capture_with_fallbacks(
        space_id=space_id,
        space_capture_func=my_space_capture,
        cache_max_age=30.0
    )
    print(f"Second call: {msg2}")  # "Using cached capture (age=2.3s)"

    return image2
```

#### Example 3: Multiple Spaces with Parallel Capture

```python
import asyncio
from context_intelligence.managers import get_capture_strategy_manager

async def capture_multiple_spaces(space_ids: List[int]):
    manager = get_capture_strategy_manager()

    # Capture all spaces in parallel
    tasks = [
        manager.capture_with_fallbacks(
            space_id=sid,
            space_capture_func=my_space_capture
        )
        for sid in space_ids
    ]

    results = await asyncio.gather(*tasks)

    # Process results
    successful_captures = {}
    failed_captures = {}

    for space_id, (success, image, message) in zip(space_ids, results):
        if success:
            successful_captures[space_id] = image
            logger.info(f"✅ Space {space_id}: {message}")
        else:
            failed_captures[space_id] = message
            logger.error(f"❌ Space {space_id}: {message}")

    return successful_captures, failed_captures
```

#### Example 4: Custom Cache Behavior

```python
from context_intelligence.managers import CaptureStrategyManager

# Create manager with custom settings
manager = CaptureStrategyManager(
    cache_ttl=120.0,        # 2-minute cache
    max_cache_entries=200,  # More cache entries
    enable_error_matrix=True
)

# Capture with short cache validity
success, image, message = await manager.capture_with_fallbacks(
    space_id=3,
    space_capture_func=my_space_capture,
    cache_max_age=30.0  # Override default TTL (only use cache if <30s old)
)

# Check cache stats
stats = manager.get_cache_stats()
print(f"Cache: {stats['total_entries']}/{stats['max_entries']} entries")

# Clear cache when needed
manager.clear_cache()
```

### Result Quality Reporting

The manager reports result quality based on which method succeeded:

```python
success, image, message = await manager.capture_with_fallbacks(
    space_id=3,
    window_id=12345,
    window_capture_func=capture_window,
    space_capture_func=capture_space
)

# Message indicates quality:
# - "Captured window 12345 in space 3"           → FULL quality (window capture)
# - "Captured space 3 using fallback method"     → DEGRADED quality (space capture)
# - "Using cached capture for space 3"           → PARTIAL quality (cache used)
# - "Unable to capture Space 3"                  → FAILED (all methods failed)
```

### Benefits

✅ **Intelligent Fallbacks** - Automatic progression through capture methods
✅ **Cache Management** - Reduces capture overhead for recent screenshots
✅ **Error Matrix Integration** - Leverages existing error handling infrastructure
✅ **Result Quality Tracking** - Know which method was used
✅ **User-Friendly Errors** - Helpful messages when all methods fail
✅ **Flexible Configuration** - Configurable TTL and cache size
✅ **Async Support** - Fully async with timeout support
✅ **Parallel Capture** - Capture multiple spaces concurrently

### API Reference

#### CaptureStrategyManager

```python
class CaptureStrategyManager:
    def __init__(
        self,
        cache_ttl: float = 60.0,
        max_cache_entries: int = 100,
        enable_error_matrix: bool = True
    )

    async def capture_with_fallbacks(
        self,
        space_id: int,
        window_id: Optional[int] = None,
        window_capture_func: Optional[Callable] = None,
        space_capture_func: Optional[Callable] = None,
        cache_max_age: Optional[float] = None
    ) -> Tuple[bool, Any, str]:
        """
        Capture with intelligent fallbacks

        Returns:
            (success, image_data, message)
        """

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""

    def clear_cache(self):
        """Clear all cache"""
```

#### CaptureCache

```python
class CaptureCache:
    def __init__(self, default_ttl: float = 60.0, max_entries: int = 100)

    def get_by_space(
        self,
        space_id: int,
        max_age: Optional[float] = None
    ) -> Optional[CachedCapture]:
        """Get cached capture by space ID"""

    def get_by_window(
        self,
        window_id: int,
        max_age: Optional[float] = None
    ) -> Optional[CachedCapture]:
        """Get cached capture by window ID"""

    def store(self, capture: CachedCapture):
        """Store capture in cache"""

    def clear(self):
        """Clear all cache"""

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
```

#### CachedCapture

```python
@dataclass
class CachedCapture:
    image: Any                    # Image data
    window_id: Optional[int]      # Window ID (if window capture)
    space_id: int                 # Space ID
    timestamp: datetime           # Capture time
    method: str                   # Capture method ("window_capture", "space_capture", etc.)
    metadata: Dict[str, Any]      # Additional metadata

    def is_valid(self, max_age_seconds: float = 60.0) -> bool:
        """Check if cache is still valid"""

    def age_seconds(self) -> float:
        """Get age in seconds since capture"""
```

#### Global Functions

```python
def get_capture_strategy_manager() -> Optional[CaptureStrategyManager]:
    """Get the global capture strategy manager instance"""

def initialize_capture_strategy_manager(
    cache_ttl: float = 60.0,
    max_cache_entries: int = 100,
    enable_error_matrix: bool = True
) -> CaptureStrategyManager:
    """Initialize the global capture strategy manager"""
```

---

## OCR Fallbacks

### Overview

**Location:** `backend/context_intelligence/managers/ocr_strategy_manager.py`

The OCR Strategy Manager implements intelligent OCR text extraction with a 4-step fallback sequence:

```
1. Primary: Claude Vision API
       ↓ (fails)
2. Fallback 1: Use cached OCR (if <5min old)
       ↓ (fails)
3. Fallback 2: Local OCR (Tesseract)
       ↓ (fails)
4. Fallback 3: Return image metadata only
```

Uses the Error Handling Matrix for graceful degradation and cache management with TTL.

### Components

#### 1. OCRCache

Manages cached OCR results with time-based expiration.

```python
from context_intelligence.managers import OCRCache, CachedOCR
from datetime import datetime

# Create cache
cache = OCRCache(default_ttl=300.0, max_entries=200)  # 5 minutes

# Store OCR result
cached = CachedOCR(
    text="Extracted text here",
    image_hash="abc123def456",
    timestamp=datetime.now(),
    method="claude_vision",
    confidence=0.95,
    metadata={"source": "primary"}
)
cache.store(cached)

# Retrieve by image hash
cached_ocr = cache.get(image_hash="abc123def456", max_age=300.0)
if cached_ocr:
    print(f"Cache hit! Age: {cached_ocr.age_seconds():.1f}s")
    print(f"Text: {cached_ocr.text}")
    print(f"Confidence: {cached_ocr.confidence:.2f}")

# Get stats
stats = cache.get_stats()
print(f"Total entries: {stats['total_entries']}")

# Clear cache
cache.clear()
```

**Features:**
- Time-based expiration (default 5 minutes)
- Image hash-based caching (MD5)
- Automatic cleanup when max_entries exceeded
- Cache validity checking
- Confidence scoring

#### 2. CachedOCR

Data class representing a cached OCR result.

```python
@dataclass
class CachedOCR:
    text: str                     # Extracted text
    image_hash: str               # Image hash (MD5)
    timestamp: datetime           # Extraction time
    method: str                   # Method used ("claude_vision", "tesseract", "metadata")
    confidence: float             # Confidence score (0.0-1.0)
    metadata: Dict[str, Any]      # Additional metadata

    def is_valid(self, max_age_seconds: float = 300.0) -> bool:
        """Check if cache is still valid"""

    def age_seconds(self) -> float:
        """Get age in seconds"""
```

#### 3. ClaudeVisionOCR

Primary OCR engine using Claude Vision API.

```python
from context_intelligence.managers import ClaudeVisionOCR

# Initialize with Anthropic client
ocr_engine = ClaudeVisionOCR(api_client=anthropic_client)

# Extract text
text, confidence = await ocr_engine.extract_text(
    image_path="/path/to/screenshot.png",
    prompt="Extract all text from this image"  # Optional custom prompt
)

print(f"Extracted: {text}")
print(f"Confidence: {confidence:.2f}")
```

**Features:**
- Uses Claude 3.5 Sonnet for highest accuracy
- Custom prompts supported
- Automatic confidence scoring (0.95 for successful extractions)
- Base64 image encoding
- Async execution

#### 4. TesseractOCR

Fallback local OCR engine.

```python
from context_intelligence.managers import TesseractOCR

# Initialize
ocr_engine = TesseractOCR()

# Check availability
if ocr_engine.is_available:
    # Extract text
    text, confidence = await ocr_engine.extract_text("/path/to/screenshot.png")
    print(f"Extracted: {text}")
    print(f"Confidence: {confidence:.2f}")
else:
    print("Tesseract not installed. Install with: brew install tesseract")
```

**Features:**
- Uses Tesseract OCR engine (local, no API needed)
- Automatic confidence parsing from TSV output
- Async execution (runs in thread pool)
- OEM 3 (best OCR engine mode)
- PSM 6 (uniform text block)

#### 5. ImageHasher & ImageMetadataExtractor

Utilities for image processing.

```python
from context_intelligence.managers import ImageHasher, ImageMetadataExtractor

# Compute image hash
hasher = ImageHasher()
image_hash = hasher.compute_hash("/path/to/image.png")
print(f"Hash: {image_hash}")

# Extract metadata
extractor = ImageMetadataExtractor()
metadata = await extractor.extract_metadata("/path/to/image.png")
print(f"Size: {metadata['width']}x{metadata['height']}")
print(f"Format: {metadata['format']}")
print(f"File size: {metadata['size_bytes']} bytes")
```

#### 6. OCRStrategyManager (Main)

Main coordinator for intelligent OCR with fallbacks.

```python
from context_intelligence.managers import (
    initialize_ocr_strategy_manager,
    get_ocr_strategy_manager
)
from anthropic import Anthropic

# Initialize manager
client = Anthropic(api_key="your-api-key")
manager = initialize_ocr_strategy_manager(
    api_client=client,
    cache_ttl=300.0,              # Cache time-to-live (5 minutes)
    max_cache_entries=200,        # Maximum cache entries
    enable_error_matrix=True      # Use Error Handling Matrix
)

# Or get existing instance
manager = get_ocr_strategy_manager()

# Extract text with full fallback chain
result = await manager.extract_text_with_fallbacks(
    image_path="/path/to/screenshot.png",
    prompt=None,                  # Optional custom prompt
    cache_max_age=300.0,          # Optional: max cache age
    skip_cache=False              # Optional: skip cache lookup
)

if result.success:
    print(f"✅ Extracted {len(result.text)} characters")
    print(f"   Method: {result.method}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Execution time: {result.execution_time:.2f}s")
    print(f"   Text: {result.text}")
else:
    print(f"❌ OCR failed: {result.error}")

# Get cache stats
stats = manager.get_cache_stats()
print(f"Cache entries: {stats['total_entries']}")

# Clear cache
manager.clear_cache()
```

### Fallback Sequence

#### Step 1: Claude Vision API (Primary)

If Claude API client is available, attempts OCR using Claude Vision.

```python
# Automatic with manager
result = await manager.extract_text_with_fallbacks(
    image_path="/path/to/screenshot.png"
)

# result.method will be "claude_vision" if successful
```

**Behavior:**
- Timeout: 60 seconds
- Priority: PRIMARY
- Confidence: 0.95 (high)
- On success: Caches result and returns immediately
- On failure: Proceeds to Step 2

**Handles:**
- Rate limits (429) → Fallback to cache or Tesseract
- Network errors → Fallback to cache or Tesseract
- API timeouts → Fallback to cache or Tesseract
- Invalid API key → Fallback to cache or Tesseract

#### Step 2: Cached OCR (Fallback 1)

If primary fails, checks cache for recent OCR results.

```python
# Cache is checked automatically
result = await manager.extract_text_with_fallbacks(
    image_path="/path/to/screenshot.png",
    cache_max_age=300.0  # Use cache if <5 minutes old
)

# result.method will be "cached_claude_vision" or "cached_tesseract" if from cache
```

**Behavior:**
- Timeout: 1 second (fast lookup)
- Priority: FALLBACK
- Uses image hash (MD5) for cache lookup
- Returns cached result if valid
- On failure: Proceeds to Step 3

#### Step 3: Tesseract OCR (Fallback 2)

If cache misses, uses local Tesseract OCR.

```python
# Automatic fallback to Tesseract
result = await manager.extract_text_with_fallbacks(
    image_path="/path/to/screenshot.png"
)

# result.method will be "tesseract" if Tesseract was used
```

**Behavior:**
- Timeout: 30 seconds
- Priority: SECONDARY
- Requires Tesseract installed (`brew install tesseract`)
- On success: Caches result for future use
- On failure: Proceeds to Step 4

**Installation:**
```bash
brew install tesseract
```

#### Step 4: Image Metadata (Fallback 3)

If all OCR methods fail, returns basic image metadata.

```python
# Final fallback
result = await manager.extract_text_with_fallbacks(
    image_path="/path/to/screenshot.png"
)

# If all methods fail, result.method will be "metadata"
# result.text will contain: "Image: 1920x1080, screenshot.png"
```

**Behavior:**
- Timeout: 5 seconds
- Priority: LAST_RESORT
- Always succeeds (unless file doesn't exist)
- Returns formatted metadata string

### Integration Examples

**In vision/adapters/ocr.py:**

```python
from context_intelligence.managers import get_ocr_strategy_manager

async def ocr_text_from_snapshot(snapshot_id: str, use_claude: bool = True) -> str:
    """Extract OCR text with intelligent fallbacks"""

    # Resolve snapshot path
    snapshot_path = _resolve_snapshot_path(snapshot_id)

    if not snapshot_path:
        return ""

    # Use OCR Strategy Manager
    manager = get_ocr_strategy_manager()

    if manager and use_claude:
        # Extract with full fallback chain
        result = await manager.extract_text_with_fallbacks(
            image_path=str(snapshot_path),
            cache_max_age=300.0,
            skip_cache=False
        )

        if result.success:
            logger.info(f"✅ OCR: {result.method} (confidence={result.confidence:.2f})")
            return result.text

    # Fallback to legacy Tesseract-only
    return await legacy_ocr(snapshot_path)
```

**In claude_vision_analyzer_main.py:**

```python
from backend.context_intelligence.managers import (
    get_ocr_strategy_manager,
    initialize_ocr_strategy_manager
)

class EnhancedClaudeVisionAnalyzer:
    def __init__(self, api_key: str, config: VisionConfig):
        # Initialize Anthropic client
        self.client = Anthropic(api_key=api_key)

        # Initialize OCR Strategy Manager
        self.ocr_strategy_manager = get_ocr_strategy_manager()
        if not self.ocr_strategy_manager:
            self.ocr_strategy_manager = initialize_ocr_strategy_manager(
                api_client=self.client,
                cache_ttl=300.0,
                max_cache_entries=200,
                enable_error_matrix=True
            )
        logger.info("✅ OCR Strategy Manager initialized")

    async def extract_text_from_image(self, image_path: str) -> str:
        """Extract text using OCR Strategy Manager"""
        if self.ocr_strategy_manager:
            result = await self.ocr_strategy_manager.extract_text_with_fallbacks(
                image_path=image_path
            )
            return result.text if result.success else ""
        return ""
```

### Usage Examples

#### Example 1: Simple Text Extraction

```python
from context_intelligence.managers import get_ocr_strategy_manager

async def extract_text_smart(image_path: str):
    manager = get_ocr_strategy_manager()

    # Extract with automatic fallbacks
    result = await manager.extract_text_with_fallbacks(image_path)

    if result.success:
        print(f"✅ Extracted {len(result.text)} characters")
        print(f"   Method: {result.method}")
        print(f"   Confidence: {result.confidence:.2f}")
        return result.text
    else:
        print(f"❌ Failed: {result.error}")
        return ""
```

#### Example 2: Custom Prompt for Specific Extraction

```python
async def extract_code_from_screenshot(image_path: str):
    manager = get_ocr_strategy_manager()

    # Custom prompt for code extraction
    result = await manager.extract_text_with_fallbacks(
        image_path=image_path,
        prompt=(
            "Extract all code from this screenshot. "
            "Preserve indentation and formatting. "
            "Return ONLY the code, nothing else."
        )
    )

    if result.success and result.method == "claude_vision":
        # Claude Vision was used (best for code)
        return result.text
    elif result.success:
        # Fallback method was used
        print(f"⚠️ Using fallback: {result.method}")
        return result.text
    else:
        return ""
```

#### Example 3: Batch Text Extraction with Caching

```python
async def extract_text_from_multiple_images(image_paths: List[str]):
    manager = get_ocr_strategy_manager()

    results = []

    for path in image_paths:
        # First call: Extracts and caches
        result = await manager.extract_text_with_fallbacks(path)

        if result.success:
            results.append({
                "path": path,
                "text": result.text,
                "method": result.method,
                "confidence": result.confidence,
                "cached": "cached" in result.method
            })

    # Second pass through same images: Uses cache
    for path in image_paths:
        result = await manager.extract_text_with_fallbacks(path)
        # Will use cache (very fast)

    return results
```

#### Example 4: Skip Cache for Fresh Extraction

```python
async def force_fresh_ocr(image_path: str):
    manager = get_ocr_strategy_manager()

    # Force fresh extraction (skip cache)
    result = await manager.extract_text_with_fallbacks(
        image_path=image_path,
        skip_cache=True  # Always extract fresh
    )

    return result.text if result.success else ""
```

#### Example 5: Adjust Cache TTL

```python
async def extract_with_short_cache(image_path: str):
    manager = get_ocr_strategy_manager()

    # Use cache only if <60 seconds old (instead of default 5 minutes)
    result = await manager.extract_text_with_fallbacks(
        image_path=image_path,
        cache_max_age=60.0  # 60 seconds
    )

    return result.text if result.success else ""
```

### Result Quality Reporting

The manager reports result quality based on which method succeeded:

```python
result = await manager.extract_text_with_fallbacks(image_path)

# Check method used:
if result.method == "claude_vision":
    # FULL quality - Primary method
    quality = "FULL"
elif result.method.startswith("cached_"):
    # PARTIAL quality - Using cache
    quality = "PARTIAL"
elif result.method == "tesseract":
    # DEGRADED quality - Fallback method
    quality = "DEGRADED"
elif result.method == "metadata":
    # MINIMAL quality - Last resort
    quality = "MINIMAL"
else:
    # FAILED - All methods failed
    quality = "FAILED"

print(f"Quality: {quality} (method={result.method}, confidence={result.confidence:.2f})")
```

### Benefits

✅ **Intelligent Fallbacks** - Automatic progression through OCR methods
✅ **Cache Management** - Reduces API calls and cost for repeated extractions
✅ **Error Matrix Integration** - Leverages existing error handling infrastructure
✅ **Result Quality Tracking** - Know which method was used and confidence level
✅ **User-Friendly Errors** - Helpful messages when all methods fail
✅ **Flexible Configuration** - Configurable TTL and cache size
✅ **Async Support** - Fully async with timeout support
✅ **Cost Optimization** - Uses cached results when possible, falls back to free Tesseract
✅ **High Accuracy** - Claude Vision provides best-in-class OCR accuracy

### API Reference

#### OCRStrategyManager

```python
class OCRStrategyManager:
    def __init__(
        self,
        api_client: Any = None,
        cache_ttl: float = 300.0,
        max_cache_entries: int = 200,
        enable_error_matrix: bool = True
    )

    async def extract_text_with_fallbacks(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        cache_max_age: Optional[float] = None,
        skip_cache: bool = False
    ) -> OCRResult:
        """
        Extract text with intelligent fallbacks

        Returns:
            OCRResult with extracted text and metadata
        """

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""

    def clear_cache(self):
        """Clear all cache"""
```

#### OCRResult

```python
@dataclass
class OCRResult:
    success: bool                 # Whether extraction succeeded
    text: str                     # Extracted text
    confidence: float             # Confidence score (0.0-1.0)
    method: str                   # Method used
    image_hash: str               # Image hash (MD5)
    metadata: Dict[str, Any]      # Additional metadata
    error: Optional[str]          # Error message if failed
    execution_time: float         # Execution time in seconds
```

#### OCRCache

```python
class OCRCache:
    def __init__(self, default_ttl: float = 300.0, max_entries: int = 200)

    def get(
        self,
        image_hash: str,
        max_age: Optional[float] = None
    ) -> Optional[CachedOCR]:
        """Get cached OCR result by image hash"""

    def store(self, ocr_result: CachedOCR):
        """Store OCR result in cache"""

    def clear(self):
        """Clear all cache"""

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
```

#### CachedOCR

```python
@dataclass
class CachedOCR:
    text: str                     # Extracted text
    image_hash: str               # Image hash
    timestamp: datetime           # Extraction time
    method: str                   # Method used
    confidence: float             # Confidence score
    metadata: Dict[str, Any]      # Additional metadata

    def is_valid(self, max_age_seconds: float = 300.0) -> bool:
        """Check if cache is still valid"""

    def age_seconds(self) -> float:
        """Get age in seconds since extraction"""
```

#### Global Functions

```python
def get_ocr_strategy_manager() -> Optional[OCRStrategyManager]:
    """Get the global OCR strategy manager instance"""

def initialize_ocr_strategy_manager(
    api_client: Any = None,
    cache_ttl: float = 300.0,
    max_cache_entries: int = 200,
    enable_error_matrix: bool = True
) -> OCRStrategyManager:
    """Initialize the global OCR strategy manager"""
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

### v1.3 (2025-10-19)
- ✅ ErrorHandlingMatrix for graceful degradation with priority-based fallbacks
- ✅ FallbackChain for defining method execution order
- ✅ PartialResultAggregator with 4 strategies (first_success, best_result, merge, union)
- ✅ ErrorRecoveryStrategy with 3 modes (continue, retry, abort)
- ✅ ErrorMessageGenerator for user-friendly messages with suggestions
- ✅ Result quality levels (FULL, DEGRADED, PARTIAL, MINIMAL, FAILED)
- ✅ CaptureStrategyManager for intelligent capture fallbacks (window→space→cache→error)
- ✅ CaptureCache with TTL-based screenshot caching (60 seconds)
- ✅ OCRStrategyManager for intelligent OCR fallbacks (Claude→cache→Tesseract→metadata)
- ✅ OCRCache with TTL-based OCR result caching (5 minutes)
- ✅ ClaudeVisionOCR and TesseractOCR engines
- ✅ Integration with reliable_screenshot_capture.py
- ✅ Integration with multi_space_capture_engine.py
- ✅ Integration with vision/adapters/ocr.py
- ✅ Integration with claude_vision_analyzer_main.py
- ✅ Comprehensive documentation with examples and flow diagrams

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
