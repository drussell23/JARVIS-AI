# Edge Case Handling System - Complete Documentation

**Version:** 1.0
**Last Updated:** 2025-10-19
**Status:** ✅ Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Space-Related Edge Cases](#space-related-edge-cases)
3. [Window Capture Edge Cases](#window-capture-edge-cases)
4. [Integration Points](#integration-points)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The Edge Case Handling System provides comprehensive, robust handling for all macOS space and window capture edge cases. It ensures JARVIS can gracefully handle failures, provide helpful error messages, and automatically retry or fallback when needed.

### Key Features

✅ **Fully Async** - All operations use `asyncio` for non-blocking execution
✅ **Dynamic** - No hardcoded space IDs, window IDs, or image sizes
✅ **Robust** - Retry logic with exponential backoff and automatic fallback
✅ **Natural Language Responses** - User-friendly error messages
✅ **Comprehensive Metadata** - Detailed information about what happened
✅ **Zero Dependencies** - Uses native macOS tools (yabai, screencapture, sips)

### Architecture

```
User Request
    ↓
Intent Analyzer
    ↓
┌─────────────────────────────────────────────┐
│  Edge Case Validation                       │
│  ├── SpaceStateManager (space validation)   │
│  └── WindowCaptureManager (window capture)  │
└─────────────────────────────────────────────┘
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
