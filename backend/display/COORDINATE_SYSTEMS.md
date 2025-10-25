# macOS Coordinate Systems - Complete Guide

## The Problem

macOS Retina displays use **TWO different coordinate systems** that are easily confused:

### 1. Physical Pixels (Screenshot Space)
- **What**: The actual hardware pixels on the display
- **Example**: 2880 x 1800 on a MacBook Pro 14" Retina display
- **Used by**:
  - `screencapture` command
  - Screenshots from Retina displays
  - PIL/Pillow Image objects from screenshots
  - Claude Vision (analyzes physical pixel screenshots)

### 2. Logical Pixels (PyAutoGUI/UI Space)
- **What**: The "virtual" coordinate system macOS uses for UI
- **Example**: 1440 x 900 (exactly half of physical)
- **Used by**:
  - PyAutoGUI (`moveTo`, `dragTo`, `click`)
  - macOS UI frameworks
  - AppleScript coordinates
  - All mouse/cursor APIs

### DPI Scale Factor
- **Retina 2x displays**: `backingScaleFactor = 2.0`
- **Standard displays**: `backingScaleFactor = 1.0`

## The Critical Rule

```
Logical Pixels = Physical Pixels ÷ DPI Scale Factor
Physical Pixels = Logical Pixels × DPI Scale Factor
```

On 2x Retina display:
```
Logical (1235, 10) = Physical (2470, 20) ÷ 2.0
Physical (2470, 20) = Logical (1235, 10) × 2.0
```

## Common Bug Patterns

### ❌ Bug #1: Using Physical Pixels with PyAutoGUI
```python
screenshot = pyautogui.screenshot()  # 2880x1800 (physical pixels)
# Claude Vision returns: x=2470, y=20 (physical pixels)
pyautogui.moveTo(2470, 20)  # ❌ WRONG! Mouse goes off-screen
```

**Why it fails**: PyAutoGUI expects logical pixels (1440x900), but receives physical pixels (2880x1800).

### ✅ Fix #1: Convert to Logical Pixels
```python
from AppKit import NSScreen

screenshot = pyautogui.screenshot()  # 2880x1800 (physical)
# Claude Vision returns: x=2470, y=20 (physical)

# Convert to logical
dpi_scale = NSScreen.mainScreen().backingScaleFactor()  # 2.0
logical_x = int(2470 / dpi_scale)  # 1235
logical_y = int(20 / dpi_scale)    # 10

pyautogui.moveTo(logical_x, logical_y)  # ✅ CORRECT!
```

### ❌ Bug #2: Double Conversion
```python
# Coordinates already in logical pixels
logical_coords = (1235, 10)

# Converting again (WRONG!)
converted = (logical_coords[0] / 2.0, logical_coords[1] / 2.0)  # (617, 5) ❌
pyautogui.moveTo(converted[0], converted[1])  # Mouse goes to wrong place
```

**Why it fails**: Coordinates were already in logical space, dividing again makes them wrong.

### ✅ Fix #2: Track Coordinate Space
```python
# Always document what space coordinates are in
physical_coords = (2470, 20)  # From screenshot/vision
logical_coords = (1235, 10)   # For PyAutoGUI

# Only convert when needed
if coord_space == "physical":
    logical_coords = (physical_coords[0] / dpi_scale,
                      physical_coords[1] / dpi_scale)

pyautogui.moveTo(logical_coords[0], logical_coords[1])  # ✅ CORRECT!
```

## Detection Methods & Coordinate Spaces

| Method | Returns | Needs Conversion? |
|--------|---------|-------------------|
| Cached coordinates (JSON) | Logical | ❌ No |
| Hardcoded coordinates | Logical | ❌ No |
| PyAutoGUI screenshot + Vision | Physical | ✅ Yes |
| `screencapture` + Vision | Physical | ✅ Yes |
| AppleScript | Logical | ❌ No |

## Best Practices

### 1. **Use Hardcoded Logical Pixels (Simplest)**
```python
# Known UI element positions in logical pixels
CONTROL_CENTER = (1235, 10)
SCREEN_MIRRORING = (1396, 177)
LIVING_ROOM_TV = (1223, 115)

# Use directly with PyAutoGUI
pyautogui.dragTo(CONTROL_CENTER[0], CONTROL_CENTER[1])
```

**Pros**:
- No conversion needed
- Fast and reliable
- No DPI confusion

**Cons**:
- Breaks if UI layout changes
- Needs updating if screen resolution changes

### 2. **Use Vision for Verification Only**
```python
# Primary: Use known logical coordinates
pyautogui.moveTo(1235, 10)

# Optional: Verify with vision
screenshot = take_screenshot()  # Physical pixels
vision_coords = claude_vision.find("Control Center")  # Physical pixels
vision_logical = (vision_coords[0] / dpi_scale, vision_coords[1] / dpi_scale)

# Check if vision agrees (within tolerance)
if abs(vision_logical[0] - 1235) < 50:  # 50px tolerance
    print("✅ Vision confirms position")
else:
    print("⚠️ UI may have moved, using vision coordinates")
    pyautogui.moveTo(vision_logical[0], vision_logical[1])
```

### 3. **Always Convert Vision Coordinates**
```python
def convert_to_logical(physical_x: int, physical_y: int) -> tuple:
    """Convert physical pixels to logical pixels"""
    from AppKit import NSScreen
    dpi_scale = NSScreen.mainScreen().backingScaleFactor()
    return (int(physical_x / dpi_scale), int(physical_y / dpi_scale))

# Use everywhere vision returns coordinates
screenshot = pyautogui.screenshot()  # 2880x1800
physical_coords = claude_vision.find_element(screenshot)  # (2470, 20)
logical_coords = convert_to_logical(physical_coords[0], physical_coords[1])  # (1235, 10)
pyautogui.moveTo(logical_coords[0], logical_coords[1])  # ✅ CORRECT
```

## JARVIS Display Connection Flow

**Recommended Approach**: Hardcoded logical coordinates + optional vision verification

```python
# Step 1: Click Control Center (hardcoded logical pixels)
pyautogui.dragTo(1235, 10)  # Control Center icon
time.sleep(0.5)

# Step 2: Click Screen Mirroring (hardcoded logical pixels)
pyautogui.moveTo(1396, 177)  # Screen Mirroring menu item
pyautogui.click()
time.sleep(0.5)

# Step 3: Click Living Room TV (hardcoded logical pixels)
pyautogui.moveTo(1223, 115)  # Living Room TV option
pyautogui.click()
```

**Why this works**:
- ✅ No coordinate conversion needed
- ✅ Fast and reliable
- ✅ No DPI confusion
- ✅ Works on 1x and 2x displays (as long as UI positions are same)

**Optional enhancement**: Use vision to verify positions before clicking:
```python
def verify_and_click(name: str, expected_logical: tuple) -> bool:
    """Verify position with vision, then click"""
    # Take screenshot (physical pixels)
    screenshot = pyautogui.screenshot()

    # Find with vision (returns physical pixels)
    physical_coords = claude_vision.find(name, screenshot)

    if physical_coords:
        # Convert to logical
        logical_coords = convert_to_logical(physical_coords[0], physical_coords[1])

        # Check if close to expected
        distance = math.sqrt(
            (logical_coords[0] - expected_logical[0])**2 +
            (logical_coords[1] - expected_logical[1])**2
        )

        if distance < 50:  # Within 50 pixels
            print(f"✅ Vision confirms {name} at expected position")
            pyautogui.moveTo(expected_logical[0], expected_logical[1])
        else:
            print(f"⚠️ {name} moved, using vision coordinates")
            pyautogui.moveTo(logical_coords[0], logical_coords[1])

        pyautogui.click()
        return True
    else:
        print(f"⚠️ Vision couldn't find {name}, using expected position")
        pyautogui.moveTo(expected_logical[0], expected_logical[1])
        pyautogui.click()
        return True
```

## Testing Coordinate Systems

```python
import pyautogui
from AppKit import NSScreen

# Test 1: Check coordinate systems
print(f"PyAutoGUI screen size (logical): {pyautogui.size()}")
screenshot = pyautogui.screenshot()
print(f"Screenshot size (physical): {screenshot.size}")

dpi_scale = NSScreen.mainScreen().backingScaleFactor()
print(f"DPI scale factor: {dpi_scale}x")

expected_logical = (screenshot.size[0] / dpi_scale, screenshot.size[1] / dpi_scale)
print(f"Expected logical size: {expected_logical}")

assert pyautogui.size() == expected_logical, "Coordinate system mismatch!"

# Test 2: Verify mouse movement
test_logical = (1235, 10)
pyautogui.moveTo(test_logical[0], test_logical[1])
actual = pyautogui.position()
assert actual.x == test_logical[0] and actual.y == test_logical[1], "Mouse movement failed!"
print("✅ All coordinate system tests passed")
```

## Edge Cases

### Multi-Monitor Setups
- Each monitor may have different DPI scale
- Coordinates may be offset by other monitors' dimensions
- Use `NSScreen.screens()` to get all displays and their scales

### UI Scaling Changes
- User may change display scaling in System Preferences
- Hardcoded coordinates will break
- Solution: Re-calibrate coordinates or use vision

### Dark Mode vs Light Mode
- UI elements may move slightly
- Colors change (affects vision detection)
- Solution: Test in both modes

### macOS Version Differences
- Control Center position changed in macOS 11+
- Menu bar layout may differ
- Solution: Version-specific coordinate sets

## Summary

**Golden Rules**:
1. **Screenshots = Physical pixels** → Always divide by DPI scale before using with PyAutoGUI
2. **PyAutoGUI = Logical pixels** → Use directly, no conversion needed
3. **Hardcoded coordinates = Logical pixels** → Fastest and most reliable
4. **Vision = Physical pixels** → Always convert to logical before clicking
5. **When in doubt**: Log both physical and logical coordinates to debug

**For JARVIS Display Connection**:
- ✅ **Primary**: Use hardcoded logical pixel coordinates (1235,10) → (1396,177) → (1223,115)
- ✅ **Optional**: Vision for verification only
- ❌ **Avoid**: Using vision as primary coordinate source (too complex, error-prone)
