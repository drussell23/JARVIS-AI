# JARVIS Display Control Architecture

## Overview
JARVIS uses an intelligent hybrid approach to control display connections through macOS Control Center, combining hardcoded coordinates with vision-based detection and UAE (Unified Awareness Engine) for dynamic adaptation.

## Multi-Layer Clicking System

### 1. Factory Pattern Selection
The system uses `control_center_clicker_factory.py` to automatically select the best available clicker:

1. **UAE-Enhanced Clicker** (Best)
   - Combines context intelligence + situational awareness
   - Self-healing when UI changes
   - Learns from every interaction
   - Confidence-weighted decision making

2. **SAI-Enhanced Clicker** (Good)
   - Situational awareness only
   - Real-time visual detection

3. **Adaptive Clicker** (Standard)
   - Multi-method detection with fallbacks
   - Self-learning coordinate cache
   - Vision integration available

4. **Basic Clicker** (Fallback)
   - Simple wrapper around adaptive clicker

### 2. Detection Methods (Priority Order)

The `AdaptiveControlCenterClicker` uses these detection methods in order:

1. **Cached Coordinates** (10ms)
   - Learned from successful clicks
   - TTL-based invalidation
   - Screen resolution aware
   - Success/failure tracking

2. **Simple Heuristic** (5ms) ⭐ FAST
   - Known correct coordinates from `simple_menubar_detector.py`
   - Control Center: (1236, 12)
   - Screen Mirroring: (1396, 177)
   - Living Room TV: (1223, 115)

3. **OCR Detection** (500ms-2s)
   - Claude Vision for icons
   - Pytesseract for text
   - Screenshot region optimization

4. **Template Matching** (300ms)
   - OpenCV pattern matching
   - Pre-saved templates

5. **Edge Detection** (400ms)
   - Contour analysis
   - Shape recognition

6. **Accessibility API** (Future)
7. **AppleScript** (Future)

### 3. UAE Integration

When UAE is available, it provides:

- **Context Intelligence**: Historical patterns, time-based predictions
- **Situational Awareness**: Real-time screen analysis
- **Adaptive Integration**: Confidence-weighted fusion of methods
- **Bidirectional Learning**: Updates both context and detection cache

### 4. Verification System

After each click:
1. Screenshot comparison (before/after)
2. Content verification (expected menu items)
3. Cache update based on success
4. Learning database update

## How It Adapts When UI Changes

### Scenario: Control Center Icon Moves

1. **Initial Click** - Uses known coordinates (1236, 12)
2. **Verification Fails** - Screenshot shows no change
3. **Fallback Chain**:
   - Cache marked as failed
   - OCR/Vision detection finds new position
   - New coordinates cached
   - UAE learns the change pattern

4. **Future Clicks** - Use new cached position

### Continuous Learning

The system stores patterns in the learning database:
- Display name
- Connection success/failure
- Time of day patterns
- Coordinates used
- Detection method success rates

## Usage in Advanced Display Monitor

```python
# Get best available clicker (UAE > SAI > Adaptive > Basic)
cc_clicker = get_best_clicker(
    vision_analyzer=self.vision_analyzer,
    enable_verification=True,
    prefer_uae=True
)

# Execute complete flow
if hasattr(cc_clicker, 'connect_to_device'):
    result = await cc_clicker.connect_to_device("Living Room TV")
```

## Performance Characteristics

- **Fast Path**: Cached/known coordinates = 2 seconds total
- **Vision Fallback**: +500ms to 2s for detection
- **Success Rate**: ~100% with adaptation
- **API Calls**: 0 (when using cached/known coordinates)

## Key Benefits

1. **Speed**: Known coordinates provide instant response
2. **Reliability**: Vision fallback when coordinates fail
3. **Adaptation**: UAE learns and adapts to UI changes
4. **Efficiency**: No unnecessary API calls when coordinates work
5. **Self-Healing**: Automatically recovers from UI changes

## Files Involved

- `adaptive_control_center_clicker.py` - Core multi-method clicker
- `uae_enhanced_control_center_clicker.py` - UAE integration layer
- `control_center_clicker_factory.py` - Automatic selection logic
- `simple_menubar_detector.py` - Known coordinate storage
- `display_state_verifier.py` - Real-time connection verification
- `advanced_display_monitor.py` - Main display control logic

## Future Enhancements

1. **Template Learning**: Auto-capture templates on success
2. **Coordinate Prediction**: ML model for position changes
3. **Multi-Monitor Support**: Different coordinates per display setup
4. **Version-Specific Coordinates**: macOS version detection