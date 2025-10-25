# Multi-Monitor Support Documentation

## Overview

JARVIS Multi-Monitor Support enables comprehensive display detection, space mapping, and screenshot capture across multiple monitors on macOS systems. This feature extends JARVIS's vision intelligence to understand and analyze content across all connected displays.

## Features

### Core Capabilities
- **Display Detection**: Automatically detect all connected monitors using macOS Core Graphics APIs
- **Space Mapping**: Map Yabai desktop spaces to their corresponding displays
- **Screenshot Capture**: Capture screenshots from all displays simultaneously
- **Performance Monitoring**: Track capture performance and optimize operations
- **Caching**: Intelligent caching to minimize API calls and improve performance

### API Endpoints
- `GET /vision/displays` - Get information about all connected displays
- `POST /vision/displays/capture` - Capture screenshots from all displays
- `GET /vision/displays/{display_id}` - Get detailed information about a specific display
- `GET /vision/displays/performance` - Get performance statistics
- `POST /vision/displays/refresh` - Force refresh of display information

## Architecture

### Core Components

#### MultiMonitorDetector
The main class responsible for multi-monitor operations:

```python
class MultiMonitorDetector:
    async def detect_displays(self) -> List[DisplayInfo]
    async def get_space_display_mapping(self) -> Dict[int, int]
    async def capture_all_displays(self) -> MonitorCaptureResult
    async def get_display_summary(self) -> Dict[str, Any]
    def get_performance_stats(self) -> Dict[str, Any]
```

#### Data Structures

**DisplayInfo**: Information about a connected display
```python
@dataclass
class DisplayInfo:
    display_id: int
    resolution: Tuple[int, int]
    position: Tuple[int, int]
    is_primary: bool
    refresh_rate: float = 60.0
    color_depth: int = 32
    name: str = ""
    spaces: List[int] = field(default_factory=list)
    active_space: int = 1
    last_updated: float = field(default_factory=time.time)
```

**SpaceDisplayMapping**: Mapping between spaces and displays
```python
@dataclass
class SpaceDisplayMapping:
    space_id: int
    display_id: int
    space_name: str = ""
    is_active: bool = False
    last_seen: float = field(default_factory=time.time)
```

**MonitorCaptureResult**: Result of multi-monitor capture operation
```python
@dataclass
class MonitorCaptureResult:
    success: bool
    displays_captured: Dict[int, np.ndarray]
    failed_displays: List[int]
    capture_time: float
    total_displays: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Usage

### Basic Usage

```python
from backend.vision.multi_monitor_detector import MultiMonitorDetector

# Initialize detector
detector = MultiMonitorDetector()

# Detect all displays
displays = await detector.detect_displays()
print(f"Found {len(displays)} displays")

# Get space mappings
mappings = await detector.get_space_display_mapping()
print(f"Mapped {len(mappings)} spaces")

# Capture screenshots
result = await detector.capture_all_displays()
if result.success:
    print(f"Captured {len(result.displays_captured)} displays")
```

### Convenience Functions

```python
from backend.vision.multi_monitor_detector import (
    detect_all_monitors,
    capture_multi_monitor_screenshots,
    get_monitor_summary
)

# Quick display detection
displays = await detect_all_monitors()

# Quick screenshot capture
result = await capture_multi_monitor_screenshots()

# Quick summary
summary = await get_monitor_summary()
```

### CLI Usage

```bash
# Detect all displays
python jarvis_monitors.py --detect

# Detailed display information
python jarvis_monitors.py --detect --detailed

# Capture screenshots
python jarvis_monitors.py --capture

# Get display summary
python jarvis_monitors.py --summary

# Get performance statistics
python jarvis_monitors.py --performance

# Run comprehensive tests
python jarvis_monitors.py --test

# JSON output
python jarvis_monitors.py --detect --json
```

## API Reference

### GET /vision/displays

Get information about all connected displays.

**Response:**
```json
{
  "success": true,
  "displays": [
    {
      "id": 1,
      "name": "Primary Display",
      "resolution": [1920, 1080],
      "position": [0, 0],
      "is_primary": true
    }
  ],
  "total_displays": 1,
  "space_mappings": {
    "1": 1
  },
  "detection_time": 1642123456.789,
  "capture_stats": {
    "total_captures": 5,
    "successful_captures": 4,
    "failed_captures": 1,
    "average_capture_time": 0.25
  }
}
```

### POST /vision/displays/capture

Capture screenshots from all connected displays.

**Response:**
```json
{
  "success": true,
  "displays_captured": {
    "1": {
      "shape": [1080, 1920, 3],
      "dtype": "uint8",
      "size_bytes": 6220800,
      "captured": true
    }
  },
  "failed_displays": [],
  "capture_time": 0.25,
  "total_displays": 1,
  "error": null,
  "metadata": {
    "capture_method": "core_graphics",
    "displays_info": {
      "1": {
        "resolution": [1920, 1080],
        "position": [0, 0],
        "is_primary": true
      }
    }
  }
}
```

### GET /vision/displays/{display_id}

Get detailed information about a specific display.

**Response:**
```json
{
  "success": true,
  "display": {
    "id": 1,
    "name": "Primary Display",
    "resolution": [1920, 1080],
    "position": [0, 0],
    "is_primary": true,
    "refresh_rate": 60.0,
    "color_depth": 32,
    "spaces": [1, 2],
    "active_space": 1,
    "last_updated": 1642123456.789
  }
}
```

### GET /vision/displays/performance

Get performance statistics for multi-monitor operations.

**Response:**
```json
{
  "success": true,
  "performance": {
    "capture_stats": {
      "total_captures": 5,
      "successful_captures": 4,
      "failed_captures": 1,
      "average_capture_time": 0.25
    },
    "cache_info": {
      "displays_cached": 2,
      "space_mappings_cached": 4,
      "last_detection_time": 1642123456.789,
      "cache_age_seconds": 2.5
    },
    "system_info": {
      "macos_available": true,
      "yabai_path": "yabai"
    }
  }
}
```

## Configuration

### Environment Variables

- `YABAI_PATH`: Path to yabai executable (default: "yabai")
- `DISPLAY_CACHE_DURATION`: Cache duration in seconds (default: 5.0)

### Dependencies

- **macOS Core Graphics**: Required for display detection and screenshot capture
- **PyObjC**: Python bindings for Objective-C frameworks
- **Yabai**: Window manager for space detection (optional)
- **NumPy**: For image data handling

## Performance Considerations

### Caching Strategy
- Display information is cached for 5 seconds by default
- Space mappings are cached separately
- Force refresh available when needed

### Capture Optimization
- Parallel capture across multiple displays
- Intelligent fallback for failed captures
- Performance statistics tracking

### Memory Usage
- Screenshots stored as NumPy arrays
- ~30-50 MB per monitor for cached screenshots
- Automatic cleanup of old cache entries

## Error Handling

### Common Issues

1. **macOS Frameworks Not Available**
   - Error: `macOS frameworks not available`
   - Solution: Ensure running on macOS with PyObjC installed

2. **Permission Denied**
   - Error: `Permission denied for screen capture`
   - Solution: Grant screen recording permissions in System Preferences

3. **Yabai Not Found**
   - Error: `Yabai CLI unavailable`
   - Solution: Install Yabai or use Core Graphics fallback

4. **Display Detection Failed**
   - Error: `No displays detected`
   - Solution: Check display connections and Core Graphics availability

### Graceful Degradation
- Falls back to Core Graphics when Yabai is unavailable
- Returns partial results when some displays fail
- Continues operation with available displays

## Testing

### Unit Tests
```bash
python -m pytest tests/test_multi_monitor_detector.py -v
```

### Integration Tests
```bash
python test_multi_monitor_integration.py
```

### CLI Tests
```bash
python jarvis_monitors.py --test
```

## Troubleshooting

### Debug Mode
Enable debug logging to see detailed operation information:

```python
import logging
logging.getLogger('backend.vision.multi_monitor_detector').setLevel(logging.DEBUG)
```

### Common Debug Commands

```bash
# Check display detection
python jarvis_monitors.py --detect --detailed

# Test capture functionality
python jarvis_monitors.py --capture

# Check performance
python jarvis_monitors.py --performance

# Run comprehensive tests
python jarvis_monitors.py --test
```

### System Requirements
- macOS 10.14+ (for Core Graphics APIs)
- Python 3.8+
- PyObjC framework
- Screen recording permissions
- Yabai (optional, for space detection)

## Future Enhancements

### Planned Features
- **Cross-Display Attention Tracking**: Track cursor/gaze across screens
- **Display-Context Memory**: Remember what each monitor is used for
- **Multi-Display Vision Fusion**: Aggregate context from all monitors
- **Virtual Monitor Emulation**: Simulate additional spaces for testing

### Performance Improvements
- **Async Capture Pipeline**: Non-blocking screenshot operations
- **Smart Caching**: Intelligent cache invalidation
- **Compression**: Image compression for reduced memory usage
- **Batch Operations**: Batch multiple operations for efficiency

## Contributing

### Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install PyObjC: `pip install pyobjc-framework-Quartz pyobjc-framework-AppKit`
4. Grant screen recording permissions
5. Run tests: `python test_multi_monitor_integration.py`

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Include unit tests for new features

---

*Generated: 2025-01-14*
*Branch: multi-monitor-support*
