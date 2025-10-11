# Advanced Core Graphics Window Capture System

## Overview

A highly advanced, robust, and dynamic window capture system with **ZERO hardcoding**. All configuration is handled through environment variables or runtime parameters.

## Features

### 🎯 Core Capabilities
- **Multi-Strategy Matching**: Exact, Contains, Fuzzy, Regex, or Custom matchers
- **Intelligent Fallback**: 5 capture strategies with automatic fallback
- **Performance Optimization**: Built-in caching with configurable TTL
- **Comprehensive Filtering**: Layer, size, alpha, visibility filters
- **Statistics Tracking**: Capture success rates, timing, cache hits
- **Backward Compatible**: Drop-in replacement for legacy `CGWindowCapture`

### 🚀 Key Improvements Over Legacy System

| Feature | Legacy | Advanced |
|---------|--------|----------|
| Window Matching | Hardcoded substring | 5 strategies (exact, fuzzy, regex, custom) |
| Capture Methods | 3 hardcoded options | 5 strategies with quality presets |
| Configuration | Hardcoded | Environment variables + runtime |
| Error Handling | Basic try/catch | Comprehensive with retry logic |
| Performance | No caching | LRU cache with TTL |
| Statistics | None | Full metrics tracking |
| Filtering | Minimal | Layer, size, alpha, visibility |

## Usage

### Quick Start (Legacy Compatible)

```python
from backend.vision.cg_window_capture import CGWindowCapture

# Get all windows
windows = CGWindowCapture.get_all_windows()

# Find window by name
window_id = CGWindowCapture.find_window_by_name("Terminal")

# Capture window
screenshot = CGWindowCapture.capture_window_by_id(window_id)

# Capture all app windows
captures = CGWindowCapture.capture_app_windows("Terminal")
```

### Advanced Usage

```python
from backend.vision.cg_window_capture import (
    AdvancedCGWindowCapture,
    CaptureConfig,
    CaptureQuality,
    WindowMatchStrategy
)

# Create custom configuration
config = CaptureConfig(
    quality=CaptureQuality.HIGH,
    match_strategy=WindowMatchStrategy.FUZZY,
    fuzzy_threshold=0.7,
    cache_ttl=10,
    min_window_size=(200, 200)
)

# Initialize engine
engine = AdvancedCGWindowCapture(config)

# Find windows with scoring
windows = engine.find_windows(
    "Terminal",
    strategy=WindowMatchStrategy.FUZZY,
    min_score=0.6
)

# Capture with quality control
result = engine.capture_window(windows[0].id, quality=CaptureQuality.HIGH)

if result.success:
    print(f"Captured in {result.capture_time:.3f}s using {result.method_used}")
    screenshot = result.screenshot
else:
    print(f"Failed: {result.error}")
```

### Custom Matcher

```python
def my_custom_matcher(window):
    """Custom window matching logic"""
    if "Terminal" in window.owner and window.width > 1000:
        return 1.0  # Perfect match
    elif "Terminal" in window.owner:
        return 0.7  # Partial match
    return 0.0  # No match

windows = engine.find_windows(
    "Terminal",
    strategy=WindowMatchStrategy.CUSTOM,
    custom_matcher=my_custom_matcher
)
```

## Configuration

### Environment Variables

All configuration can be set via environment variables - **ZERO hardcoding**:

```bash
# Capture behavior
export CG_CAPTURE_RETRY_COUNT=3
export CG_CAPTURE_RETRY_DELAY=0.1
export CG_CAPTURE_TIMEOUT=5.0

# Window filtering
export CG_MIN_WINDOW_WIDTH=100
export CG_MIN_WINDOW_HEIGHT=100
export CG_MAX_WINDOW_WIDTH=10000
export CG_MAX_WINDOW_HEIGHT=10000
export CG_ALLOWED_LAYERS=0           # Comma-separated
export CG_MIN_ALPHA=0.1

# Performance
export CG_CACHE_TTL=5                 # Seconds
export CG_ENABLE_CACHE=true

# Matching
export CG_FUZZY_THRESHOLD=0.6         # 0.0 - 1.0

# Advanced
export CG_INCLUDE_OFFSCREEN=true
```

### Runtime Configuration

```python
config = CaptureConfig(
    quality=CaptureQuality.BALANCED,
    retry_count=5,
    retry_delay=0.2,
    min_window_size=(150, 150),
    max_window_size=(8000, 8000),
    allowed_layers=[0],
    min_alpha=0.2,
    cache_ttl=10,
    enable_cache=True,
    match_strategy=WindowMatchStrategy.FUZZY,
    fuzzy_threshold=0.7,
    include_offscreen=True,
    capture_timeout=5.0
)
```

## Window Matching Strategies

### 1. EXACT
Perfect string match (case-insensitive)
```python
windows = engine.find_windows("Terminal", strategy=WindowMatchStrategy.EXACT)
```

### 2. CONTAINS
Substring match (default legacy behavior)
```python
windows = engine.find_windows("Term", strategy=WindowMatchStrategy.CONTAINS)
```

### 3. FUZZY (Recommended)
Similarity scoring with Levenshtein-inspired algorithm
```python
windows = engine.find_windows(
    "Terminl",  # Typo!
    strategy=WindowMatchStrategy.FUZZY,
    min_score=0.6
)
# Still finds "Terminal" with score ~0.8
```

### 4. REGEX
Regular expression matching
```python
windows = engine.find_windows(
    r"^Terminal.*",  # Pattern
    strategy=WindowMatchStrategy.REGEX
)
```

### 5. CUSTOM
Your own matching function
```python
def prioritize_large_windows(window):
    score = 0.0
    if "Terminal" in window.owner:
        score = 0.5
    if window.area > 1000000:
        score += 0.5
    return score

windows = engine.find_windows(
    "Terminal",
    strategy=WindowMatchStrategy.CUSTOM,
    custom_matcher=prioritize_large_windows
)
```

## Capture Quality Presets

### FAST
Quick capture, may sacrifice quality
- Tries: `default`, `no_framing`
- Best for: Real-time monitoring

### BALANCED (Default)
Balance between speed and quality
- Tries: `default`, `no_framing`, `opaque`
- Best for: General purpose

### HIGH
High quality capture
- Tries: All strategies except `best_resolution`
- Best for: Important captures

### MAXIMUM
Maximum quality, all strategies
- Tries: All 5 capture strategies
- Best for: Critical captures

## Capture Strategies (Automatic Fallback)

The system tries multiple capture strategies in order:

1. **default**: Standard capture with all effects
2. **no_framing**: Capture without window frame/shadow
3. **opaque**: Force opaque rendering
4. **combined**: Combine no-framing and opaque
5. **best_resolution**: Best available resolution

## WindowInfo Object

Enhanced window information with computed properties:

```python
@dataclass
class WindowInfo:
    id: int                    # Window ID
    name: str                  # Window title
    owner: str                 # Application name
    bounds: Dict[str, float]   # Window bounds
    layer: WindowLayer         # Window layer enum
    alpha: float               # Window transparency
    on_screen: bool            # Currently visible
    workspace: Optional[int]   # Desktop space ID
    pid: int                   # Process ID
    memory_usage: int          # Memory usage

    # Computed properties
    width: int                 # Window width
    height: int                # Window height
    area: int                  # Total area
    center_x: float            # Center X coordinate
    center_y: float            # Center Y coordinate
    score: float               # Match score (0.0-1.0)
```

## CaptureResult Object

Comprehensive capture result with metadata:

```python
@dataclass
class CaptureResult:
    success: bool                      # Capture succeeded
    window_id: int                     # Window ID
    screenshot: Optional[np.ndarray]   # Screenshot data (RGB)
    width: int                         # Image width
    height: int                        # Image height
    capture_time: float                # Time taken (seconds)
    method_used: str                   # Strategy used
    error: Optional[str]               # Error message if failed
    metadata: Dict[str, Any]           # Additional metadata
```

## Statistics

Track capture performance:

```python
stats = engine.get_statistics()
print(stats)
# {
#     'total_captures': 100,
#     'successful_captures': 98,
#     'failed_captures': 2,
#     'success_rate': 0.98,
#     'average_capture_time': 0.143,
#     'cache_hits': 45
# }
```

## Advanced Features

### Caching

Automatically caches window lists for configurable TTL:

```python
# First call - fetches from Core Graphics
windows = engine.get_all_windows()  # ~100ms

# Second call within cache_ttl - instant from cache
windows = engine.get_all_windows()  # ~1ms

# Force refresh
windows = engine.get_all_windows(force_refresh=True)
```

### Filtering

Dynamic filtering based on configuration:

```python
config = CaptureConfig(
    allowed_layers=[0],           # Only normal windows
    min_alpha=0.5,                # No transparent windows
    min_window_size=(200, 200),   # Minimum size
    include_offscreen=False       # Only visible windows
)
```

### Batch Capture

Capture multiple windows efficiently:

```python
# Capture first 5 Terminal windows
results = engine.capture_app_windows("Terminal", max_windows=5)

for window_id, result in results.items():
    if result.success:
        print(f"Window {window_id}: {result.width}x{result.height}")
```

## Error Handling

Comprehensive error handling with detailed messages:

```python
result = engine.capture_window(window_id)

if not result.success:
    print(f"Capture failed: {result.error}")
    print(f"Tried method: {result.method_used}")
    print(f"Time spent: {result.capture_time:.3f}s")
```

## Performance Tips

1. **Enable Caching**: Significantly faster for repeated queries
2. **Use Fuzzy Matching**: More reliable than exact matching
3. **Set Appropriate Quality**: FAST for monitoring, HIGH for analysis
4. **Filter Early**: Use `allowed_layers` and size filters
5. **Batch Operations**: Use `capture_app_windows()` for multiple windows

## Troubleshooting

### "Failed to capture window"
- **Cause**: Missing Screen Recording permission
- **Solution**: System Settings > Privacy & Security > Screen Recording > Enable for Python

### "Window not found"
- **Cause**: Matching strategy too strict
- **Solution**: Use `WindowMatchStrategy.FUZZY` with lower threshold

### Slow Performance
- **Cause**: Cache disabled or expired
- **Solution**: Increase `cache_ttl` or enable caching

### Empty Screenshot
- **Cause**: Window minimized or obscured
- **Solution**: Check `window.on_screen` before capturing

## Integration with JARVIS

The advanced system is automatically used by JARVIS multi-space capture:

```python
# In multi_space_capture_engine.py
from .cg_window_capture import CGWindowCapture

# Legacy API still works
screenshot = CGWindowCapture.capture_window_by_id(window_id)

# Or use advanced features
from .cg_window_capture import get_capture_engine, CaptureConfig

engine = get_capture_engine(CaptureConfig(quality=CaptureQuality.HIGH))
result = engine.capture_window(window_id)
```

## Migration from Legacy

The new system is 100% backward compatible:

```python
# Old code - still works
from backend.vision.cg_window_capture import CGWindowCapture
screenshot = CGWindowCapture.capture_window_by_id(window_id)

# New code - enhanced features
from backend.vision.cg_window_capture import get_capture_engine
engine = get_capture_engine()
result = engine.capture_window(window_id)
screenshot = result.screenshot
```

## Examples

### Find All Chrome Windows
```python
chrome_windows = engine.find_windows(
    "Chrome",
    strategy=WindowMatchStrategy.FUZZY,
    min_score=0.7
)

for w in chrome_windows:
    print(f"{w.name} - Score: {w.score:.2f}")
```

### Capture Largest Window
```python
windows = engine.find_windows("Terminal")
largest = max(windows, key=lambda w: w.area)
result = engine.capture_window(largest.id)
```

### Capture with Retry
```python
config = CaptureConfig(retry_count=5, retry_delay=0.2)
engine = AdvancedCGWindowCapture(config)
result = engine.capture_window(window_id, quality=CaptureQuality.MAXIMUM)
```

### Filter by Size
```python
config = CaptureConfig(
    min_window_size=(800, 600),
    max_window_size=(2000, 1500)
)
engine = AdvancedCGWindowCapture(config)
windows = engine.get_all_windows()  # Only windows in size range
```

## Architecture

```
AdvancedCGWindowCapture
├── Window Discovery (Core Graphics)
│   ├── get_all_windows() → List[WindowInfo]
│   └── Caching Layer (configurable TTL)
│
├── Window Matching (Multiple Strategies)
│   ├── Exact Match
│   ├── Contains Match
│   ├── Fuzzy Match (Levenshtein)
│   ├── Regex Match
│   └── Custom Matcher (user-defined)
│
├── Window Filtering
│   ├── Layer Filter
│   ├── Alpha Filter
│   ├── Size Filter
│   └── Visibility Filter
│
├── Capture Engine (5 Fallback Strategies)
│   ├── Default
│   ├── No Framing
│   ├── Opaque
│   ├── Combined
│   └── Best Resolution
│
└── Statistics & Monitoring
    ├── Success Rate Tracking
    ├── Performance Metrics
    └── Cache Hit Rate
```

## License

Part of the JARVIS AI Assistant system. All configuration is dynamic with zero hardcoding.
