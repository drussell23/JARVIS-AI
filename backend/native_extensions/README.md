# JARVIS Fast Capture C++ Extension

High-performance screen capture engine for JARVIS Vision System, providing 10x faster capture speeds compared to pure Python implementations.

## Features

- **Lightning Fast**: 10-50ms per window capture (vs 100-500ms with Python)
- **Multi-Window Capture**: Parallel capture of multiple windows simultaneously
- **GPU Acceleration**: Hardware-accelerated capture when available
- **Zero Copy**: Efficient memory management with minimal overhead
- **Dynamic Discovery**: No hardcoded values - everything is discovered at runtime
- **Thread Safe**: Can be used from multiple Python threads
- **Fallback Support**: Seamless fallback to Python if C++ extension unavailable

## Performance Comparison

| Operation | Python (Quartz) | C++ Fast Capture | Improvement |
|-----------|----------------|------------------|-------------|
| Single Window | ~300ms | ~30ms | 10x faster |
| 10 Windows | ~3000ms | ~150ms | 20x faster |
| Full Screen | ~500ms | ~50ms | 10x faster |
| Memory Usage | ~200MB | ~50MB | 4x less |

## Building

### Prerequisites

- macOS 10.14 or later
- Python 3.7+
- CMake 3.12+
- Xcode Command Line Tools
- pybind11 (installed automatically)

### Quick Build

```bash
cd backend/native_extensions
./build.sh
```

### Manual Build

```bash
# Install dependencies
pip install pybind11

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make -j4

# Install
make install
```

### Clean Build

```bash
./build.sh clean
```

## Usage

### Drop-in Replacement

The enhanced vision system is a drop-in replacement for the existing Python implementation:

```python
# Simply import the enhanced version instead
from backend.vision.enhanced_screen_vision import EnhancedScreenVisionSystem

# Use exactly like the original
vision = EnhancedScreenVisionSystem()
image = await vision.capture_screen()
```

### Direct C++ Extension Usage

For maximum control and performance:

```python
from backend.native_extensions.fast_capture_wrapper import FastCaptureEngine, CaptureConfig

# Initialize engine
engine = FastCaptureEngine()

# Configure capture
config = CaptureConfig(
    output_format="raw",
    use_gpu_acceleration=True,
    parallel_capture=True
)

# Single window capture
result = engine.capture_frontmost_window(config)
if result['success']:
    image = result['image']  # numpy array
    print(f"Captured in {result['capture_time_ms']}ms")

# Multi-window capture (parallel)
windows = engine.capture_all_windows(config)
print(f"Captured {len(windows)} windows")

# Get performance metrics
metrics = engine.get_metrics()
print(f"Average capture time: {metrics['avg_capture_time_ms']}ms")
```

### Async Usage

```python
# Async single window
image = await vision.capture_screen()

# Async multi-window
captures = await vision.capture_multiple_windows(
    app_names=['Safari', 'Chrome'],
    visible_only=True
)
```

## API Reference

### CaptureConfig

Configuration object for capture operations:

- `capture_cursor` (bool): Include cursor in capture
- `capture_shadow` (bool): Include window shadows
- `output_format` (str): "auto", "jpeg", "png", "raw"
- `jpeg_quality` (int): JPEG compression quality (0-100)
- `use_gpu_acceleration` (bool): Enable GPU acceleration
- `parallel_capture` (bool): Enable parallel multi-window capture
- `max_width` (int): Maximum capture width (0 = no limit)
- `max_height` (int): Maximum capture height (0 = no limit)

### FastCaptureEngine Methods

- `capture_window(window_id)`: Capture specific window by ID
- `capture_window_by_name(app_name, window_title)`: Capture by app/title
- `capture_frontmost_window()`: Capture the active window
- `capture_all_windows()`: Capture all windows in parallel
- `capture_visible_windows()`: Capture only visible windows
- `get_all_windows()`: Get window information without capturing
- `get_metrics()`: Get performance statistics

## Architecture

```
┌─────────────────────────────────────┐
│         Python Application          │
├─────────────────────────────────────┤
│    Enhanced Screen Vision System    │
│         (Python Wrapper)            │
├─────────────────────────────────────┤
│       Fast Capture Wrapper          │
│      (Python/C++ Bridge)            │
├─────────────────────────────────────┤
│      C++ Fast Capture Engine        │
│  (High-Performance Implementation)  │
├─────────────────────────────────────┤
│        macOS Core Graphics          │
│         (System APIs)               │
└─────────────────────────────────────┘
```

## Troubleshooting

### Import Error

If you get an import error:

1. Ensure the extension is built: `./build.sh`
2. Check Python path includes the native_extensions directory
3. Verify the .so file exists in the directory

### Permission Errors

Screen recording permission is required on macOS:
1. Go to System Preferences → Security & Privacy → Privacy
2. Select "Screen Recording"
3. Add your Terminal/Python to the allowed list

### Performance Issues

1. Ensure Release build: `cmake -DCMAKE_BUILD_TYPE=Release`
2. Check GPU acceleration is enabled
3. Monitor system resources during capture

## Development

### Adding New Features

1. Modify the C++ implementation in `src/fast_capture.cpp`
2. Update Python bindings in `src/python_bindings.cpp`
3. Add Python wrapper methods in `fast_capture_wrapper.py`
4. Update the enhanced vision system if needed
5. Rebuild: `./build.sh`

### Running Tests

```bash
# Run performance comparison
./build.sh test

# Or manually
cd backend/vision
python test_enhanced_vision.py
```

### Debugging

Build with debug symbols:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

## Future Enhancements

- [ ] Metal API integration for better GPU acceleration
- [ ] Video capture support
- [ ] Window change detection
- [ ] OCR integration at C++ level
- [ ] Cross-platform support (Windows/Linux)

## License

Part of the JARVIS AI Agent project.