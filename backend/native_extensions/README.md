# JARVIS Native C++ Extensions

This directory contains high-performance C++ extensions for JARVIS that provide significant speed improvements for critical operations.

## Extensions

### 1. Fast Capture Engine
- **Purpose**: Ultra-fast screen capture (10x faster than Python alternatives)
- **Features**:
  - Direct macOS API integration
  - Parallel window capture
  - GPU acceleration support
  - Minimal memory footprint

### 2. Vision ML Router
- **Purpose**: Lightning-fast vision command analysis (<5ms)
- **Features**:
  - Zero hardcoding pattern matching
  - Linguistic analysis in C++
  - Learning capabilities
  - Response caching

## Building

### Build All Extensions (Recommended)
```bash
./build.sh
```

### Build Specific Extension
```bash
./build.sh capture   # Build Fast Capture only
./build.sh vision    # Build Vision ML only
```

### Clean Build
```bash
./build.sh clean
```

### Build and Test
```bash
./build.sh test
```

## Requirements

### macOS
- CMake 3.15+
- C++ compiler with C++17 support
- Python 3.8+ with development headers
- pybind11 (auto-installed)

### Python Dependencies
```bash
pip install pybind11 setuptools
```

## Usage

### Fast Capture
```python
from backend.native_extensions.fast_capture_wrapper import FastCaptureEngine

engine = FastCaptureEngine()
screenshot = engine.capture_screen()
```

### Vision ML Router (C++)
```python
import vision_ml_router

# Analyze command
score, action = vision_ml_router.analyze("describe what's on my screen")
print(f"Action: {action}, Confidence: {score}")

# Learn from execution
vision_ml_router.learn("describe screen", "describe", 1)  # 1 = success
```

### Hybrid Vision Router (Recommended)
```python
from backend.voice.hybrid_vision_router import HybridVisionRouter

router = HybridVisionRouter()
intent = await router.analyze_command("what am I looking at?")
print(f"Action: {intent.final_action}, Confidence: {intent.combined_confidence}")
```

## Performance

| Operation | Python Only | With C++ | Improvement |
|-----------|-------------|----------|-------------|
| Screen Capture | 200-500ms | 20-50ms | 10x faster |
| Vision Analysis | 50-100ms | 2-5ms | 20x faster |
| Pattern Matching | 20-30ms | <1ms | 30x faster |

## Troubleshooting

### Build Failures

1. **CMake not found**
   ```bash
   brew install cmake
   ```

2. **Python headers missing**
   ```bash
   # macOS
   brew install python@3.9
   
   # Linux
   sudo apt-get install python3-dev
   ```

3. **C++ compiler issues**
   ```bash
   # Check compiler version
   g++ --version  # Should be 7.0+
   ```

### Import Errors

If extensions fail to import:
1. Check build output for errors
2. Verify `.so` or `.dylib` files exist
3. Ensure Python version matches build version
4. Try rebuilding with `./build.sh clean && ./build.sh`

### Fallback Mode

Both extensions have Python fallbacks:
- Fast Capture → Falls back to `pyautogui` or `PIL`
- Vision ML → Falls back to pure Python analysis

The system automatically uses fallbacks if C++ extensions aren't available.

## Development

### Adding New Extensions

1. Create your C++ source file
2. Add a `setup_<name>.py` for building
3. Update `build.sh` to include your extension
4. Create a Python wrapper if needed

### Testing

Run the integrated test:
```bash
./test_integrated_build.sh
```

Or test individual components:
```python
python3 -c "import fast_capture; print(fast_capture.VERSION)"
python3 -c "import vision_ml_router; print('Vision ML available')"
```

## Notes

- The C++ extensions are optional but highly recommended for performance
- Python fallbacks ensure the system works even without C++ extensions
- Build once and the extensions persist across JARVIS restarts
- Extensions are architecture-specific (Intel vs Apple Silicon)