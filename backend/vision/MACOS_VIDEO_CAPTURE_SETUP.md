# macOS Native Video Capture Setup Guide

This guide will help you enable native macOS video capture with the purple screen recording indicator for JARVIS Vision System.

## 🟣 What is Native Video Capture?

Native macOS video capture uses AVFoundation to record your screen at 30 FPS, showing the purple screen recording indicator in your menu bar. This is more efficient and provides better integration than fallback methods.

## 📋 Prerequisites

- macOS 10.14 (Mojave) or later
- Python 3.8+
- Screen recording permissions granted to Terminal/your IDE

## 🚀 Quick Setup

```bash
# Option 1: Install all pyobjc frameworks (recommended)
pip3 install pyobjc

# Option 2: Install only required frameworks
pip3 install pyobjc-framework-AVFoundation \
             pyobjc-framework-CoreMedia \
             pyobjc-framework-libdispatch \
             pyobjc-framework-Cocoa \
             pyobjc-framework-Quartz
```

## 🔧 Detailed Installation

### 1. Using the Installation Script

```bash
cd backend/vision
bash install_macos_video_frameworks.sh
```

This script will:
- Install all required frameworks
- Verify the installation
- Show if native capture is available

### 2. Manual Installation

```bash
# Update pip first
pip3 install --upgrade pip

# Install core framework
pip3 install pyobjc-core

# Install required frameworks
pip3 install pyobjc-framework-AVFoundation  # Screen capture API
pip3 install pyobjc-framework-CoreMedia     # Media processing
pip3 install pyobjc-framework-Cocoa         # macOS integration
pip3 install pyobjc-framework-Quartz        # CoreVideo support
pip3 install pyobjc-framework-libdispatch   # Grand Central Dispatch
```

### 3. Verify Installation

```python
# Test if frameworks are available
python3 -c "
import AVFoundation
import CoreMedia
from Quartz import CoreVideo
from Cocoa import NSObject
import libdispatch
print('✅ All frameworks loaded successfully!')
print('🟣 Native video capture is ready!')
"
```

## 🔐 Grant Screen Recording Permission

1. Open **System Preferences** → **Security & Privacy** → **Privacy**
2. Select **Screen Recording** from the left sidebar
3. Click the lock to make changes
4. Add Terminal (or your IDE) to the allowed apps
5. Restart Terminal/IDE for permissions to take effect

## 🧪 Test Native Capture

### Simple Test
```bash
cd backend/vision
python3 test_video_simple.py
```

### Purple Indicator Test (30 seconds)
```bash
python3 test_purple_indicator.py
```

This will:
- Start native video capture
- Show the purple indicator for 30 seconds
- Stop capture and remove the indicator

## 🔍 Verify Native Mode

When running tests, look for:
```
✅ Video streaming started successfully!
   Capture method: macos_native
   🟣 Check for purple screen recording indicator on macOS
```

## 🛠️ Troubleshooting

### Purple indicator not appearing?

1. **Check capture method**: Should show `macos_native`, not `fallback`
2. **Verify permissions**: Ensure screen recording is allowed
3. **Test frameworks**:
   ```bash
   python3 -c "import AVFoundation; print('AVFoundation OK')"
   ```

### ImportError: No module named 'AVFoundation'?

```bash
# Reinstall with system python
/usr/bin/python3 -m pip install pyobjc-framework-AVFoundation

# Or use conda/miniforge
conda install -c conda-forge pyobjc-framework-avfoundation
```

### Still using fallback mode?

Check the error message:
```python
python3 -c "
try:
    import AVFoundation
    import CoreMedia
    from Quartz import CoreVideo
    from Cocoa import NSObject
    import libdispatch
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Missing: {e}')
"
```

## 📊 Performance Comparison

| Feature | Native macOS | OpenCV | Screenshot Loop |
|---------|--------------|--------|-----------------|
| FPS | 30 | 10-30 | 1-5 |
| CPU Usage | Low | Medium | High |
| Memory | Efficient | Moderate | Low |
| Indicator | 🟣 Purple | None | None |
| Hardware Accel | ✅ Yes | ❌ No | ❌ No |
| Latency | <50ms | 100-200ms | 500ms+ |

## 🎥 Using Video Streaming

```python
from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig

# Configure for video streaming
config = VisionConfig(
    enable_video_streaming=True,
    prefer_video_over_screenshots=True
)

analyzer = ClaudeVisionAnalyzer(api_key, config)

# Start streaming (purple indicator appears)
await analyzer.start_video_streaming()

# Use video frames for analysis
result = await analyzer.analyze_screenshot(
    await analyzer.capture_screen(),
    "What's on screen?"
)

# Stop streaming (purple indicator disappears)
await analyzer.stop_video_streaming()
```

## 🔗 Related Documentation

- [Video Streaming Guide](VIDEO_STREAMING_GUIDE.md) - Comprehensive video streaming documentation
- [Vision Integration Guide](VISION_INTEGRATION_GUIDE.md) - Full vision system documentation
- [Test Examples](test_video_simple.py) - Example code for testing

## ✨ Benefits of Native Capture

1. **Privacy Awareness**: Purple indicator shows when screen is being recorded
2. **Hardware Acceleration**: Uses Metal/GPU for efficient processing
3. **Low Latency**: Direct access to screen buffer
4. **System Integration**: Works with macOS security features
5. **Power Efficient**: Better battery life on MacBooks

With native video capture enabled, JARVIS can analyze your screen in real-time at 30 FPS while showing the standard macOS screen recording indicator!