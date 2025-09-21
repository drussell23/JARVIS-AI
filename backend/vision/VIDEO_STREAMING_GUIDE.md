# Video Streaming Integration Guide for JARVIS Vision System

## Overview

The video streaming integration enables JARVIS to capture and analyze continuous video streams instead of just taking screenshots. This provides real-time understanding of screen content with intelligent memory management for 16GB macOS systems.

## Key Benefits

### 1. Real-Time Understanding
- **30 FPS video capture** instead of periodic screenshots
- **Motion detection** triggers immediate analysis
- **Continuous context** for better AI understanding
- **Lower latency** for voice interactions

### 2. macOS Screen Recording Indicator
- **Purple indicator** appears when video streaming is active
- **Privacy-aware** - users know when JARVIS is viewing
- **System-level integration** using AVFoundation

### 3. Memory-Safe Video Processing
- **Sliding window on video frames** for large screens
- **Adaptive quality** based on memory pressure
- **Frame buffer management** with configurable limits
- **Automatic fallback** to screenshots if needed

## Configuration

### Environment Variables

```bash
# Video Streaming Configuration
export VISION_VIDEO_STREAMING=true
export VISION_PREFER_VIDEO=true

# Video Quality Settings
export VIDEO_STREAM_FPS=30
export VIDEO_STREAM_RESOLUTION=1920x1080
export VIDEO_STREAM_DISPLAY_ID=0

# Memory Management
export VIDEO_STREAM_MEMORY_LIMIT_MB=800    # 800MB for video
export VIDEO_STREAM_BUFFER_SIZE=10          # Max frames in buffer
export VIDEO_STREAM_FRAME_THRESHOLD_MB=50   # Per-frame memory limit

# Sliding Window for Video
export VIDEO_STREAM_SLIDING_WINDOW=true
export VIDEO_STREAM_WINDOW_SIZE=640x480
export VIDEO_STREAM_WINDOW_OVERLAP=0.2
export VIDEO_STREAM_MAX_WINDOWS=4

# Analysis Settings
export VIDEO_STREAM_ANALYZE_INTERVAL=30     # Analyze every 30 frames (1 second at 30fps)
export VIDEO_STREAM_MOTION_DETECTION=true
export VIDEO_STREAM_MOTION_THRESHOLD=0.1

# Adaptive Quality
export VIDEO_STREAM_ADAPTIVE=true
export VIDEO_STREAM_MIN_FPS=10
export VIDEO_STREAM_MIN_RES=960x540
```

## Usage

### Basic Video Streaming

```python
from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer

# Start video streaming
result = await analyzer.start_video_streaming()
# macOS will show purple screen recording indicator

# Get status
status = await analyzer.get_video_streaming_status()
print(f"Capturing at {status['metrics']['frames_processed']} frames")

# Stop streaming
await analyzer.stop_video_streaming()
```

### Real-Time Analysis with Video

```python
# Analyze video stream for specific duration
results = await analyzer.analyze_video_stream(
    query="What is the user doing on screen?",
    duration_seconds=10.0
)

# Results include multiple frame analyses over time
for frame_result in results['results']:
    print(f"Time: {frame_result['timestamp']}s")
    print(f"Analysis: {frame_result['analysis']}")
```

### Switching Between Modes

```python
# Switch to video mode (shows indicator)
await analyzer.switch_to_video_mode()

# Switch back to screenshot mode (no indicator)
await analyzer.switch_to_screenshot_mode()
```

### Motion-Triggered Analysis

```python
# Register callback for motion detection
async def on_motion_detected(data):
    print(f"Motion detected: {data['motion_score']}")
    # Could trigger more detailed analysis

video = await analyzer.get_video_streaming()
video.register_callback('motion_detected', on_motion_detected)
```

## Memory Management

### Video-Specific Memory Strategy

1. **Frame Buffer**: Limited to 10 frames (configurable)
2. **Per-Frame Limit**: 50MB maximum per frame
3. **Total Video Memory**: 800MB dedicated limit
4. **Sliding Window**: Activates for frames >2MP or memory <3GB

### Adaptive Quality Levels

| Memory State | FPS | Resolution | Analysis Interval |
|-------------|-----|------------|-------------------|
| Normal | 30 | 1920x1080 | Every 1s |
| Medium | 20 | 1280x720 | Every 1.5s |
| Low | 10 | 960x540 | Every 2s |

### Integration with Vision System

```
Total Memory Budget (16GB System):
├── Vision Analyzer: 2GB limit
├── Video Streaming: 800MB limit
├── Screen Sharing: 500MB limit
├── Continuous Analyzer: 200MB limit
└── Cache & Buffers: 100MB limit
Total: ~3.6GB maximum
```

## Voice Activation Integration

When integrated with voice activation:

```python
# JARVIS can now respond to real-time queries
User: "What am I looking at?"
JARVIS: *analyzes current video frame* "You're viewing VS Code with..."

User: "What just happened on screen?"
JARVIS: *checks recent frame analyses* "A notification just appeared..."

User: "Tell me when something changes"
JARVIS: *monitors motion detection* "I'll watch for changes..."
```

## Platform Support

### macOS (Primary - Full Features)
- Native AVFoundation capture
- System screen recording indicator
- Hardware acceleration
- Low latency

### Fallback Support
- OpenCV capture (if installed)
- Screenshot loop (universal fallback)
- No indicator in fallback modes

## Performance Optimization

### For Best Performance

```bash
# High-performance settings
export VIDEO_STREAM_FPS=60
export VIDEO_STREAM_RESOLUTION=1920x1080
export VIDEO_STREAM_SLIDING_WINDOW=false
export VIDEO_STREAM_ANALYZE_INTERVAL=60  # Every 2 seconds
```

### For Memory Conservation

```bash
# Memory-safe settings
export VIDEO_STREAM_FPS=15
export VIDEO_STREAM_RESOLUTION=1280x720
export VIDEO_STREAM_SLIDING_WINDOW=true
export VIDEO_STREAM_WINDOW_SIZE=480x360
export VIDEO_STREAM_ADAPTIVE=true
```

## Troubleshooting

### "Failed to create screen input"
- Grant screen recording permission in System Preferences
- Security & Privacy → Screen Recording → Enable your app

### High Memory Usage
- Reduce FPS or resolution
- Enable sliding window
- Increase analyze interval

### No Purple Indicator
- Verify `VISION_VIDEO_STREAMING=true`
- Check if using fallback mode (no AVFoundation)
- Ensure proper permissions granted

## Example: Complete Integration

```python
import asyncio
from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig

async def main():
    # Configure for video streaming
    config = VisionConfig(
        enable_video_streaming=True,
        prefer_video_over_screenshots=True,
        enable_screen_sharing=True,  # Can run simultaneously
        process_memory_limit_mb=2048,
        memory_threshold_percent=60
    )
    
    analyzer = ClaudeVisionAnalyzer(api_key, config)
    
    # Start video streaming
    await analyzer.start_video_streaming()
    print("🎥 Video streaming active - check for purple indicator")
    
    # Real-time analysis loop
    while True:
        # Get user voice command
        command = await get_voice_command()
        
        if "what do you see" in command:
            # Analyze current video frame
            result = await analyzer.analyze_screenshot(
                await analyzer.capture_screen(),
                "Describe what's currently on screen"
            )
            speak(result['description'])
            
        elif "watch for changes" in command:
            # Monitor for motion
            await analyzer.analyze_video_stream(
                "Alert me to any significant changes",
                duration_seconds=30.0
            )
            
        elif "stop watching" in command:
            break
    
    # Cleanup
    await analyzer.stop_video_streaming()
    await analyzer.cleanup_all_components()
```

## Benefits Over Screenshot Mode

1. **Continuous Context**: No gaps between captures
2. **Motion Awareness**: Detects and responds to changes
3. **Better for Demos**: Screen sharing + video analysis
4. **Voice Integration**: More responsive to queries
5. **Efficiency**: Analyze only when needed (motion/interval)

The video streaming mode transforms JARVIS from a periodic observer to a continuous, aware assistant that truly understands what's happening on your screen in real-time.