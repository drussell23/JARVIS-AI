# Screen Sharing Integration Guide for JARVIS Vision System

## Overview

The screen sharing module provides memory-safe, real-time screen sharing capabilities integrated with the JARVIS Vision System. It's designed specifically for macOS systems with 16GB RAM and includes aggressive memory management to prevent crashes.

## Key Features

### 1. Memory-Safe Design
- **Dynamic Memory Limits**: Configurable per-component memory limits
- **Adaptive Quality Control**: Automatically reduces quality when memory is low
- **Emergency Cleanup**: Aggressive cleanup when memory is critical
- **Memory Monitoring**: Continuous tracking of system and process memory

### 2. Integration with Vision System
- **Shared Resources**: Screen captures are shared between vision analysis and screen sharing
- **Priority Management**: Configurable priority between vision and sharing
- **Coordinated Memory Management**: Components communicate memory pressure

### 3. No Hardcoding
- **Environment Variables**: All settings configurable via environment
- **Dynamic Configuration**: Runtime adjustable parameters
- **Adaptive Behavior**: Automatically adjusts to system conditions

## Configuration

### Environment Variables

```bash
# Screen Sharing Configuration
export SCREEN_SHARE_WEBRTC_ENABLED=true
export SCREEN_SHARE_SIGNAL_SERVER=ws://localhost:8765
export SCREEN_SHARE_STUN_SERVERS='["stun:stun.l.google.com:19302"]'

# Quality Settings
export SCREEN_SHARE_TARGET_FPS=15
export SCREEN_SHARE_MIN_FPS=5
export SCREEN_SHARE_MAX_FPS=30
export SCREEN_SHARE_INITIAL_QUALITY=75
export SCREEN_SHARE_MIN_QUALITY=30
export SCREEN_SHARE_MAX_QUALITY=95

# Resolution Settings
export SCREEN_SHARE_INITIAL_RES=1280x720
export SCREEN_SHARE_MIN_RES=640x360
export SCREEN_SHARE_MAX_RES=1920x1080

# Memory Management
export SCREEN_SHARE_MAX_BUFFER=5
export SCREEN_SHARE_MEMORY_LIMIT_MB=500
export SCREEN_SHARE_MEMORY_WARNING=0.7
export SCREEN_SHARE_MEMORY_CRITICAL=0.9

# Adaptive Control
export SCREEN_SHARE_ADAPTIVE_QUALITY=true
export SCREEN_SHARE_QUALITY_CHECK_INTERVAL=2.0
export SCREEN_SHARE_CPU_THRESHOLD=60

# Integration
export SCREEN_SHARE_WITH_VISION=true
export SCREEN_SHARE_VISION_PRIORITY=balanced  # vision_first, sharing_first, balanced

# Vision System Settings
export VISION_SCREEN_SHARING=true
export VISION_CONTINUOUS_ENABLED=true
export VISION_MONITOR_INTERVAL=3.0
export VISION_PROCESS_LIMIT_MB=2048
export VISION_MEMORY_SAFETY=true
```

## Usage

### Basic Screen Sharing

```python
from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer

# Initialize with screen sharing enabled
analyzer = ClaudeVisionAnalyzer(api_key, config)

# Start screen sharing
result = await analyzer.start_screen_sharing()
if result['success']:
    print(f"Sharing URL: {result['sharing_url']}")
    
# Get current status
status = await analyzer.get_screen_sharing_status()
print(f"FPS: {status['metrics']['current_fps']}")
print(f"Quality: {status['metrics']['current_quality']}")

# Stop sharing
await analyzer.stop_screen_sharing()
```

### With Memory Monitoring

```python
# Check memory health before starting
health = await analyzer.check_memory_health()
if health['healthy']:
    await analyzer.start_screen_sharing()
else:
    print(f"Memory warnings: {health['warnings']}")
```

### Event Handling

```python
# Register callbacks for events
screen_sharing = await analyzer.get_screen_sharing()

def on_quality_change(data):
    print(f"Quality changed: {data['quality']} (reason: {data['reason']})")

def on_memory_warning(data):
    print(f"Memory warning: {data['level']} - {data['available_mb']}MB")

screen_sharing.register_callback('quality_changed', on_quality_change)
screen_sharing.register_callback('memory_warning', on_memory_warning)
```

## Memory Management Strategy

### 1. Component Memory Limits
- **Vision Analyzer**: 2GB process limit
- **Screen Sharing**: 500MB dedicated limit
- **Continuous Analyzer**: 200MB limit
- **Cache**: 100MB limit

### 2. Adaptive Quality Levels

| Memory State | Quality | FPS | Resolution |
|-------------|---------|-----|------------|
| Normal (>50%) | 75-95 | 15-30 | 1280x720-1920x1080 |
| Warning (30-50%) | 50-75 | 10-15 | 960x540-1280x720 |
| Critical (<30%) | 30-50 | 5-10 | 640x360-960x540 |

### 3. Priority Modes

- **vision_first**: Vision analysis gets priority, sharing pauses during analysis
- **sharing_first**: Screen sharing continues, vision analysis may be delayed
- **balanced**: Both share resources equally with dynamic adjustment

## Performance Optimization

### 1. For 16GB Systems

```bash
# Recommended settings for 16GB RAM
export VISION_PROCESS_LIMIT_MB=2048
export SCREEN_SHARE_MEMORY_LIMIT_MB=500
export VISION_MAX_CONCURRENT=10
export SCREEN_SHARE_TARGET_FPS=15
export SCREEN_SHARE_INITIAL_RES=1280x720
```

### 2. For Heavy Workloads

```bash
# Conservative settings for stability
export VISION_MEMORY_THRESHOLD=50
export SCREEN_SHARE_ADAPTIVE_QUALITY=true
export SCREEN_SHARE_MIN_FPS=5
export SCREEN_SHARE_MIN_QUALITY=30
export VISION_REJECT_ON_MEMORY=true
```

### 3. For Best Quality

```bash
# Maximum quality (requires good memory conditions)
export SCREEN_SHARE_INITIAL_QUALITY=95
export SCREEN_SHARE_TARGET_FPS=30
export SCREEN_SHARE_MAX_RES=1920x1080
export VISION_COMPRESSION=false
```

## Troubleshooting

### Common Issues

1. **"Insufficient memory for screen sharing"**
   - Reduce other component limits
   - Lower initial quality/resolution
   - Enable more aggressive adaptive quality

2. **Frequent quality drops**
   - Check CPU usage
   - Reduce target FPS
   - Enable compression

3. **Screen sharing stops unexpectedly**
   - Check memory warnings in logs
   - Monitor system memory
   - Review crash logs

### Debug Mode

```bash
# Enable detailed logging
export VISION_DEBUG=true
export SCREEN_SHARE_DEBUG=true
export PYTHONUNBUFFERED=1
```

### Memory Monitoring

```python
# Get detailed memory stats
stats = analyzer.get_all_memory_stats()
print(f"Process: {stats['system']['process_mb']}MB")
print(f"Available: {stats['system']['available_mb']}MB")

# Component-specific stats
for component, metrics in stats['components'].items():
    print(f"{component}: {metrics}")
```

## Architecture

### Component Interaction

```
┌─────────────────────────────────────────────────┐
│          Claude Vision Analyzer Main            │
│  - Memory Monitor                               │
│  - Configuration Management                     │
│  - Component Orchestration                      │
└─────────────────┬───────────────────────────────┘
                  │
     ┌────────────┴────────────┬─────────────────┐
     │                         │                 │
┌────▼──────────────┐ ┌───────▼────────┐ ┌─────▼──────────────┐
│ Continuous Screen │ │ Screen Sharing │ │  Claude Vision API │
│    Analyzer       │ │    Manager     │ │    Integration     │
│ - Capture Buffer  │ │ - WebRTC       │ │ - Image Analysis   │
│ - Event System    │ │ - Streaming    │ │ - Compression      │
│ - Memory Safety   │ │ - Quality Ctrl │ │ - Caching          │
└───────────────────┘ └────────────────┘ └────────────────────┘
```

### Memory Flow

1. **Capture**: Screen captured by continuous analyzer
2. **Buffer**: Stored in circular buffer (limited size)
3. **Share**: Frame buffer shared between components
4. **Stream**: Encoded and sent to peers
5. **Cleanup**: Old frames automatically removed

## Future Enhancements

1. **WebRTC Full Implementation**
   - STUN/TURN server integration
   - Peer-to-peer connectivity
   - NAT traversal

2. **Recording Capability**
   - Save streams to disk
   - Configurable compression
   - Time-based segments

3. **Multi-Stream Support**
   - Multiple quality streams
   - Selective region sharing
   - Application-specific capture

4. **Enhanced Analytics**
   - Bandwidth usage tracking
   - Quality metrics history
   - Performance analytics

## Best Practices

1. **Always check memory health before starting**
2. **Use event callbacks to monitor state**
3. **Configure appropriate limits for your system**
4. **Enable adaptive quality for stability**
5. **Monitor logs for memory warnings**
6. **Test with reduced limits first**
7. **Use balanced priority mode for most cases**

## Example: Production Setup

```python
import os
from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig

# Production configuration
config = VisionConfig(
    # Memory safety
    enable_memory_safety=True,
    process_memory_limit_mb=2048,
    memory_warning_threshold_mb=1536,
    reject_on_memory_pressure=True,
    
    # Screen sharing
    enable_screen_sharing=True,
    enable_continuous_monitoring=True,
    
    # Performance
    max_concurrent_requests=10,
    cache_enabled=True,
    compression_enabled=True,
    
    # Features
    enable_metrics=True
)

# Initialize
analyzer = ClaudeVisionAnalyzer(api_key, config)

# Setup callbacks
async def handle_memory_warning(data):
    logger.warning(f"Memory warning: {data}")
    # Could trigger cleanup or notifications

# Start services
await analyzer.start_continuous_monitoring()
result = await analyzer.start_screen_sharing()

# Run with monitoring
try:
    # Your application logic here
    pass
finally:
    # Cleanup
    await analyzer.cleanup_all_components()
```