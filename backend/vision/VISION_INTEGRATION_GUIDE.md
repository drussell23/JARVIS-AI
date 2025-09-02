# JARVIS Vision System Integration Guide

## Overview

The JARVIS vision system is now fully integrated with sliding window support in `claude_vision_analyzer.py`. This guide explains how all vision components work together.

## Core Integration File: `claude_vision_analyzer.py`

This is the **main vision analyzer** that integrates all vision capabilities:

### Key Features:
1. **Smart Analysis** - Automatically chooses between full or sliding window analysis
2. **Memory-Aware** - Adapts to available system memory (optimized for 16GB RAM)
3. **Fully Configurable** - Everything via environment variables (NO hardcoding)
4. **Sliding Window Support** - Integrated for efficient large image processing
5. **Caching System** - LRU cache with memory awareness
6. **Dynamic Entity Extraction** - Learns patterns over time

### Main Methods:

```python
# Smart analysis (auto-selects method)
result = await analyzer.smart_analyze(screenshot, query)

# Force sliding window
result = await analyzer.analyze_with_sliding_window(screenshot, query)

# Quick analysis
result = await analyzer.quick_analysis(screenshot)

# Backward compatible async method
result = await analyzer.analyze_screenshot_async(screenshot, query)
```

## Integration with Rust Components

The vision system integrates with Rust components via `rust_integration.py`:

1. **Zero-Copy Capture** - `jarvis-rust-core/src/vision/capture.rs`
2. **Sliding Window** - `jarvis-rust-core/src/vision/sliding_window.rs`
3. **Image Compression** - `jarvis-rust-core/src/vision/compression.rs`
4. **macOS Optimization** - `jarvis-rust-core/src/vision/macos_optimization.rs`

## Environment Variables

### Core Vision Settings
```bash
# Image Processing
VISION_MAX_IMAGE_DIM=1536        # Max image dimension before resize
VISION_JPEG_QUALITY=85           # JPEG compression quality
VISION_COMPRESSION=true          # Enable compression

# API Settings
VISION_MODEL=claude-3-5-sonnet-20241022
VISION_MAX_TOKENS=1500
VISION_API_TIMEOUT=30

# Cache Settings
VISION_CACHE_ENABLED=true
VISION_CACHE_SIZE_MB=100
VISION_CACHE_TTL_MIN=30

# Performance
VISION_MEMORY_THRESHOLD=70       # Memory usage % threshold
VISION_THREAD_POOL=2
```

### Sliding Window Settings
```bash
# Window Configuration
VISION_WINDOW_WIDTH=400          # Window width in pixels
VISION_WINDOW_HEIGHT=300         # Window height in pixels
VISION_WINDOW_OVERLAP=0.3        # Overlap percentage (0.0-1.0)
VISION_MAX_WINDOWS=4             # Max windows to analyze

# Smart Analysis Thresholds
VISION_SLIDING_THRESHOLD_PX=800000   # Use sliding if image > 800k pixels
VISION_SLIDING_MEMORY_MB=2000        # Use sliding if memory < 2GB

# Optimization
VISION_PRIORITIZE_CENTER=true    # Prioritize center regions
VISION_ADAPTIVE_SIZING=true      # Adapt window size to memory
```

## Usage Examples

### 1. Basic Usage
```python
from claude_vision_analyzer import ClaudeVisionAnalyzer

# Initialize
analyzer = ClaudeVisionAnalyzer(api_key)

# Smart analysis (auto-chooses method)
result = await analyzer.smart_analyze(screenshot, "What's on screen?")
```

### 2. Force Sliding Window
```python
# Force sliding window for detailed analysis
result = await analyzer.analyze_with_sliding_window(
    screenshot, 
    "Find all buttons and UI elements",
    window_config={
        'window_width': 500,
        'window_height': 400,
        'max_windows': 6
    }
)
```

### 3. JARVIS Integration
```python
from jarvis_sliding_window_example import JarvisSlidingWindowVision

# Initialize JARVIS vision
jarvis_vision = JarvisSlidingWindowVision(api_key)

# Process vision command
command = JarvisVisionCommand(
    command_type='find_element',
    query='close button for WhatsApp',
    priority='high'
)
result = await jarvis_vision.process_vision_command(command, screenshot)
```

## Decision Logic

The system automatically chooses sliding window when:

1. **Large Image**: Total pixels > 800,000 (configurable)
2. **Low Memory**: Available RAM < 2GB
3. **Search Query**: Contains words like "find", "locate", "search", "where"
4. **High Detail**: Query requires detailed analysis

## Memory Optimization

For 16GB RAM systems:

1. **Adaptive Window Sizing**: Reduces window size when memory < 2GB
2. **Priority-Based Analysis**: Analyzes center regions first
3. **Caching**: Avoids re-analyzing identical regions
4. **Compression**: JPEG compression reduces API payload by 60-70%

## Integration Flow

```
User Request
    ↓
JarvisVisionCommand
    ↓
smart_analyze() [Decides method]
    ↙        ↘
Full Analysis   Sliding Window
    ↓              ↓
    ↓         Generate Windows
    ↓              ↓
    ↓         Analyze Each Window
    ↓              ↓
    ↓         Combine Results
    ↘          ↙
     Final Result
```

## Performance Tips

1. **Use Smart Analysis**: Let the system decide the best method
2. **Configure Thresholds**: Adjust based on your system's RAM
3. **Enable Caching**: Significantly improves repeated analyses
4. **Prioritize Center**: Most important content is usually centered
5. **Adaptive Sizing**: Automatically adjusts to memory pressure

## Testing

Run integration tests:
```bash
python test_sliding_window_integration.py
```

Run JARVIS examples:
```bash
python jarvis_sliding_window_example.py
```

## Troubleshooting

1. **Out of Memory**: Reduce `VISION_MAX_WINDOWS` or window dimensions
2. **Slow Analysis**: Enable `VISION_COMPRESSION` and reduce `VISION_JPEG_QUALITY`
3. **Missing Rust**: Run `maturin develop` in `jarvis-rust-core/`

## Future Enhancements

1. **GPU Acceleration**: Metal/CUDA support for faster processing
2. **Intelligent Window Selection**: ML-based region importance
3. **Streaming Analysis**: Process windows as they complete
4. **Multi-Monitor Support**: Handle multiple displays efficiently