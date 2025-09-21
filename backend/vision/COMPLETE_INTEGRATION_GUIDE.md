# Complete Vision System Integration Guide

## Overview

The `claude_vision_analyzer_main.py` now integrates **6 memory-optimized components**, creating the most comprehensive vision system for JARVIS. All components are optimized for 16GB RAM macOS systems with **NO hardcoded values**.

## All 6 Integrated Components

### 1. **Continuous Screen Analyzer** (`continuous_screen_analyzer.py`)
- **Class**: `MemoryAwareScreenAnalyzer`
- **Purpose**: Real-time screen monitoring with memory management
- **Key Features**:
  - Circular buffer for captures
  - Dynamic interval adjustment (1-10s based on memory)
  - Emergency cleanup when memory < 1GB
  - Weak references for callbacks

### 2. **Window Analysis** (`window_analysis.py`)
- **Class**: `MemoryAwareWindowAnalyzer`
- **Purpose**: Analyze window content and workspace layout
- **Key Features**:
  - Configurable app categories
  - LRU cache with memory limits
  - Lazy imports for efficiency
  - Workspace layout detection

### 3. **Window Relationship Detector** (`window_relationship_detector.py`)
- **Class**: `ConfigurableWindowRelationshipDetector`
- **Purpose**: Detect relationships between windows
- **Key Features**:
  - Dynamic app list loading
  - Confidence-based relationships
  - Window grouping by project/task
  - Pattern learning and persistence

### 4. **Swift Vision Integration** (`swift_vision_integration.py`)
- **Class**: `MemoryAwareSwiftVisionIntegration`
- **Purpose**: Metal-accelerated vision processing
- **Key Features**:
  - Circuit breaker protection
  - Metal memory monitoring
  - Dynamic quality adjustment
  - Automatic Python fallback

### 5. **Memory-Efficient Analyzer** (`memory_efficient_vision_analyzer.py`)
- **Class**: `MemoryEfficientVisionAnalyzer`
- **Purpose**: Smart compression and caching strategies
- **Key Features**:
  - 5 compression strategies (text, ui, activity, detailed, quick)
  - Persistent cache with TTL
  - Batch region processing
  - Change detection optimization

### 6. **Simplified Vision System** (`vision_system_claude_only.py`)
- **Class**: `SimplifiedVisionSystem`
- **Purpose**: Direct Claude API access with query templates
- **Key Features**:
  - 9+ configurable query templates
  - No local ML models (faster)
  - Custom template support
  - Memory statistics tracking

## Complete Environment Variables List (70+)

### Main Vision Analyzer
```bash
VISION_MAX_IMAGE_DIM=1536         # Maximum image dimension
VISION_JPEG_QUALITY=85            # JPEG compression quality
VISION_CACHE_SIZE_MB=100          # Cache size limit
VISION_MEMORY_THRESHOLD=70        # Memory threshold percent
VISION_MODEL=claude-3-5-sonnet-20241022  # Claude model
VISION_MAX_TOKENS=1500            # Max API tokens
VISION_CACHE_TTL_MIN=30           # Cache TTL in minutes
VISION_MAX_CONCURRENT=2           # Max concurrent requests
VISION_THREAD_POOL=2              # Thread pool size
```

### Continuous Screen Analyzer
```bash
VISION_MONITOR_INTERVAL=3.0       # Update interval seconds
VISION_MAX_CAPTURES=10            # Max captures in memory
VISION_MEMORY_LIMIT_MB=200        # Component memory limit
VISION_CAPTURE_RETENTION=300      # Capture retention seconds
VISION_CACHE_DURATION=5.0         # Analysis cache duration
VISION_LOW_MEMORY_MB=2000         # Low memory threshold
VISION_CRITICAL_MEMORY_MB=1000    # Critical memory threshold
VISION_DYNAMIC_INTERVAL=true      # Enable dynamic intervals
VISION_MIN_INTERVAL=1.0           # Minimum interval
VISION_MAX_INTERVAL=10.0          # Maximum interval
```

### Window Analyzer
```bash
WINDOW_ANALYZER_MAX_MEMORY_MB=100 # Max memory usage
WINDOW_MAX_CACHED=50              # Max cached windows
WINDOW_CACHE_TTL=300              # Cache TTL seconds
WINDOW_MAX_PER_ANALYSIS=20        # Max windows per analysis
WINDOW_OCR_TIMEOUT=5.0            # OCR timeout seconds
WINDOW_SKIP_MINIMIZED=true        # Skip minimized windows
WINDOW_SCREEN_WIDTH=1920          # Screen width
WINDOW_SCREEN_HEIGHT=1080         # Screen height
WINDOW_OVERLAP_THRESHOLD=0.2      # Overlap threshold
WINDOW_LOW_MEMORY_MB=2000         # Low memory threshold
WINDOW_CRITICAL_MEMORY_MB=1000    # Critical memory threshold
```

### Window Relationship Detector
```bash
WINDOW_REL_MAX_MEMORY_MB=50       # Max memory usage
WINDOW_REL_MAX_CACHED=100         # Max cached relationships
WINDOW_REL_MAX_GROUPS=20          # Max cached groups
WINDOW_REL_CACHE_TTL=300          # Cache TTL seconds
WINDOW_REL_MIN_CONFIDENCE=0.5     # Min confidence threshold
WINDOW_REL_GROUP_MIN_CONF=0.6     # Group min confidence
WINDOW_REL_MAX_ANALYZE=50         # Max windows to analyze
WINDOW_REL_TITLE_SIM=0.6          # Title similarity threshold
WINDOW_REL_WORD_MIN_LEN=3         # Min word length
WINDOW_REL_LOW_MEMORY_MB=2000     # Low memory threshold
```

### Swift Vision Integration
```bash
SWIFT_VISION_MAX_MEMORY_MB=300    # Max memory usage
SWIFT_VISION_MAX_CACHED=20        # Max cached results
SWIFT_VISION_CACHE_TTL=300        # Cache TTL seconds
SWIFT_VISION_MAX_DIMENSION=4096   # Max image dimension
SWIFT_VISION_BATCH_SIZE=5         # Batch processing size
SWIFT_VISION_TIMEOUT=10.0         # Processing timeout
SWIFT_VISION_JPEG_QUALITY=80      # JPEG quality
SWIFT_VISION_LOW_QUALITY=60       # Low memory quality
SWIFT_VISION_HIGH_QUALITY=95      # High memory quality
SWIFT_VISION_METAL_LIMIT_MB=1000  # Metal memory limit
SWIFT_VISION_CB_THRESHOLD=3       # Circuit breaker threshold
```

### Memory-Efficient Analyzer
```bash
VISION_CACHE_DIR=./vision_cache   # Cache directory
VISION_CACHE_SIZE_MB=500          # Cache size limit
VISION_MAX_MEMORY_MB=2048         # Max memory usage
VISION_MEMORY_PRESSURE_THRESHOLD=0.8  # Memory pressure threshold
VISION_CACHE_TTL_HOURS=24         # Cache TTL hours
VISION_MAX_TOKENS=1024            # Max API tokens
VISION_BATCH_MAX_REGIONS=10       # Max batch regions
VISION_CHANGE_THRESHOLD=0.05      # Change detection threshold
VISION_MAX_WORKERS=3              # Thread pool workers

# Compression settings per type
VISION_TEXT_FORMAT=PNG            # Text compression format
VISION_TEXT_QUALITY=95            # Text quality
VISION_TEXT_MAX_DIM=2048          # Text max dimension
VISION_UI_FORMAT=JPEG             # UI compression format
VISION_UI_QUALITY=85              # UI quality
VISION_UI_MAX_DIM=1920            # UI max dimension
# ... (similar for activity, detailed, quick)
```

### Simplified Vision System
```bash
VISION_MAX_RESPONSE_CACHE=10      # Response cache size
VISION_CACHE_TTL=300              # Cache TTL seconds
VISION_ENABLE_MEMORY_STATS=true   # Enable memory stats
VISION_CONFIDENCE_THRESHOLD=0.9   # Confidence threshold
VISION_DEFAULT_TIMEOUT=30.0       # Default timeout

# Query Templates
VISION_QUERY_GENERAL              # General analysis template
VISION_QUERY_ELEMENT              # Element finding template
VISION_QUERY_TEXT_AREA            # Text area reading template
VISION_QUERY_TEXT_ALL             # All text reading template
VISION_QUERY_NOTIFICATIONS        # Notifications check template
VISION_QUERY_WEATHER              # Weather check template
VISION_QUERY_ERRORS               # Error check template
VISION_QUERY_WORKSPACE            # Workspace analysis template
VISION_QUERY_ACTIVITY             # Activity analysis template
```

## Usage Examples

### Basic Usage
```python
from claude_vision_analyzer_main import ClaudeVisionAnalyzer

analyzer = ClaudeVisionAnalyzer(api_key)

# Smart analysis (auto-selects method)
result = await analyzer.smart_analyze(screenshot, "Analyze screen")

# Use specific compression strategy
result = await analyzer.analyze_with_compression_strategy(
    screenshot, "Read text", strategy="text"
)

# Batch analyze regions
regions = [{"x": 0, "y": 0, "width": 200, "height": 150}]
results = await analyzer.batch_analyze_regions(screenshot, regions)

# Check for notifications
notifications = await analyzer.check_for_notifications()

# Find UI element
element = await analyzer.find_ui_element("close button")
```

### Advanced Configuration
```python
# Configure for low memory system
os.environ['VISION_MEMORY_LIMIT_MB'] = '150'
os.environ['SWIFT_VISION_MAX_MEMORY_MB'] = '200'
os.environ['WINDOW_REL_MAX_MEMORY_MB'] = '30'
os.environ['VISION_JPEG_QUALITY'] = '70'
os.environ['VISION_MAX_IMAGE_DIM'] = '1024'

analyzer = ClaudeVisionAnalyzer(api_key)
```

### Memory Management
```python
# Get comprehensive memory stats
stats = analyzer.get_all_memory_stats()
print(f"Total memory used: {stats['system']['process_mb']}MB")

# Cleanup when needed
await analyzer.cleanup_all_components()
```

## Memory Optimization Summary

### Total Memory Budget
- **Vision System Total**: ~1GB maximum
- **Per Component Breakdown**:
  - Swift Vision: 300MB
  - Memory-Efficient Analyzer: 200MB
  - Continuous Analyzer: 200MB
  - Window Analyzer: 100MB
  - Relationship Detector: 50MB
  - Simplified Vision: Minimal
  - Main Analyzer Cache: 100MB

### Dynamic Adjustments
1. **Quality Reduction**: JPEG quality drops from 95% → 60% under memory pressure
2. **Interval Increase**: Monitoring interval increases from 1s → 10s when memory low
3. **Cache Eviction**: LRU eviction when memory limits approached
4. **Circuit Breaking**: Swift Vision disabled temporarily on repeated failures
5. **Emergency Cleanup**: All caches cleared when memory < 1GB

### Best Practices
1. **Set appropriate limits** based on your system's available RAM
2. **Monitor memory stats** regularly using `get_all_memory_stats()`
3. **Use compression strategies** appropriate to your use case
4. **Enable only needed components** via environment variables
5. **Configure cache TTLs** based on your usage patterns

## Troubleshooting

### High Memory Usage
```bash
# Reduce memory limits
export VISION_MEMORY_LIMIT_MB=100
export SWIFT_VISION_MAX_MEMORY_MB=150
export WINDOW_ANALYZER_MAX_MEMORY_MB=50

# Reduce quality settings
export VISION_JPEG_QUALITY=70
export VISION_MAX_IMAGE_DIM=1024

# Reduce cache sizes
export VISION_CACHE_SIZE_MB=50
export WINDOW_MAX_CACHED=20
```

### Slow Performance
```bash
# Increase intervals
export VISION_MONITOR_INTERVAL=5.0
export VISION_MIN_INTERVAL=2.0

# Enable Swift acceleration
export VISION_SWIFT_ENABLED=true
export SWIFT_VISION_ENABLED=true

# Use quick compression
export VISION_QUICK_QUALITY=60
export VISION_QUICK_MAX_DIM=800
```

### Component Not Loading
```bash
# Enable specific components
export VISION_CONTINUOUS_ENABLED=true
export VISION_WINDOW_ANALYSIS_ENABLED=true
export VISION_RELATIONSHIP_ENABLED=true
export VISION_SWIFT_ENABLED=true
export VISION_MEMORY_EFFICIENT_ENABLED=true
export VISION_SIMPLIFIED_ENABLED=true
```

## Conclusion

The integrated vision system provides unprecedented flexibility and memory efficiency for JARVIS on 16GB RAM macOS systems. With 70+ environment variables and 6 specialized components, it can be configured for any use case while maintaining optimal performance.

**Key Achievement**: All hardcoded values have been eliminated, making the system fully configurable and adaptable to different environments and requirements.