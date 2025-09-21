# Quadtree-Based Spatial Intelligence Guide

## Overview

The Quadtree-Based Spatial Intelligence system minimizes data processing and API calls through intelligent spatial division. It adaptively subdivides screen regions based on visual complexity and importance, allowing JARVIS to focus computational resources on the most relevant areas.

## Architecture

### Memory Allocation (50MB Total)

1. **Quadtree Structures (20MB)**
   - Adaptive subdivision trees
   - Node metadata and relationships
   - Spatial indices

2. **Importance Maps (15MB)**
   - Region weighting data
   - Heatmaps for importance detection
   - Temporal importance tracking

3. **Query Cache (15MB)**
   - Smart query results
   - Region analysis cache
   - Performance optimization data

### Core Components

#### 1. QuadNode Structure
```python
@dataclass
class QuadNode:
    x: int                              # Top-left X coordinate
    y: int                              # Top-left Y coordinate
    width: int                          # Region width
    height: int                         # Region height
    level: int                          # Tree depth level
    importance: float                   # 0.0-1.0 importance score
    complexity: float                   # Visual complexity measure
    last_update: Optional[datetime]     # Last modification time
    hash_value: Optional[str]           # Perceptual hash
    children: Optional[List[QuadNode]]  # Four child nodes (NW, NE, SW, SE)
    cached_result: Optional[Dict]       # Cached analysis result
```

#### 2. Adaptive Subdivision Logic

The system decides whether to subdivide a node based on:
- **Visual Complexity**: Edge density, color variance, texture patterns
- **Importance Score**: UI elements, text regions, active areas
- **Node Size**: Minimum size constraints (default 50x50 pixels)
- **Tree Depth**: Maximum depth limit (default 6 levels)

#### 3. Importance Detection

Importance is calculated using multiple factors:
- **Edge Density**: High edge count indicates UI elements
- **Color Variance**: Diverse colors suggest rich content
- **Position Bias**: Center regions often more important
- **Motion Detection**: Changes over time increase importance
- **UI Patterns**: Buttons, dialogs, error messages

### Multi-Language Implementation

#### Python (Core Logic)
- Quadtree construction and traversal
- ML-based importance detection
- Integration with Claude Vision API
- Caching and optimization logic

#### Rust (High Performance)
- SIMD-accelerated importance calculation
- Parallel region processing
- Efficient memory management
- Fast perceptual hashing

#### Swift (Native macOS)
- Window detection and classification
- UI element recognition
- Accessibility API integration
- Real-time region tracking

## Usage Examples

### Basic Usage

```python
from backend.vision.intelligence.quadtree_spatial_intelligence import (
    get_quadtree_spatial_intelligence, RegionImportance
)

# Initialize
quadtree = get_quadtree_spatial_intelligence()

# Build quadtree from image
image = np.array(pil_image)  # Your screenshot
await quadtree.build_quadtree(image, "screenshot_001")

# Query important regions
result = await quadtree.query_regions(
    "screenshot_001",
    importance_threshold=0.7,  # Only high-importance regions
    max_regions=10             # Limit to 10 regions
)

# Process regions
for node in result.nodes:
    print(f"Region at ({node.x}, {node.y}): {node.width}x{node.height}")
    print(f"Importance: {node.importance:.2f}, Complexity: {node.complexity:.2f}")
```

### Integration with Claude Vision

```python
from backend.vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer

# Enable Quadtree optimization
os.environ['QUADTREE_SPATIAL_ENABLED'] = 'true'
os.environ['QUADTREE_OPTIMIZE_API'] = 'true'

analyzer = ClaudeVisionAnalyzer(api_key)

# Analyze with automatic spatial optimization
result, metrics = await analyzer.analyze_screenshot(
    screenshot,
    "Analyze this screen"
)

# Check spatial analysis results
if 'spatial_analysis' in result:
    print(f"Detected {result['spatial_analysis']['regions_detected']} important regions")
    print(f"Coverage: {result['spatial_analysis']['coverage_ratio']:.1%}")
```

### Focused Analysis

```python
# Define regions of interest
focus_regions = [
    {"x": 100, "y": 200, "width": 300, "height": 400},  # Error dialog
    {"x": 500, "y": 600, "width": 200, "height": 100}   # Submit button
]

# Analyze with spatial focus
result = await analyzer.analyze_with_spatial_focus(
    screenshot,
    "What's happening in these areas?",
    focus_regions=focus_regions
)
```

### Region Optimization

```python
# Find optimal regions for processing
optimization = await analyzer.optimize_regions_with_quadtree(
    screenshot,
    importance_threshold=0.6
)

print(f"Found {optimization['regions_found']} regions")
for region in optimization['regions'][:5]:
    print(f"- ({region['x']}, {region['y']}) {region['width']}x{region['height']}")
    print(f"  Importance: {region['importance']:.2f}")

# Get recommendations
for rec in optimization['recommendations']:
    print(f"{rec['type']}: {rec['reason']}")
```

## Configuration

### Environment Variables

```bash
# Enable/disable Quadtree
export QUADTREE_SPATIAL_ENABLED=true

# Quadtree parameters
export QUADTREE_MAX_DEPTH=6              # Maximum tree depth
export QUADTREE_MIN_NODE_SIZE=50         # Minimum node size in pixels
export QUADTREE_IMPORTANCE_THRESHOLD=0.7 # Default importance threshold

# Caching
export QUADTREE_ENABLE_CACHING=true
export QUADTREE_CACHE_DURATION=5         # Cache duration in minutes

# Performance
export QUADTREE_USE_RUST=true           # Use Rust acceleration
export QUADTREE_USE_SWIFT=true          # Use Swift for macOS
export QUADTREE_OPTIMIZE_API=true       # Optimize API calls
export QUADTREE_MAX_REGIONS=10          # Max regions per analysis
```

## Performance Optimization

### 1. Adaptive Subdivision

The quadtree adapts based on content:
- **High complexity areas**: More subdivision for detail
- **Low complexity areas**: Less subdivision to save memory
- **Static regions**: Cached and skipped in future analyses

### 2. Smart Caching

- **Perceptual hashing**: Detect unchanged regions
- **Query caching**: Reuse results for identical queries
- **Temporal caching**: Track changes over time

### 3. API Call Reduction

When enabled, Quadtree:
1. Identifies important regions
2. Adds spatial hints to prompts
3. Focuses Claude's attention on key areas
4. Reduces overall processing time

### Example Performance Gains

```
Standard Analysis:
- Time: 2.5s
- Full image processed
- No spatial optimization

Quadtree-Optimized:
- Time: 1.8s (28% faster)
- 10 key regions identified
- 65% coverage with focused analysis
- Quadtree overhead: 0.15s
```

## Best Practices

### 1. Importance Thresholds

- **0.3-0.4**: Include most content (high coverage)
- **0.5-0.6**: Balanced approach (recommended)
- **0.7-0.8**: Only critical regions (errors, dialogs)
- **0.9+**: Emergency/critical content only

### 2. Memory Management

```python
# Clean up old quadtrees
await quadtree.cleanup_old_data(max_age_hours=1)

# Get memory usage
stats = quadtree.get_statistics()
print(f"Memory usage: {stats['memory_usage']['total']/1024/1024:.1f} MB")
```

### 3. Access Pattern Optimization

```python
# Let Quadtree learn from usage patterns
await quadtree.optimize_for_patterns("screenshot_001")

# This adjusts importance based on:
# - Frequently accessed regions
# - User interaction patterns
# - Historical analysis results
```

## Advanced Features

### 1. Multi-Scale Analysis

```python
# Query at different scales
scales = [0.3, 0.5, 0.7, 0.9]
multi_scale_results = []

for scale in scales:
    result = await quadtree.query_regions(
        image_id,
        importance_threshold=scale,
        max_regions=20
    )
    multi_scale_results.append({
        'scale': scale,
        'regions': len(result.nodes),
        'coverage': result.coverage_ratio
    })
```

### 2. Temporal Tracking

```python
# Update existing quadtree with new frame
await quadtree.update_regions(
    image_id,
    new_image,
    changed_bounds=detected_changes  # Only update changed areas
)

# Track region changes over time
for node in quadtree.quadtrees[image_id].children:
    if node.change_frequency > 0.5:
        print(f"Dynamic region detected at ({node.x}, {node.y})")
```

### 3. Batch Processing

```python
# Process multiple regions efficiently
from backend.vision.jarvis-rust-core import RegionBatchProcessor

processor = RegionBatchProcessor(max_batch_size=5)
processor.add_regions([(node.bounds, node.importance) for node in nodes])

while processor.has_more():
    batch = processor.next_batch()
    # Process batch with API calls
```

## Integration with Other Components

### With Anomaly Detection
```python
# Focus on anomalous regions
if anomaly_detected:
    anomaly_bounds = anomaly.get_bounds()
    await quadtree.build_quadtree(
        image, 
        image_id, 
        focus_regions=[anomaly_bounds]
    )
```

### With Solution Memory Bank
```python
# Optimize regions around known problem areas
if problem_detected:
    problem_regions = solution_memory.get_problem_regions()
    optimization = await analyzer.analyze_with_spatial_focus(
        screenshot,
        "Check for known issues",
        focus_regions=problem_regions
    )
```

## Troubleshooting

### High Memory Usage
- Reduce `QUADTREE_MAX_DEPTH`
- Increase `QUADTREE_MIN_NODE_SIZE`
- Enable more aggressive caching
- Clean up old trees regularly

### Poor Region Detection
- Lower importance threshold
- Check importance calculation weights
- Verify Swift/Rust components are enabled
- Analyze importance maps visually

### Cache Misses
- Increase cache duration
- Check perceptual hash sensitivity
- Monitor query patterns
- Verify cache size limits

## Future Enhancements

1. **ML-Based Importance**: Train neural networks for better importance detection
2. **Predictive Subdivision**: Anticipate important regions based on context
3. **Cross-Frame Optimization**: Share quadtrees across video frames
4. **Distributed Processing**: Split large images across multiple workers
5. **GPU Acceleration**: Use Metal/CUDA for importance calculations