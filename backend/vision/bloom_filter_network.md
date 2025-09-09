# 🚀 JARVIS Vision - Bloom Filter Network (4.4)

## Purpose: Prevent Redundant Processing Through Efficient Duplicate Detection

The Bloom Filter Network is part of the Efficient Processing System (EPS) in JARVIS Vision's Performance Optimization Layer. It uses hierarchical bloom filters to efficiently detect and prevent duplicate processing of screenshots, regions, and UI elements.

## 🎯 Key Capabilities

### 1. **Hierarchical Duplicate Detection**
- **Global Filter (4MB)**: System-wide deduplication across all vision processing
- **Regional Filters (1MB × 4)**: Quadrant-specific deduplication for window regions
- **Element Filter (2MB)**: UI element-specific deduplication

### 2. **Memory-Efficient Architecture**
- Total memory footprint: 10MB (4MB + 4MB + 2MB)
- Probabilistic data structure with configurable false positive rates
- Automatic saturation management and intelligent reset strategies

### 3. **High-Performance Operations**
- SIMD-accelerated hashing in Rust
- Lock-free concurrent access
- Hierarchical short-circuiting for fast lookups

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Global Filter (4MB)             │
│    10 hash functions, weekly reset      │
│    Expected: 100K elements              │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼───────┐         ┌───────▼──────┐
│ Regional  │         │  Regional    │
│ Filters   │         │  Filters     │
│ (1MB × 4) │         │  (1MB × 4)   │
│ Q1 │ Q2   │         │  Q3 │ Q4     │
└───┬───────┘         └───────┬──────┘
    │                         │
    └────────────┬────────────┘
                 │
         ┌───────▼──────┐
         │   Element    │
         │  Filter      │
         │   (2MB)      │
         │ 5 hash func  │
         └──────────────┘
```

## 🔧 Configuration

### Environment Variables
```bash
# Enable/disable bloom filter network
BLOOM_FILTER_ENABLED=true

# Memory allocation (MB)
BLOOM_GLOBAL_SIZE_MB=4.0      # Global filter size
BLOOM_REGIONAL_SIZE_MB=1.0    # Per-quadrant size
BLOOM_ELEMENT_SIZE_MB=2.0     # Element filter size

# Features
BLOOM_HIERARCHICAL=true       # Enable hierarchical checking
BLOOM_USE_RUST=true          # Use Rust for fast hashing
BLOOM_USE_SWIFT=true         # Use Swift for UI tracking
```

### Python Configuration
```python
from backend.vision.bloom_filter_network import get_bloom_filter_network

# Get singleton instance with spec-compliant configuration
network = get_bloom_filter_network()  # 4MB/1MB×4/2MB

# Or create custom configuration
from backend.vision.bloom_filter_network import BloomFilterNetwork
network = BloomFilterNetwork(
    global_size_mb=4.0,      # 4MB for global
    regional_size_mb=1.0,    # 1MB × 4 for regional
    element_size_mb=2.0      # 2MB for element
)
```

## 💻 Usage Examples

### Basic Duplicate Detection
```python
from backend.vision.bloom_filter_network import (
    get_bloom_filter_network, 
    VisionBloomFilterIntegration
)

# Initialize
network = get_bloom_filter_network()
integration = VisionBloomFilterIntegration(network)

# Check if image is duplicate
image_hash = "sha256_hash_of_screenshot"
is_duplicate = integration.is_image_duplicate(
    image_hash,
    window_context={'app': 'Safari', 'timestamp': time.time()}
)

if not is_duplicate:
    # Process the new screenshot
    process_screenshot()
```

### Regional Duplicate Detection
```python
# Check UI elements by quadrant
element_data = {
    'type': 'button',
    'text': 'Submit',
    'bounds': {'x': 100, 'y': 100, 'width': 80, 'height': 30}
}

context = {
    'quadrant': 0,  # Top-left quadrant
    'screen_width': 1920,
    'screen_height': 1080,
    'x': 100,
    'y': 100
}

is_duplicate = integration.is_window_region_duplicate(
    element_data, context
)
```

### Direct Network Access
```python
from backend.vision.bloom_filter_network import BloomFilterLevel

# Check and add at specific level
data = b"unique_element_signature"
is_duplicate, found_level = network.check_and_add(
    data,
    level=BloomFilterLevel.ELEMENT
)

# Element promotion happens automatically:
# Element → Regional → Global (based on access patterns)
```

### Integration with Vision Analyzer
```python
# The bloom filter is automatically integrated
analyzer = ClaudeVisionAnalyzer()

# Analyze screenshot - bloom filter prevents duplicates
result, metrics = await analyzer.analyze_screenshot(
    screenshot,
    "What's happening on screen?"
)

# Check bloom filter stats in result
if 'bloom_filter_stats' in result:
    stats = result['bloom_filter_stats']
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print(f"Memory: {stats['memory_usage_mb']}MB")
```

## 📊 Performance Characteristics

### Hash Functions by Level
- **Global Filter**: 10 hash functions (maximum accuracy)
- **Regional Filters**: 7 hash functions per quadrant
- **Element Filter**: 5 hash functions (optimized for speed)

### False Positive Rates
- Target: < 1% false positive rate
- Actual rates depend on saturation level
- Automatic reset when saturation > 80%

### Reset Schedules
- **Global**: Weekly reset (7 days)
- **Regional**: Daily reset (24 hours)
- **Element**: Hourly reset (1 hour)

### Performance Benchmarks
- **Insertion**: ~50,000 elements/second
- **Query**: ~100,000 queries/second
- **Memory overhead**: < 10MB total

## 🔄 Hierarchical Operation

1. **Check Global First**
   - If found → return DUPLICATE (short-circuit)
   - If not found → check Regional

2. **Check Regional (if needed)**
   - Determine quadrant from context
   - If found → promote to Global, return DUPLICATE
   - If not found → check Element

3. **Check Element (if needed)**
   - If found → promote to Regional & Global
   - If not found → add to appropriate level

## 🧮 Quadrant Determination

The system automatically determines quadrants:
```python
def _get_quadrant_index(context):
    if 'quadrant' in context:
        return context['quadrant'] % 4
    
    if 'x' in context and 'y' in context:
        x, y = context['x'], context['y']
        width = context.get('screen_width', 1920)
        height = context.get('screen_height', 1080)
        
        quadrant = 0
        if x > width / 2:
            quadrant += 1
        if y > height / 2:
            quadrant += 2
        return quadrant
    
    return None  # Check all quadrants
```

## 🚦 Memory Management

### Saturation Monitoring
```python
# Get network statistics
stats = network.get_network_stats()

# Check saturation levels
print(f"Global saturation: {stats['global_filter']['saturation_level']:.2%}")
for idx, rf in enumerate(stats['regional_filters']):
    print(f"Regional Q{idx}: {rf['saturation_level']:.2%}")
print(f"Element saturation: {stats['element_filter']['saturation_level']:.2%}")
```

### Automatic Optimization
```python
# Optimize network (resets filters > 85% saturated)
network.optimize_network()

# Manual reset
network.reset_network()  # Reset all
network.reset_network(BloomFilterLevel.REGIONAL)  # Reset specific level
```

## 🌍 Multi-Language Support

### Rust Integration (High Performance)
```rust
use jarvis_rust_core::vision::bloom_filter_network::{
    BloomFilterNetwork, BloomFilterLevel, get_global_bloom_network
};

// Get global network
let network = get_global_bloom_network();

// Check and add
let data = b"element_data";
let (is_duplicate, level) = network.check_and_add(
    data, 
    BloomFilterLevel::Element
);
```

### Swift Integration (macOS Native)
```swift
import SwiftBloomFilterNetwork

// Create network
let network = SwiftBloomFilterNetwork(
    globalSizeMB: 4.0,
    regionalSizeMB: 1.0,
    elementSizeMB: 2.0
)

// UI element tracking
let tracker = SwiftUIElementTracker(bloomNetwork: network)
let isDuplicate = tracker.isUIElementDuplicate(
    axElement,
    windowContext: ["app": "Finder"]
)
```

## 🔍 Debugging and Monitoring

### Enable Debug Logging
```python
import logging
logging.getLogger('bloom_filter_network').setLevel(logging.DEBUG)
```

### Monitor Performance
```python
# Get detailed metrics
metrics = network.get_network_stats()
print(json.dumps(metrics, indent=2))

# Key metrics:
# - total_checks: Total duplicate checks performed
# - hit_rate: Percentage of elements found as duplicates
# - hierarchical_efficiency: Percentage resolved at higher levels
# - saturation levels by filter
```

## 🎯 Best Practices

1. **Context is Key**: Always provide context (app, window, position) for better duplicate detection
2. **Level Selection**: Use appropriate level - Global for screenshots, Regional for windows, Element for UI
3. **Monitor Saturation**: Watch saturation levels and optimize when needed
4. **Batch Operations**: Group related checks for better performance
5. **Trust the Hierarchy**: Let promotion happen naturally based on access patterns

## 🚨 Common Issues

### High False Positive Rate
- Check saturation levels
- Run `optimize_network()` to reset saturated filters
- Consider increasing filter sizes

### Memory Usage Higher Than Expected
- Verify configuration matches spec (4MB/1MB×4/2MB)
- Check for memory leaks in long-running processes
- Monitor system memory with `get_network_stats()`

### Performance Degradation
- Enable Rust hashing: `BLOOM_USE_RUST=true`
- Check CPU usage during hashing operations
- Consider reducing hash functions if needed

## 📈 Future Enhancements

1. **Adaptive Hash Functions**: Dynamically adjust based on saturation
2. **Learned Reset Schedules**: ML-based reset timing
3. **Distributed Bloom Filters**: Share across multiple JARVIS instances
4. **Compression Integration**: Combine with other EPS components
5. **GPU Acceleration**: CUDA/Metal for massive parallel checks

---

The Bloom Filter Network ensures JARVIS Vision operates efficiently by preventing redundant processing at multiple levels, saving computational resources while maintaining high accuracy.