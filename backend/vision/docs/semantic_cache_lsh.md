# Semantic Cache with LSH Documentation

## Overview

The Semantic Cache with LSH (Locality-Sensitive Hashing) is a multi-level intelligent caching system designed to eliminate redundant API calls through semantic matching. It provides a 250MB cache allocation across 4 specialized cache levels, each optimized for different access patterns.

## Architecture

### Multi-Level Cache Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Semantic Cache with LSH                   │
├─────────────────────────────────────────────────────────────┤
│  L1: Exact Match Cache (20MB, 30s TTL)                      │
│  ├─ O(1) lookup for identical queries                       │
│  └─ LRU eviction policy                                     │
├─────────────────────────────────────────────────────────────┤
│  L2: Semantic Similarity Cache (100MB, 5min TTL)           │
│  ├─ LSH-based similarity search                             │
│  ├─ Cosine similarity threshold: 0.85                       │
│  └─ SIMD-accelerated computations                          │
├─────────────────────────────────────────────────────────────┤
│  L3: Contextual Cache (80MB, 30min TTL)                    │
│  ├─ Goal and workflow-based matching                        │
│  └─ Application context awareness                           │
├─────────────────────────────────────────────────────────────┤
│  L4: Predictive Cache (50MB, dynamic TTL)                   │
│  ├─ Pattern-based pre-computation                           │
│  └─ Access history learning                                 │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Locality-Sensitive Hashing (LSH)**
- Random projection-based hashing for fast similarity search
- Multiple hash tables for improved recall
- Configurable hash size and table count

### 2. **Embedding Generation**
- Support for multiple embedding models
- Fallback hash-based embeddings
- 384-dimensional vectors (compatible with sentence-transformers)

### 3. **SIMD Acceleration**
- AVX2 support for x86_64 architectures
- Optimized cosine similarity computation
- Parallel batch processing

### 4. **Smart Eviction Policies**
- Value-based scoring system
- Factors: age, access count, recency, size
- Configurable eviction thresholds

### 5. **Pattern Recognition**
- Sequential access pattern learning
- Confidence-based predictions
- Automatic cache pre-warming

## Implementation Details

### Python Implementation (`semantic_cache_lsh.py`)

```python
# Core classes
- SemanticCacheWithLSH: Main cache orchestrator
- L1ExactMatchCache: Fast exact matching
- L2SemanticSimilarityCache: LSH-based similarity
- L3ContextualCache: Context-aware caching
- L4PredictiveCache: Pattern-based predictions
- LSHIndex: Locality-sensitive hashing implementation
- PatternPredictor: Access pattern analysis
```

### Rust Implementation (`semantic_cache_lsh.rs`)

```rust
// High-performance components
- SemanticCacheLSH: Core cache with DashMap
- LSHIndex: SIMD-optimized hashing
- SimilarityComputer: AVX2 cosine similarity
- CachePredictor: Pattern mining
```

### Swift Implementation (`semantic_cache_swift.swift`)

```swift
// Native macOS integration
- SemanticCacheSwift: NSObject-compatible cache
- EmbeddingGenerator: Core ML integration
- SimilarityComputer: Accelerate framework
- PredictiveCacheWarmer: Background pre-warming
```

## Usage Examples

### Basic Usage

```python
# Initialize cache
cache = await get_semantic_cache()

# Store with semantic embedding
await cache.put(
    key="click blue button",
    value={"action": "click", "target": "submit_button"},
    embedding=generate_embedding("click blue button"),
    cache_levels=[CacheLevel.L1_EXACT, CacheLevel.L2_SEMANTIC]
)

# Query with similarity matching
result = await cache.get(
    key="tap the blue button",  # Similar but not exact
    embedding=generate_embedding("tap the blue button")
)
# Returns: (value, CacheLevel.L2_SEMANTIC, similarity_score)
```

### Context-Aware Caching

```python
# Store with context
context = {
    'app_id': 'vscode',
    'workflow': 'debugging',
    'goal': 'set_breakpoint'
}

await cache.put(
    key="add breakpoint",
    value={"action": "click", "line": 42},
    context=context,
    cache_levels=[CacheLevel.L3_CONTEXTUAL]
)

# Retrieve with context
result = await cache.get(
    key="set debug point",
    context=context
)
```

### Integration with Vision Analyzer

```python
# In claude_vision_analyzer_main.py
analyzer = ClaudeVisionAnalyzer()

# Semantic cache is automatically used when enabled
result = await analyzer.analyze_screenshot(
    image,
    "Click the save button",
    use_cache=True
)

# The analyzer will:
# 1. Generate embedding for the prompt
# 2. Check all cache levels
# 3. Return cached result if similarity > threshold
# 4. Otherwise, call API and cache result
```

## Configuration

### Environment Variables

```bash
# Cache sizes (MB)
JARVIS_L1_CACHE_SIZE=20
JARVIS_L2_CACHE_SIZE=100
JARVIS_L3_CACHE_SIZE=80
JARVIS_L4_CACHE_SIZE=50

# TTL settings (seconds)
JARVIS_L1_TTL=30
JARVIS_L2_TTL=300
JARVIS_L3_TTL=1800
JARVIS_L4_TTL_DYNAMIC=true

# LSH parameters
JARVIS_LSH_NUM_TABLES=12
JARVIS_LSH_HASH_SIZE=10
JARVIS_SIMILARITY_THRESHOLD=0.85

# Features
JARVIS_ENABLE_SIMD=true
JARVIS_ENABLE_PREDICTIVE_CACHE=true
```

### Programmatic Configuration

```python
# Configure semantic cache
analyzer._semantic_cache_config = {
    'enabled': True,
    'similarity_threshold': 0.85,
    'use_embeddings': True,
    'cache_levels': ['L1', 'L2', 'L3'],
    'memory_allocation': {
        'l1': 20 * 1024 * 1024,  # 20MB
        'l2': 100 * 1024 * 1024,  # 100MB
        'l3': 80 * 1024 * 1024,   # 80MB
        'l4': 50 * 1024 * 1024    # 50MB
    }
}
```

## Performance Characteristics

### Cache Hit Rates (Typical)
- L1 (Exact): 15-25% hit rate, <0.1ms latency
- L2 (Semantic): 35-50% hit rate, 1-5ms latency
- L3 (Contextual): 20-30% hit rate, 0.5-2ms latency
- L4 (Predictive): 10-20% hit rate, varies

### Memory Efficiency
- Automatic eviction when approaching limits
- Value-based scoring for intelligent eviction
- Compression for large cached values

### Throughput
- 10,000+ queries/second for L1 cache
- 1,000+ similarity searches/second for L2
- Parallel batch processing support

## Advanced Features

### 1. **Anomaly Detection Integration**
Cache automatically bypasses for anomalous patterns to ensure fresh analysis.

### 2. **Goal System Integration**
Contextual cache leverages goal hierarchy for intelligent matching.

### 3. **Memory Pressure Handling**
Swift implementation monitors system memory and adjusts cache size dynamically.

### 4. **Cross-Language Optimization**
- Python: ML model integration, high-level orchestration
- Rust: SIMD operations, concurrent data structures
- Swift: Native macOS features, Core ML

## Monitoring and Statistics

```python
# Get comprehensive statistics
stats = cache.get_statistics()

# Output:
{
    'total_requests': 10000,
    'cache_bypass_count': 150,
    'overall_hit_rate': 0.65,
    'memory_usage_mb': 178.5,
    'layers': {
        'L1_EXACT': {
            'entries': 1500,
            'hits': 2500,
            'misses': 7500,
            'hit_rate': 0.25,
            'current_size_mb': 18.2
        },
        'L2_SEMANTIC': {
            'entries': 5000,
            'hits': 4000,
            'misses': 6000,
            'hit_rate': 0.40,
            'current_size_mb': 95.3
        }
        # ... other layers
    }
}
```

## Best Practices

1. **Embedding Quality**: Use high-quality embeddings for better semantic matching
2. **Context Design**: Include relevant context keys for improved L3 performance
3. **TTL Tuning**: Adjust TTL based on data volatility
4. **Monitoring**: Regularly check hit rates and adjust thresholds
5. **Memory Management**: Monitor cache sizes and eviction rates

## Troubleshooting

### Low Hit Rates
- Check similarity threshold (default: 0.85)
- Verify embedding quality
- Analyze query patterns

### High Memory Usage
- Review eviction policies
- Check for memory leaks in cached values
- Adjust cache size allocations

### Performance Issues
- Enable SIMD optimizations
- Reduce LSH table count for faster lookups
- Use batch operations where possible

## Future Enhancements

1. **Distributed Caching**: Redis backend for multi-instance sharing
2. **Advanced Embeddings**: Integration with GPT embeddings
3. **Smart Prefetching**: ML-based prediction improvements
4. **Compression**: Automatic value compression for large entries
5. **Persistence**: Optional disk-based cache persistence