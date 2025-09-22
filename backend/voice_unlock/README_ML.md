# JARVIS Voice Unlock with ML Optimization

## Overview

The JARVIS Voice Unlock system now features comprehensive ML optimization designed specifically for 16GB RAM systems. The system implements extremely lazy loading, predictive model caching, dynamic memory management, and intelligent resource allocation.

## Key ML Optimizations

### 1. **Extreme Lazy Loading**
- Models are **never** loaded until actually needed
- Lazy loading for feature extractors and anti-spoofing components
- Deferred initialization of all ML components
- Thread-safe concurrent loading with timeout support

### 2. **Predictive Model Loading**
- Tracks user access patterns over time
- Predicts which models are likely to be needed
- Preloads models during idle time based on predictions
- Time-based likelihood calculations (e.g., user typically authenticates at 9am)

### 3. **Dynamic Memory Management**
- Aggressive memory cleanup when usage exceeds 80%
- Automatic model unloading based on:
  - Time since last access
  - Prediction score
  - Current memory pressure
- Memory-mapped file support for large models

### 4. **Intelligent Caching**
- LRU cache with configurable size limits
- Cache hit rate tracking and optimization
- Automatic cache eviction based on memory pressure
- Separate caches for models, scalers, and PCA components

### 5. **Model Optimization**
- Automatic quantization (float32 → float16)
- PCA dimensionality reduction (100 → 50 features)
- Lightweight OneClassSVM instead of deep neural networks
- Model compression and pruning

## Architecture

```
voice_unlock/
├── ml/                          # ML optimization module
│   ├── ml_manager.py           # Core ML manager with lazy loading
│   ├── optimized_voice_auth.py # Optimized authenticator
│   ├── performance_monitor.py   # Real-time monitoring
│   └── ml_integration.py       # Integration layer
├── voice_unlock_integration.py  # Main integration
├── jarvis_voice_unlock.py      # CLI and service
└── config.py                   # Dynamic configuration
```

## Installation

```bash
cd backend/voice_unlock
./install.sh
```

This will:
1. Install Python dependencies
2. Build the proximity service
3. Set up LaunchAgent for macOS
4. Create necessary directories
5. Configure permissions

## Usage

### Command Line Interface

```bash
# Enroll a new user
jarvis-voice-unlock enroll john

# Test authentication
jarvis-voice-unlock test

# Show system status
jarvis-voice-unlock status

# Configure settings
jarvis-voice-unlock configure
```

### Python API

```python
from backend.voice_unlock.voice_unlock_integration import create_voice_unlock_system

# Create and start system
system = await create_voice_unlock_system()

# Enroll user
result = await system.enroll_user("john", audio_samples)

# Authenticate
auth_result = await system.authenticate_with_voice(timeout=10.0)

# Get status
status = system.get_status()

# Cleanup
await system.stop()
```

## Configuration

### Memory Settings

Edit environment variables or use the configuration API:

```bash
# Maximum memory for ML models (MB)
export VOICE_UNLOCK_MAX_MEMORY=400

# Cache size (MB)
export VOICE_UNLOCK_CACHE_SIZE=150

# Enable GPU (not recommended for 16GB systems)
export VOICE_UNLOCK_USE_GPU=false
```

### Performance Tuning

```python
# config.py settings
performance:
  max_memory_mb: 400          # Total ML memory limit
  cache_size_mb: 150          # Model cache size
  max_cpu_percent: 25         # CPU usage limit
  background_monitoring: true  # Enable monitoring

# ML Manager settings
ml_config = {
    'enable_lazy_loading': True,
    'enable_predictive_loading': True,
    'preload_threshold': 0.7,      # Preload if >70% likely
    'aggressive_unload_timeout': 60,  # Unload after 60s
    'enable_quantization': True,
    'enable_mmap': True
}
```

## Memory Usage

### Per-User Memory Footprint
- Model: ~5-10MB (quantized)
- Scaler: ~1MB
- PCA: ~2MB (optional)
- Features: ~1-2MB per enrollment
- **Total**: ~10-15MB per user

### System Memory Usage
- Base system: ~50MB
- Feature extractors: ~30MB (lazy loaded)
- Anti-spoofing: ~20MB (lazy loaded)
- Cache overhead: ~10MB
- **Total**: ~100-150MB + user models

### Maximum Capacity (16GB System)
- Available for ML: ~500MB
- Maximum users: 30-40
- With aggressive cleanup: 50-60

## Performance Characteristics

### Model Loading
- Cold start: 100-200ms
- Cached: 20-50ms
- Lazy loaded: 150-300ms (first access)
- Preloaded: 0ms (instant)

### Authentication Time
- Feature extraction: 50-100ms
- Model inference: 20-50ms
- Anti-spoofing: 30-50ms
- **Total**: 100-200ms

### Resource Usage
- CPU: <25% during authentication
- Memory: Dynamically managed
- Disk I/O: Minimal (memory-mapped)

## Monitoring

### Real-time Monitoring

The system includes comprehensive performance monitoring:

```python
# Get performance report
report = system.ml_system.get_performance_report()

# Monitor specific metrics
monitor = system.ml_system.monitor
history = monitor.get_metric_history('inference_time_ms', 60)

# Set up alerts
def alert_handler(alert):
    print(f"Alert: {alert['metric']} = {alert['value']}")
    
monitor.add_alert_callback(alert_handler)
```

### Metrics Tracked
- Memory usage (system, ML, cache)
- CPU usage
- Model load times
- Inference times
- Cache hit rates
- Authentication success rates

### Performance Alerts
- High memory usage (>80%)
- High CPU usage (>50%)
- Slow inference (>100ms)
- Low cache hit rate (<70%)

## Troubleshooting

### High Memory Usage

1. Check current memory status:
   ```bash
   jarvis-voice-unlock status
   ```

2. Force cleanup:
   ```python
   system.ml_system._cleanup_resources()
   ```

3. Reduce limits:
   ```bash
   export VOICE_UNLOCK_MAX_MEMORY=300
   export VOICE_UNLOCK_CACHE_SIZE=100
   ```

### Slow Performance

1. Check cache performance:
   ```python
   report = system.ml_system.get_performance_report()
   print(f"Cache hit rate: {report['ml_performance']['cache']['hit_rate']}%")
   ```

2. Enable more aggressive caching:
   ```python
   config.performance.cache_size_mb = 200
   ```

3. Reduce feature dimensions:
   ```python
   system.ml_system.authenticator.pca_components = 30
   ```

### Authentication Failures

1. Check model status:
   ```python
   for user_id, model in system.ml_system.authenticator.user_models.items():
       print(f"{user_id}: {model.model_path.exists()}")
   ```

2. Re-enroll user:
   ```bash
   jarvis-voice-unlock enroll username
   ```

## Best Practices

1. **Memory Management**
   - Monitor memory usage regularly
   - Set appropriate limits for your system
   - Enable aggressive cleanup for constrained systems

2. **Model Management**
   - Enroll users during off-peak hours
   - Regularly clean up unused models
   - Use quantization for all models

3. **Performance Optimization**
   - Enable predictive loading for frequent users
   - Increase cache size if memory allows
   - Use PCA for dimensionality reduction

4. **Security**
   - Keep anti-spoofing at 'high' level
   - Enable audit logging
   - Regularly update enrolled voices

## Advanced Features

### Custom Model Loaders

```python
# Register custom lazy loader
def custom_loader():
    # Custom loading logic
    return model

ml_manager.register_lazy_loader(
    "custom_model",
    custom_loader,
    Path("path/to/model"),
    "custom"
)
```

### Predictive Loading Tuning

```python
# Adjust prediction parameters
ml_manager.config['preload_threshold'] = 0.8  # More aggressive
ml_manager.config['max_preload_queue'] = 5    # Preload more models
```

### Memory Profiling

```python
# Export detailed diagnostics
system.ml_system.export_diagnostics("diagnostics.json")

# Analyze memory usage
import json
with open("diagnostics.json") as f:
    diag = json.load(f)
    print(json.dumps(diag['performance_report'], indent=2))
```

## Future Enhancements

1. **GPU Acceleration** (for systems with more RAM)
   - Metal Performance Shaders for M1/M2
   - CUDA support for NVIDIA GPUs

2. **Federated Learning**
   - Privacy-preserving model updates
   - Distributed training across devices

3. **Edge Optimization**
   - Model distillation
   - Neural network pruning
   - INT8 quantization

4. **Advanced Caching**
   - Redis integration for distributed systems
   - Predictive cache warming
   - Multi-level caching

## Contributing

When adding ML features:
1. Always consider memory impact
2. Implement lazy loading where possible
3. Add performance metrics
4. Test with memory constraints
5. Document resource usage