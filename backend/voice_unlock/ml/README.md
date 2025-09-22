# ML Optimization for Voice Unlock System

## Overview

This module implements a memory-optimized machine learning system for voice authentication, specifically designed for 16GB RAM macOS systems. It features dynamic model loading, intelligent caching, model quantization, and comprehensive performance monitoring.

## Key Features

### 1. **Dynamic Model Loading/Unloading** (`ml_manager.py`)
- **Lazy Loading**: Models are loaded only when needed
- **Automatic Unloading**: Unused models are automatically unloaded after a timeout
- **Memory Mapping**: Large model files are memory-mapped for efficient access
- **LRU Cache**: Recently used models are kept in memory for fast access

### 2. **Model Optimization** (`ml_manager.py`)
- **Quantization**: Reduces model precision from float32 to float16 for 50% memory savings
- **Compression**: Model pruning and compression techniques
- **PCA Dimensionality Reduction**: Reduces feature dimensions from ~100 to 50
- **Lightweight Models**: Uses OneClassSVM instead of deep neural networks

### 3. **Memory Management** (`ml_manager.py`)
- **Memory Limits**: Configurable maximum memory usage (default: 500MB)
- **Cache Size Limits**: Separate limit for model cache (default: 200MB)
- **Automatic Cleanup**: Triggered when memory usage exceeds thresholds
- **Memory Monitoring**: Real-time tracking of memory usage

### 4. **Performance Monitoring** (`performance_monitor.py`)
- **Real-time Metrics**: CPU, memory, inference time tracking
- **Model-specific Metrics**: Per-model performance tracking
- **Alert System**: Configurable thresholds with callback support
- **Performance Reports**: Exportable JSON reports with recommendations

### 5. **Optimized Authentication** (`optimized_voice_auth.py`)
- **Feature Caching**: Extracted features are cached for repeated use
- **Batch Processing**: Multiple authentications can be processed efficiently
- **Adaptive Thresholds**: Authentication thresholds adjust based on conditions
- **Continuous Learning**: Models can be updated with new samples

## Architecture

```
ml/
├── __init__.py              # Module exports
├── ml_manager.py           # Core ML model manager with caching
├── optimized_voice_auth.py # Optimized voice authentication
├── performance_monitor.py   # Performance monitoring system
├── ml_integration.py       # Integration layer
├── test_ml_optimization.py # Comprehensive test suite
└── README.md              # This file
```

## Usage

### Basic Integration

```python
from backend.voice_unlock.ml import VoiceUnlockMLSystem

# Initialize the ML system
ml_system = VoiceUnlockMLSystem()

# Enroll a user
audio_samples = [...]  # List of numpy arrays
result = ml_system.enroll_user("user_id", audio_samples)

# Authenticate a user
audio_data = ...  # Numpy array
auth_result = ml_system.authenticate_user("user_id", audio_data)

# Get performance report
report = ml_system.get_performance_report()
print(f"Memory usage: {report['system_health']['ml_memory_mb']}MB")

# Cleanup when done
ml_system.cleanup()
```

### Configuration

The system uses the global voice unlock configuration with these key settings:

```python
# Maximum memory for ML models (MB)
max_memory_mb = 500

# Maximum cache size (MB)
cache_size_mb = 200

# Enable/disable optimizations
use_gpu = False  # GPU disabled by default
enable_quantization = True
enable_compression = True

# Performance limits
max_cpu_percent = 25
```

### Memory Optimization Strategies

1. **Model Lifecycle**:
   - Models are loaded on-demand during authentication
   - Unused models are unloaded after 5 minutes (configurable)
   - Cache evicts least-recently-used models when full

2. **Feature Optimization**:
   - PCA reduces features from ~100 to 50 dimensions
   - Features are cached to avoid recomputation
   - Batch processing reduces overhead

3. **Quantization**:
   - Float32 → Float16 conversion saves 50% memory
   - Minimal impact on accuracy (~1-2% reduction)
   - Faster inference on modern CPUs

4. **Resource Monitoring**:
   - Background thread monitors memory usage
   - Automatic cleanup triggered at 80% memory usage
   - Degraded mode activated under high load

## Performance Characteristics

Based on testing with synthetic data:

- **Memory per User**: ~10-15MB (model + features)
- **Inference Time**: 20-50ms (cached), 100-200ms (cold start)
- **Cache Hit Rate**: >70% in typical usage
- **Maximum Users**: ~30-40 on 16GB system
- **CPU Usage**: <25% during authentication

## Testing

Run the comprehensive test suite:

```bash
cd backend/voice_unlock/ml
python test_ml_optimization.py
```

This will:
1. Test memory efficiency with multiple users
2. Verify dynamic loading/unloading
3. Measure cache performance
4. Compare quantization impact
5. Run stress tests
6. Generate performance reports

## Monitoring and Diagnostics

### Real-time Monitoring

```python
# Get current stats
stats = ml_system.get_performance_report()

# Monitor specific metrics
monitor = ml_system.monitor
history = monitor.get_metric_history('inference_time_ms', duration_minutes=60)

# Set up alerts
def alert_handler(alert):
    print(f"Alert: {alert['metric']} = {alert['value']}")
    
monitor.add_alert_callback(alert_handler)
```

### Export Diagnostics

```python
# Export full diagnostics
ml_system.export_diagnostics("diagnostics.json")

# Export performance report
monitor.export_report("performance_report.json")
```

## Troubleshooting

### High Memory Usage

1. Check active model count:
   ```python
   stats = ml_system.get_performance_report()
   print(f"Active models: {stats['system_stats']['models']['active']}")
   ```

2. Force cleanup:
   ```python
   ml_system._cleanup_resources()
   ```

3. Reduce cache size:
   ```python
   ml_system.ml_manager.cache.max_memory = 100 * 1024 * 1024  # 100MB
   ```

### Slow Inference

1. Check cache hit rate:
   ```python
   print(f"Cache hit rate: {ml_system.ml_manager._get_cache_hit_rate()}%")
   ```

2. Enable quantization:
   ```python
   ml_system.ml_manager.config['enable_quantization'] = True
   ```

3. Reduce feature dimensions:
   ```python
   ml_system.authenticator.pca_components = 30  # From 50
   ```

### Model Loading Errors

1. Check model paths:
   ```python
   for user_id, model in ml_system.authenticator.user_models.items():
       print(f"{user_id}: {model.model_path.exists()}")
   ```

2. Clear corrupted models:
   ```python
   ml_system.authenticator.remove_user("user_id")
   ```

## Future Improvements

1. **GPU Acceleration**: Enable Metal Performance Shaders for M1/M2 Macs
2. **Model Distillation**: Create smaller models from larger ones
3. **Federated Learning**: Privacy-preserving model updates
4. **Edge TPU Support**: Hardware acceleration for inference
5. **Advanced Caching**: Predictive model loading based on usage patterns

## Contributing

When adding new ML features:

1. Always consider memory impact
2. Add performance metrics
3. Test with memory constraints
4. Document configuration options
5. Include stress tests