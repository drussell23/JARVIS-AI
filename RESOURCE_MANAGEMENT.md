# JARVIS Resource Management System

## 🎯 Overview

The Resource Management System ensures JARVIS runs efficiently on 16GB MacBook systems by enforcing strict memory and CPU limits. The system targets **70% maximum memory usage** (11.2GB on 16GB systems), leaving 4.8GB free for other applications.

## 🔑 Key Features

### 1. **Strict Memory Control**
- **Target**: Keep total system memory ≤ 70%
- **JARVIS Limit**: 2GB maximum for all JARVIS processes
- **ML Model Limit**: 400MB for machine learning models
- **One Model Rule**: Only ONE ML model loaded at a time

### 2. **CPU Throttling**
- **Threshold**: 50% CPU usage triggers throttling
- **Levels**: 0 (none) → 3 (severe)
- **Delays**: 0ms → 100ms → 500ms → 1000ms

### 3. **Predictive Loading**
- Tracks ML model usage patterns
- Predicts next likely model
- Preloads only if memory available

### 4. **Emergency Measures**
- At 85% memory: Force unload all models
- At 90% memory: Recommend JARVIS restart
- Automatic garbage collection

## 📊 Memory Budget (16GB System)

| Component | Memory | Purpose |
|-----------|--------|---------|
| System (70%) | 11.2GB | Total allowed |
| JARVIS Core | 2.0GB | Main processes |
| ML Models | 400MB | One model at a time |
| Voice Cache | 150MB | Audio processing |
| Other Services | 450MB | Vision, cleanup, etc |
| **Free (30%)** | **4.8GB** | **Other apps** |

## 🚦 How It Works

### Model Loading Flow
```
1. Request model load
   ↓
2. Check memory (<65%?)
   ↓
3. Unload current model
   ↓
4. Load new model
   ↓
5. Update tracking
```

### Throttling Behavior
- **Level 0**: Normal operation
- **Level 1**: 100ms delay between operations
- **Level 2**: 500ms delay, reduced cache
- **Level 3**: 1s delay, disable predictive loading

## 🛠️ Integration

The resource manager is automatically activated when JARVIS starts:

```python
# In backend/__init__.py
from .resource_manager import get_resource_manager

# In ML Manager
if not rm.request_ml_model(model_id):
    raise MemoryError("Resource manager denied loading")
```

## 📈 Monitoring

Check resource status anytime:

```python
from backend import get_backend_status
status = get_backend_status()
print(f"Memory: {status['resources']['memory_percent']}%")
print(f"Throttle: {status['resources']['throttle_level']}")
```

## 🚨 Troubleshooting

### High Memory Usage
- Resource manager will automatically unload models
- Check which services are active
- Consider restarting JARVIS if >90%

### ML Model Denied
- System memory is above 65%
- Wait for memory to free up
- Check for memory leaks

### Performance Degraded
- High throttle level active
- Close other applications
- Check CPU temperature

## ✅ Benefits

1. **Stability**: Prevents memory crashes
2. **Performance**: Maintains responsive system
3. **Efficiency**: Loads only what's needed
4. **Smart**: Learns usage patterns
5. **Automatic**: No manual intervention

## 🔧 Configuration

Default limits in `resource_manager.py`:

```python
MAX_MEMORY_PERCENT = 70.0    # System target
MAX_JARVIS_MEMORY_MB = 2048  # 2GB for JARVIS
MAX_ML_MEMORY_MB = 400       # 400MB for ML
MAX_CPU_PERCENT = 50.0       # CPU throttle threshold
```

## 📝 Example Output

```
🤖 JARVIS Resource Manager Active
  Memory: 68.5% (within target ✅)
  CPU: 35.2%
  Throttle: Level 0 (none)
  ML Model: voice_auth_1
  JARVIS Memory: 856MB

⚠️ Memory pressure detected at 75.2%
  → Unloading model: voice_auth_1
  → Freed: 385MB
  → New memory: 69.8% ✅
```

## 🎯 Goal Achievement

With this system, JARVIS on your 16GB MacBook will:
- ✅ Stay under 70% memory usage
- ✅ Load only one ML model at a time
- ✅ Throttle operations during high CPU
- ✅ Predict and optimize model loading
- ✅ Prevent system crashes from memory exhaustion

The resource manager ensures JARVIS is a good citizen on your system, using resources efficiently while maintaining performance!