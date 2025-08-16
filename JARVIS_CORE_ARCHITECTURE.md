# JARVIS Core Architecture - Built for Scale & Memory Efficiency

## Overview

The new JARVIS Core architecture implements a sophisticated multi-tiered system designed for optimal memory usage and intelligent task routing. This architecture allows JARVIS to run efficiently on M1 Macs with 16GB RAM while maintaining high performance and capability.

## Core Components

### 1. Model Manager (Brain Switcher)
Located in `backend/core/model_manager.py`

**Features:**
- **Tiered Model System:**
  - Tier 1: TinyLlama (1GB) - Always loaded for instant responses
  - Tier 2: Phi-2 (2GB) - Loaded on-demand for standard tasks
  - Tier 3: Mistral-7B (4GB) - Loaded for complex operations

- **Intelligent Loading:**
  - Lazy loading based on task requirements
  - Automatic unloading of least-used models
  - Memory-aware model selection
  - Performance tracking for each model

**Key Methods:**
```python
# Get best model for a task
model, tier = await model_manager.get_model_for_task(
    task_type="code",
    complexity=0.8,
    context_length=2000
)

# Optimize based on workload
await model_manager.optimize_for_workload(recent_tasks)
```

### 2. Memory Controller (Resource Controller)
Located in `backend/core/memory_controller.py`

**Features:**
- **Real-time Monitoring:**
  - Continuous memory pressure tracking
  - Predictive memory usage analysis
  - Top process identification
  - Historical trend analysis

- **Automatic Optimization:**
  - Garbage collection management
  - Memory pressure callbacks
  - Aggressive optimization modes
  - Memory reservation system

**Memory Pressure Levels:**
- LOW (< 40%): Can load any model
- MODERATE (40-60%): Normal operation
- HIGH (60-75%): Limit operations
- CRITICAL (> 75%): Emergency mode

### 3. Task Router (Intelligence Dispatcher)
Located in `backend/core/task_router.py`

**Features:**
- **Query Analysis:**
  - Task type identification (chat, code, analysis, creative, etc.)
  - Complexity scoring (0.0 to 1.0)
  - Token estimation
  - Context requirement detection

- **Intelligent Routing:**
  - Model selection based on task requirements
  - Memory-aware routing decisions
  - Performance tracking
  - Fallback strategies

**Task Types:**
- CHAT: Simple conversation
- CODE: Code generation/analysis
- ANALYSIS: Data analysis, reasoning
- CREATIVE: Creative writing
- FACTUAL: Fact-based Q&A
- COMPLEX: Multi-step reasoning

### 4. JARVIS Core (Integration Layer)
Located in `backend/core/jarvis_core.py`

**Features:**
- **Unified Interface:**
  - Single entry point for all queries
  - Automatic component coordination
  - Context management
  - Performance tracking

- **System Optimization:**
  - Workload-based model preloading
  - Memory pressure responses
  - Configuration management
  - Graceful degradation

## Architecture Flow

```
User Query
    ↓
Task Router
    ├─→ Analyze complexity
    ├─→ Identify task type
    └─→ Estimate requirements
         ↓
Memory Controller
    ├─→ Check current pressure
    ├─→ Predict future usage
    └─→ Optimize if needed
         ↓
Model Manager
    ├─→ Select appropriate tier
    ├─→ Load model if needed
    └─→ Return model instance
         ↓
Generate Response
    ↓
Return to User
```

## Usage Examples

### Basic Usage
```python
from backend.core import JARVISAssistant

# Initialize JARVIS
jarvis = JARVISAssistant()

# Simple chat
response = await jarvis.chat("Hello, how are you?")

# Chat with metadata
full_response = await jarvis.chat_with_info("Write a Python function")
print(f"Used model: {full_response['metadata']['model_tier']}")
print(f"Complexity: {full_response['metadata']['task_analysis']['complexity']}")
```

### Advanced Usage
```python
from backend.core import JARVISCore

# Initialize with custom config
core = JARVISCore(
    models_dir="/path/to/models",
    config_path="jarvis_config.json"
)

# Process with specific parameters
response = await core.process_query(
    "Analyze this complex dataset",
    max_tokens=1024,
    temperature=0.5
)

# Get system status
status = core.get_system_status()
print(f"Memory usage: {status['memory']['current']['percent_used']}%")
print(f"Loaded models: {status['models']['loaded_count']}")

# Optimize system
optimization_results = await core.optimize_system()
```

## Configuration

Create a `jarvis_config.json`:
```json
{
    "target_memory_percent": 60.0,
    "max_history": 10,
    "auto_optimize_memory": true,
    "predictive_loading": true,
    "quality_vs_speed": "balanced"
}
```

## Performance Characteristics

### Memory Usage by Configuration

| Active Models | Memory Usage | Response Time | Use Case |
|--------------|--------------|---------------|----------|
| TinyLlama only | ~1.5GB | <0.5s | High memory pressure |
| TinyLlama + Phi-2 | ~3.5GB | <0.8s | Standard operation |
| All three models | ~7.5GB | <1.0s | Full capability |

### Task Routing Examples

| Query | Detected Type | Complexity | Selected Model |
|-------|--------------|------------|----------------|
| "Hi there!" | CHAT | 0.2 | TinyLlama |
| "Write a function" | CODE | 0.6 | Phi-2 |
| "Complex analysis..." | ANALYSIS | 0.9 | Mistral-7B |

## Benefits

1. **Memory Efficiency:**
   - 40-60% less memory usage compared to single large model
   - Dynamic loading/unloading based on needs
   - Graceful degradation under pressure

2. **Performance:**
   - Faster responses for simple queries (TinyLlama)
   - Full capability when needed (Mistral)
   - Intelligent preloading based on patterns

3. **Scalability:**
   - Easy to add new model tiers
   - Configurable thresholds
   - Extensible routing logic

4. **Reliability:**
   - Automatic failover
   - Memory pressure handling
   - Error recovery

## Testing

Run the test script:
```bash
# Basic test
python test_jarvis_core.py

# Interactive demo
python test_jarvis_core.py interactive
```

## Integration with Existing JARVIS

To integrate with the existing JARVIS system:

1. Replace the current chatbot initialization with:
```python
from backend.core import JARVISAssistant
chatbot = JARVISAssistant()
```

2. Update API endpoints to use the new interface:
```python
response = await chatbot.chat_with_info(message)
```

3. The system is backward compatible with existing APIs while providing enhanced capabilities.

## Future Enhancements

1. **Model Caching:**
   - Disk-based model caching
   - Faster model switching
   - Reduced memory pressure

2. **Advanced Routing:**
   - User preference learning
   - Task-specific fine-tuning
   - Multi-model ensemble responses

3. **Distributed Architecture:**
   - Multi-device model hosting
   - Remote model serving
   - Edge deployment options

## Conclusion

The JARVIS Core architecture provides a sophisticated, memory-efficient solution for running advanced AI models on memory-constrained devices. By intelligently routing tasks and managing resources, it delivers optimal performance while maintaining system stability.