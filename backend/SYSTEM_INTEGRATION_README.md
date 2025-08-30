# JARVIS System Integration & Performance (Phase 0D)

## Overview

The System Integration & Performance layer ensures all JARVIS components work seamlessly together under the strict memory constraints of a 16GB RAM macOS system. This implementation provides dynamic resource management, intelligent error recovery, and graceful degradation strategies without any hardcoding.

## Key Features

### 1. **Unified Resource Management** (`unified_resource_manager.py`)
- Dynamic memory allocation across components
- Intelligent resource prioritization
- Real-time memory pressure monitoring
- Automatic resource rebalancing
- Support for component restart policies

### 2. **Cross-Component Memory Sharing** (`cross_component_memory_sharing.py`)
- Shared model registry for ML models
- Memory buffer pools for audio/image processing
- LRU cache with configurable eviction policies
- Computation task queue for resource-intensive operations
- Automatic memory deduplication

### 3. **Component Health Monitoring** (`component_health_monitor.py`)
- Real-time health status tracking
- Custom health checks per component
- Automatic failure detection
- Recovery strategy execution
- Health score calculation

### 4. **Dynamic Configuration Management** (`dynamic_config_manager.py`)
- Hot-reload configuration changes
- Multiple configuration sources (files, env vars, learned)
- Schema validation
- Configuration optimization based on usage patterns
- Automatic rollback on invalid changes

### 5. **Graceful Degradation Strategies** (`graceful_degradation_manager.py`)
- Feature priority system
- Progressive feature disabling
- Recovery planning
- Dependency management
- Emergency shutdown capabilities

### 6. **Comprehensive Error Recovery** (`error_recovery_system.py`)
- Intelligent error categorization
- Circuit breaker pattern implementation
- Retry strategies with exponential backoff
- Error pattern detection
- Cascading failure prevention

### 7. **Performance Monitoring Dashboard** (`performance_dashboard.py`)
- Real-time metrics visualization
- WebSocket-based live updates
- Alert management
- System status overview
- Historical data tracking

### 8. **System Integration Coordinator** (`system_integration_coordinator.py`)
- Central orchestration of all systems
- Cross-system optimization
- Emergency handling
- State persistence
- Unified reporting

## Quick Start

### 1. Launch JARVIS with Full Optimization

```bash
cd backend
./start_jarvis_optimized.py
```

This will start:
- Event-driven JARVIS core at http://localhost:8888
- Performance Dashboard at http://localhost:8889
- All resource management systems

### 2. Configuration

The system uses `backend/config/resource_management_config.yaml` as the base configuration. Key settings:

```yaml
system:
  total_memory_gb: 16          # Total system RAM
  jarvis_max_memory_gb: 12     # Maximum for JARVIS
  target_usage_percent: 60     # Target memory usage

components:
  voice:
    priority: 1                # Highest priority
    min_memory_mb: 512
    max_memory_mb: 3072
```

### 3. Environment Variables

- `JARVIS_USER`: User name (default: "Sir")
- `JARVIS_DEBUG`: Enable debug mode
- `JARVIS_PERFORMANCE_DASHBOARD`: Enable dashboard
- `ANTHROPIC_API_KEY`: Required for Claude integration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   System Integration Coordinator             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Resource   │  │    Health    │  │   Performance    │  │
│  │   Manager    │  │   Monitor    │  │   Dashboard      │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │            │
│  ┌──────┴──────┐  ┌──────┴───────┐  ┌────────┴─────────┐  │
│  │   Memory     │  │    Error     │  │   Degradation    │  │
│  │   Sharing    │  │   Recovery   │  │   Manager        │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │            │
│  ┌──────┴─────────────────┴────────────────────┴────────┐  │
│  │                    Event Bus                          │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
└─────────────────────────┼───────────────────────────────────┘
                          │
     ┌────────────────────┼───────────────────┐
     │                    │                   │
┌────┴─────┐      ┌───────┴──────┐   ┌───────┴──────┐
│  Voice   │      │    Vision    │   │   Control    │
│  System  │      │    System    │   │   System     │
└──────────┘      └──────────────┘   └──────────────┘
```

## Memory Management Strategy

### 1. **Priority-Based Allocation**
- Components assigned priorities (0=critical, 5=optional)
- Memory allocated based on priority during pressure
- Critical components never reduced below minimum

### 2. **Degradation Levels**
- **Full**: All features enabled
- **Reduced**: Non-essential features disabled
- **Minimal**: Only core functionality

### 3. **Memory Pressure Thresholds**
- Normal: < 60% usage
- High: 60-70% usage (reduce features)
- Critical: > 85% usage (minimal mode)

## Feature Management

### Voice System Features
1. **Deep Learning** (High Priority)
   - Wake word neural network
   - Advanced NLP processing
   - Memory: 1024MB

2. **Personalization** (Normal Priority)
   - User preference learning
   - Custom wake words
   - Memory: 512MB

3. **Adaptation** (Low Priority)
   - Real-time model updates
   - Noise adaptation
   - Memory: 256MB

### Vision System Features
1. **Claude Analysis** (High Priority)
   - AI-powered screen analysis
   - Memory: 512MB

2. **Multi-Window** (Normal Priority)
   - Multiple window tracking
   - Memory: 256MB

3. **Caching** (Low Priority)
   - Screenshot caching
   - Memory: 512MB

## Monitoring & Debugging

### Performance Dashboard

Access http://localhost:8889 to view:
- Real-time memory usage
- Component health status
- Active errors and alerts
- Degradation state
- System metrics

### Event Web UI

Access http://localhost:8888 to view:
- Real-time event stream
- Component interactions
- Performance metrics
- Debug information

### Health Checks

The system performs automatic health checks:
- Heartbeat monitoring (every 30s)
- Memory usage tracking
- Response time monitoring
- Error rate calculation

## Error Recovery Strategies

### 1. **Memory Errors**
- Strategy: Degrade → Clear caches → Restart component
- Circuit breaker: 5 failures in 60s

### 2. **Network Errors**
- Strategy: Retry with backoff → Fallback mode
- Max retries: 5 with exponential backoff

### 3. **Component Failures**
- Strategy: Restart → Isolate → Escalate
- Max restart attempts: 3 (configurable)

### 4. **API Errors**
- Strategy: Fallback to offline mode
- Circuit breaker: 5 failures in 300s

## Configuration Examples

### Adjust Memory Limits

```yaml
# In resource_management_config.yaml
system:
  jarvis_max_memory_gb: 10  # Reduce if other apps need more

components:
  voice:
    max_memory_mb: 2048  # Reduce voice memory allocation
```

### Custom Degradation Strategy

```yaml
degradation:
  triggers:
    - type: "memory_pressure"
      threshold: "high"
      action: "reduce_features"
      features_to_disable:
        - "voice.adaptation"
        - "vision.multi_window"
```

### Health Check Configuration

```yaml
health_monitoring:
  checks:
    memory_usage:
      warning_percent: 65   # Earlier warning
      critical_percent: 80  # Lower critical threshold
```

## Troubleshooting

### High Memory Usage

1. Check dashboard for component memory usage
2. Review degradation state
3. Manually trigger recovery:
   ```python
   # In Python console
   from backend.core.graceful_degradation_manager import get_degradation_manager
   await get_degradation_manager().attempt_recovery()
   ```

### Component Not Responding

1. Check health status in dashboard
2. Review error logs
3. Force component restart via event

### Configuration Not Loading

1. Check file syntax (YAML validation)
2. Review validation errors in logs
3. Check file permissions

## Performance Tips

### 1. **Optimize Component Features**
- Disable unused features in config
- Adjust memory allocations based on usage
- Enable aggressive garbage collection

### 2. **Memory Sharing**
- Share models between components when possible
- Use buffer pools for audio/image processing
- Enable cache compression

### 3. **Monitoring**
- Keep dashboard open during heavy usage
- Set up alerts for critical thresholds
- Review performance metrics regularly

## Emergency Procedures

### Emergency Shutdown

If system becomes unresponsive:

```python
# Trigger emergency shutdown
from backend.core.system_integration_coordinator import SystemIntegrationCoordinator
coordinator = SystemIntegrationCoordinator()
await coordinator.handle_emergency("manual_trigger")
```

This will:
1. Disable all non-essential features
2. Clear all caches
3. Apply emergency configuration
4. Force garbage collection

### Manual Recovery

To manually recover from degraded state:

1. Stop non-JARVIS applications
2. Wait for memory to free
3. Trigger recovery from dashboard
4. Monitor health status

## Future Enhancements

### Swift Performance Bridges (Planned)
- Native Swift implementations for:
  - Audio processing (VAD, noise reduction)
  - System monitoring (memory, CPU tracking)
  - UI automation (accessibility API)

### Advanced Features (Planned)
- Machine learning for predictive resource allocation
- Distributed processing support
- Cloud offloading for heavy tasks
- Advanced anomaly detection

## Contributing

When adding new components or features:

1. Register with resource manager
2. Define degradation levels
3. Implement health checks
4. Add error recovery strategies
5. Update dashboard metrics

Remember: **No hardcoding!** All limits, thresholds, and configurations must be dynamic.