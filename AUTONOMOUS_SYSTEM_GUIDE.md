# JARVIS Autonomous System Guide

## Overview

The JARVIS Autonomous System is a revolutionary zero-configuration architecture that automatically discovers, connects, and maintains all services with no hardcoding. It's optimized for 16GB macOS systems with intelligent memory management.

## Key Features

### 🎯 Zero Configuration
- **No hardcoded ports or URLs** - Everything is discovered dynamically
- **Automatic service registration** - Services announce themselves
- **Dynamic endpoint discovery** - APIs are found automatically
- **Self-configuring frontend** - React app gets config at runtime

### 🔧 Self-Healing Capabilities
- **ML-powered recovery strategies** - Learns from failures
- **Automatic port relocation** - Services move if ports are blocked
- **Health monitoring** - Continuous health checks with scoring
- **Intelligent failover** - Automatic switching to healthy services

### 🧠 Intelligent Orchestration
- **Service mesh architecture** - All components interconnected
- **Pattern learning** - System learns optimal configurations
- **Memory-aware operations** - Stays under 4GB total usage
- **Performance optimization** - Routes optimized based on latency

## Quick Start

```bash
# Start JARVIS with autonomous mode (default if components available)
./start_system.py

# Or explicitly enable autonomous mode
./start_system.py --autonomous

# Use traditional mode without autonomous features
./start_system.py --no-autonomous

# Other options work as before
./start_system.py --backend-only  # Just the backend
./start_system.py --no-browser    # Don't open browser
```

## Architecture

### 1. Autonomous Orchestrator (`backend/core/autonomous_orchestrator.py`)
- Discovers services on the network
- Monitors health and performance
- Applies healing strategies
- Manages service registry

### 2. Zero Config Mesh (`backend/core/zero_config_mesh.py`)
- Creates service interconnections
- Routes messages between services
- Handles failover automatically
- Optimizes communication paths

### 3. Dynamic Config Service (`frontend/src/services/DynamicConfigService.js`)
- Frontend service discovery
- Automatic backend connection
- Real-time configuration updates
- Self-healing connections

### 4. ML Audio Handler (`frontend/src/utils/DynamicMLAudioHandler.js`)
- Zero-config WebSocket connections
- Automatic reconnection
- Intelligent error recovery
- Performance monitoring

## How It Works

### Service Discovery Flow

1. **Startup Phase**
   - Orchestrator starts and begins port scanning
   - Common ports checked first (3000, 8000, 8010, etc.)
   - Services identified by response patterns

2. **Registration Phase**
   - Discovered services registered with metadata
   - Health endpoints identified
   - Capabilities and dependencies mapped

3. **Connection Phase**
   - Service mesh establishes connections
   - Dependencies automatically linked
   - Optimal routes calculated

4. **Monitoring Phase**
   - Continuous health checks
   - Performance metrics collected
   - Anomalies detected and handled

### Self-Healing Process

1. **Detection**
   - Health score drops below threshold
   - Connection failures detected
   - Performance degradation noticed

2. **Diagnosis**
   - Error patterns analyzed
   - Historical data consulted
   - Best strategy selected

3. **Healing**
   - Strategy applied (restart, relocate, etc.)
   - Success measured
   - Learning updated

4. **Verification**
   - Health rechecked
   - Performance validated
   - System stabilized

## Frontend Integration

The frontend automatically discovers and connects to backend services:

```javascript
// No hardcoding needed!
import configService from './services/DynamicConfigService';

// Wait for discovery
await configService.waitForConfig();

// Use discovered endpoints
const apiUrl = configService.getApiUrl('ml_audio_config');
const wsUrl = configService.getWebSocketUrl('audio/ml/stream');
```

## Backend Integration

Add the autonomous service API to your FastAPI app:

```python
from backend.api.autonomous_service_api import router as autonomous_router

app.include_router(autonomous_router)
```

## Memory Management

The system intelligently manages memory across all components:

- **Orchestrator**: 512MB limit
- **Service Mesh**: 256MB limit  
- **Backend**: 2GB limit
- **Frontend**: 1GB limit
- **Total System**: 4GB target

## Monitoring

### Real-time Status
- Access `/services/discovery` for all discovered services
- WebSocket at `/services/monitor` for live updates
- Health checks at `/services/health/{service_name}`

### Diagnostics
- Full diagnostics at `/services/diagnostics`
- Healing history included
- Performance metrics available

## Troubleshooting

### Service Not Discovered
1. Check if service is running on expected ports
2. Verify health endpoint responds
3. Check firewall settings

### High Memory Usage
1. System automatically triggers optimization
2. Check `/services/diagnostics` for details
3. Adjust memory limits if needed

### Connection Issues
1. Service mesh automatically retries
2. Check health scores
3. Manual healing: POST to `/services/heal/{service_name}`

## Advanced Features

### Custom Service Registration
```python
await orchestrator.register_service(ServiceInfo(
    name="custom_service",
    port=9000,
    endpoints={"health": "/health", "api": "/api"}
))
```

### Event Subscriptions
```javascript
configService.on('service-relocated', (event) => {
    console.log(`Service moved from port ${event.oldPort} to ${event.newPort}`);
});
```

### Pattern Learning
The system learns from successes and failures:
- Successful healing strategies get higher priority
- Failed patterns are avoided
- Optimization improves over time

## Benefits

1. **Zero DevOps** - No configuration files to maintain
2. **Resilient** - Automatically recovers from failures
3. **Adaptive** - Learns and improves over time
4. **Efficient** - Optimized for available resources
5. **Transparent** - Full visibility into system state

## Future Enhancements

- Kubernetes integration for cloud deployment
- Multi-node clustering support
- Advanced ML prediction models
- GraphQL endpoint discovery
- Service dependency visualization

Start your journey with truly autonomous systems today! 🚀