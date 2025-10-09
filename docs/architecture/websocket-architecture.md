# Unified WebSocket Architecture

## Overview

The Unified WebSocket System resolves all WebSocket endpoint conflicts in the JARVIS AI Agent by introducing a TypeScript-based router that manages all WebSocket connections through a single, intelligent routing system.

## Problem Solved

Previously, there were three conflicting WebSocket endpoints all trying to handle `/ws/vision`:
- `backend/api/vision_websocket.py` - Line 246
- `backend/api/enhanced_vision_api.py` - Line 231  
- `backend/api/vision_api.py` - Line 654

This caused routing conflicts where the backend couldn't determine which handler to use, resulting in connection failures.

## Solution Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│                 │     │                      │     │                 │
│  Frontend Apps  ├────►│  TypeScript Router   ├────►│  Python Backend │
│  (Port 3000)    │ WS  │  (Port 8001)         │ IPC │  (Port 8000)    │
│                 │     │                      │     │                 │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
         │                        │                            │
         │                        ├── /ws/vision ──────────────┤
         │                        ├── /ws/voice ───────────────┤
         │                        ├── /ws/automation ──────────┤
         │                        └── /ws ─────────────────────┘
```

## Key Components

### 1. TypeScript WebSocket Router (`UnifiedWebSocketRouter.ts`)
- **Port**: 8001
- **Features**:
  - Dynamic route discovery and registration
  - Pattern-based routing with fuzzy matching
  - Middleware support for authentication and rate limiting
  - Circuit breaker pattern for fault tolerance
  - Automatic reconnection with exponential backoff

### 2. Python-TypeScript Bridge (`python_ts_bridge.py`)
- **Communication**: ZeroMQ for high-performance IPC
- **Features**:
  - Bidirectional function calls between TypeScript and Python
  - Automatic type conversion and serialization
  - Event publishing/subscribing
  - Message correlation for request/response patterns

### 3. Dynamic WebSocket Client (`DynamicWebSocketClient.ts`)
- **Features**:
  - Automatic endpoint discovery
  - Self-healing connections
  - Message type learning
  - Performance optimization through routing intelligence

### 4. Unified Vision Handler (`unified_vision_handler.py`)
- Consolidates all vision WebSocket handlers
- Provides consistent message routing
- Integrates with all vision subsystems:
  - Claude AI Core
  - Vision System V2
  - Autonomous Action Executor
  - Pattern Learning System

## Configuration

### WebSocket Routes (`websocket-routes.json`)
```json
{
  "routes": [
    {
      "path": "/ws/vision",
      "capabilities": ["vision", "monitoring", "claude"],
      "pythonModule": "backend.api.unified_vision_handler"
    }
  ]
}
```

### Environment Variables
```bash
WEBSOCKET_PORT=8001           # TypeScript router port
PYTHON_BACKEND_URL=http://localhost:8000
ENABLE_DYNAMIC_ROUTING=true
ENABLE_RATE_LIMIT=true
```

## Usage

### Starting the System
```bash
cd backend
./start_unified_backend.sh
```

This script:
1. Builds the TypeScript WebSocket router
2. Starts the TypeScript server on port 8001
3. Starts the Python FastAPI backend on port 8000
4. Establishes the Python-TypeScript bridge

### Frontend Connection
```javascript
// Old way (conflicting)
const ws = new WebSocket('ws://localhost:8000/vision/ws/vision');

// New way (unified)
const ws = new WebSocket('ws://localhost:8001/ws/vision');
```

Or use the provided service:
```javascript
import { getUnifiedWebSocketService } from './services/UnifiedWebSocketService';

const wsService = getUnifiedWebSocketService();
await wsService.connect('vision');
```

## Message Flow Example

1. **Frontend sends message**:
   ```javascript
   ws.send(JSON.stringify({
     type: 'request_workspace_analysis'
   }));
   ```

2. **TypeScript router receives and routes**:
   - Validates message format
   - Applies rate limiting
   - Routes to appropriate handler

3. **Python handler processes**:
   - Captures screenshot
   - Analyzes with Claude AI
   - Composes response

4. **Response flows back**:
   - Through TypeScript router
   - With error handling and retry logic
   - To frontend WebSocket client

## Error Handling

### Circuit Breaker
- Opens after 5 consecutive failures
- Prevents cascading failures
- Automatically recovers after timeout

### Retry Logic
- Exponential backoff with jitter
- Maximum 3 retry attempts
- Only retries transient errors

### Reconnection
- Automatic client reconnection
- Message replay on reconnection
- Connection state persistence

## Performance Features

- **Message batching**: Reduces network overhead
- **Connection pooling**: Reuses WebSocket connections
- **Smart routing**: ML-based endpoint selection
- **Caching**: Response caching for repeated queries
- **Rate limiting**: Prevents system overload

## Testing

Run the comprehensive test suite:
```bash
cd backend
python tests/test_unified_websocket.py
```

Tests include:
- Connection and discovery
- Message routing
- Error handling
- Rate limiting
- Concurrent clients
- Performance benchmarks

## Monitoring

The system provides real-time statistics:
```javascript
const stats = wsService.getStats();
console.log(stats);
// {
//   connections: [...],
//   discoveredEndpoints: [...],
//   totalMessages: 1234,
//   errorRate: 0.02
// }
```

## Migration Guide

For existing code using the old endpoints:

1. **Update WebSocket URLs**:
   ```bash
   node backend/websocket/initialize_frontend.js
   ```

2. **Update message handlers** to use unified format
3. **Test with new endpoint**
4. **Remove old WebSocket route definitions**

## Benefits

1. **No More Conflicts**: Single routing point eliminates conflicts
2. **Better Performance**: TypeScript router optimized for WebSocket handling
3. **Enhanced Reliability**: Circuit breakers and retry logic
4. **Easier Debugging**: Centralized logging and monitoring
5. **Future-Proof**: Easy to add new routes and capabilities
6. **Type Safety**: TypeScript ensures message format consistency

## Future Enhancements

- GraphQL subscriptions over WebSocket
- WebRTC data channels for binary data
- Horizontal scaling with Redis pub/sub
- WebSocket compression
- End-to-end encryption