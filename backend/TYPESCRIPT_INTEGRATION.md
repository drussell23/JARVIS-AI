# TypeScript WebSocket Integration for JARVIS

## Overview

This document describes the TypeScript layer added to enhance WebSocket stability and dynamic capabilities while maintaining zero hardcoding principles.

## Architecture

### 1. Dynamic WebSocket Client (`websocket/DynamicWebSocketClient.ts`)
- **Auto-discovery**: Automatically discovers available WebSocket endpoints through multiple methods
- **Smart routing**: Routes messages to the best endpoint based on capabilities
- **Self-healing**: Implements exponential backoff reconnection strategies
- **Type safety**: Provides strong typing for WebSocket messages
- **Learning system**: Learns message types and patterns dynamically

### 2. TypeScript-Python Bridge (`bridges/WebSocketBridge.ts`)
- **Seamless integration**: Bridges TypeScript frontend with Python backend
- **Type conversion**: Automatically converts between Python and TypeScript types
- **Correlation tracking**: Tracks request-response pairs for async operations
- **Error handling**: Provides robust error handling with retry mechanisms

## Features

### Zero Hardcoding
- No hardcoded endpoints - discovers dynamically
- No hardcoded message types - learns from actual messages
- No hardcoded reconnection delays - adapts based on network conditions

### Auto-Discovery Methods
1. **API Discovery**: Queries `/api/websocket/endpoints`
2. **DOM Discovery**: Scans HTML for data-websocket attributes
3. **Network Scan**: Tests common WebSocket paths
4. **Config Discovery**: Reads from configuration files

### Reconnection Strategies
- **Exponential**: Delays double with each attempt (default)
- **Linear**: Fixed increment between attempts
- **Fibonacci**: Uses Fibonacci sequence for delays

### Message Validation
- Learns message schemas automatically
- Validates incoming messages against learned schemas
- Provides type safety without manual configuration

## Usage

### JavaScript (No Compilation Required)
```javascript
// Create client
const client = new DynamicWebSocketClient({
    autoDiscover: true,
    reconnectStrategy: 'exponential'
});

// Connect to best available endpoint
await client.connect();

// Or connect to specific capability
await client.connect('vision');

// Handle messages
client.on('workspace_update', (data) => {
    console.log('Workspace updated:', data);
});

// Send messages
await client.send({
    type: 'request_analysis'
}, 'vision');
```

### TypeScript with Bridge
```typescript
import { WebSocketBridge } from './bridges/WebSocketBridge';

// Create bridge
const bridge = new WebSocketBridge();

// Connect to Python backend
await bridge.connectToPython();

// Call Python functions
const result = await bridge.callPythonFunction(
    'vision.unified_vision_system',
    'process_vision_request',
    ['describe my screen']
);

// Subscribe to Python events
bridge.subscribeToPythonEvent('vision_update', (event) => {
    console.log('Vision update:', event);
});
```

## Integration with Existing Code

### Frontend Integration
1. Include the dynamic WebSocket client:
```html
<script src="/static/js/dynamic-websocket.js"></script>
```

2. Create and use the client:
```javascript
const client = new DynamicWebSocketClient();
await client.connect('vision');
```

### Backend Integration
The Python backend automatically supports the TypeScript client through:
- WebSocket discovery API at `/api/websocket/endpoints`
- Type hints in Python messages for automatic conversion
- Correlation ID support for request-response tracking

## Benefits

### Stability
- Automatic reconnection with intelligent backoff
- Connection health monitoring with heartbeats
- Graceful degradation when endpoints fail

### Performance
- Connection pooling and reuse
- Message batching capabilities
- Latency-based endpoint selection

### Developer Experience
- Type safety without manual type definitions
- Auto-completion in TypeScript-aware editors
- Runtime type validation with helpful errors

### Maintainability
- No hardcoded values to update
- Self-documenting through discovery
- Automatic adaptation to backend changes

## Testing

### Basic Test Page
Open `test_websocket.html` for basic WebSocket testing.

### Enhanced Test Page
Open `test_enhanced_websocket.html` for full dynamic client testing with:
- Endpoint discovery visualization
- Real-time statistics
- Connection management
- Message logging

## Compilation (Optional)

TypeScript files can be used directly in modern browsers, but for older browser support:

```bash
# Install dependencies
npm install

# Compile TypeScript
npm run build

# Or compile specific file
npm run compile-websocket
```

## Future Enhancements

1. **ML-based Routing**: Integrate TensorFlow.js for intelligent message routing
2. **Compression**: Add automatic message compression for large payloads
3. **Encryption**: Add end-to-end encryption for sensitive data
4. **Federation**: Support for multi-server WebSocket federation
5. **GraphQL Subscriptions**: Add GraphQL subscription support

## Troubleshooting

### Endpoints Not Discovered
- Check if backend is running
- Verify `/api/websocket/endpoints` is accessible
- Check browser console for errors

### Connection Failures
- Verify WebSocket endpoint URLs
- Check for firewall/proxy issues
- Ensure proper CORS headers

### Type Conversion Issues
- Check Python type hints are correct
- Verify message structure matches expectations
- Enable debug logging in the bridge

## Contributing

When adding new WebSocket endpoints:
1. Register them in `websocket_discovery_api.py`
2. Add capability keywords for routing
3. Document message types in this file

The TypeScript layer enhances the existing system without replacing it, providing a robust foundation for real-time communication.