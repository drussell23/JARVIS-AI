# JARVIS Event-Driven Integration Guide

## Quick Start

### 1. Run JARVIS with Event-Driven Architecture

```bash
cd backend
./start_jarvis_event_driven.py
```

### 2. Access the Event Web UI

Open http://localhost:8888 in your browser to see:
- Real-time event stream
- Performance metrics
- System statistics
- Event filtering and debugging

## What's New

### Components with Event Integration

1. **Voice System** (`ml_enhanced_voice_system_v2.py`)
   - Publishes wake word detection events
   - Responds to memory pressure
   - Tracks performance metrics

2. **Vision System** (`intelligent_vision_integration_v2.py`)
   - Publishes screen capture and analysis events
   - Integrates with voice commands automatically
   - Manages cache based on memory

3. **Memory Controller** (`memory_controller_v2.py`)
   - Monitors system memory in real-time
   - Publishes pressure change events
   - Coordinates model loading/unloading

4. **System Control** (`macos_controller_v2.py`)
   - Executes commands from voice events
   - Publishes operation results
   - Implements safety through events

5. **Event Coordinator** (`jarvis_event_coordinator.py`)
   - Central hub for all components
   - Manages workflows across systems
   - Handles error recovery

## How It Works

### Event Flow Example

```
User: "Hey JARVIS, what's on my screen?"

1. Voice System → VOICE_WAKE_WORD_DETECTED event
2. Coordinator → Prepares systems for interaction
3. Voice System → VOICE_COMMAND_RECEIVED event
4. Coordinator → Determines this needs vision
5. Vision System → VISION_SCREEN_CAPTURED event
6. Vision System → VISION_ANALYSIS_COMPLETE event
7. Voice System → VOICE_RESPONSE_GENERATED event
```

### Key Benefits

- **No Direct Dependencies**: Components don't import each other
- **Easy Testing**: Replay events to test workflows
- **Better Debugging**: See all events in Web UI
- **Automatic Coordination**: Events trigger appropriate responses
- **Memory Efficient**: Components respond to pressure events

## Migration Path

### For Existing Code

1. **Replace Direct Calls** with Event Publishing:
   ```python
   # OLD
   vision_result = self.vision_system.analyze()
   
   # NEW
   VisionEvents.analysis_requested(source="my_component", query="analyze screen")
   # Vision system will publish result events
   ```

2. **Subscribe to Events** Instead of Callbacks:
   ```python
   # OLD
   self.vision_system.on_complete = self.handle_vision_result
   
   # NEW
   @subscribe_to(EventTypes.VISION_ANALYSIS_COMPLETE)
   async def handle_vision_result(event: Event):
       results = event.payload["results"]
   ```

3. **Use Event Builder** for Custom Events:
   ```python
   from core.event_types import EventBuilder
   
   builder = EventBuilder()
   builder.publish(
       "mycomponent.custom_event",
       source="my_component",
       payload={"data": "value"}
   )
   ```

## Configuration

### Event Bus Config (`event_bus_config.yaml`)

- Queue sizes per priority
- Processing timeouts
- Persistence settings
- Filter configurations

### Environment Variables

- `ANTHROPIC_API_KEY`: Required for Claude integration
- `JARVIS_USER`: User name (default: "Sir")
- `JARVIS_DEBUG`: Enable debug mode
- `JARVIS_WEB_UI`: Enable web interface

## Monitoring

### Performance Metrics

Access via Web UI or programmatically:

```python
from core.event_metrics import get_metrics_collector

metrics = get_metrics_collector()
report = metrics.get_performance_report(duration_minutes=5)
```

### Event Replay

Debug issues by replaying events:

```python
from core.event_replay import get_event_replayer

replayer = get_event_replayer()
session_id = replayer.create_session(
    start_time=time.time() - 3600,  # Last hour
    event_types=["voice.*", "vision.*"]
)
replayer.start_replay(session_id)
```

## Troubleshooting

### Events Not Being Received

1. Check subscription pattern matches event type
2. Verify event bus is started
3. Check Web UI for event flow
4. Look for errors in logs

### High Memory Usage

1. Monitor memory events in Web UI
2. Check loaded models in memory stats
3. Adjust thresholds in config
4. Enable aggressive cleanup

### Component Not Responding

1. Check component status in coordinator
2. Look for error events
3. Check if component subscriptions are active
4. Try component recovery via coordinator

## Next Steps

1. **Extend Event Types**: Add your custom events in `event_types.py`
2. **Create Workflows**: Build multi-step workflows in coordinator
3. **Add Filters**: Implement custom filters for events
4. **Monitor Performance**: Use metrics to optimize

The event-driven architecture makes JARVIS more modular, testable, and scalable while maintaining efficiency on your 16GB MacBook Pro!