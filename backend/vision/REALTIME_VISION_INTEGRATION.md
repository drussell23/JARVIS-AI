# JARVIS Real-Time Vision Integration

## Overview

The Claude Vision Analyzer has been enhanced with real-time monitoring capabilities and autonomous behaviors, allowing JARVIS to continuously see and understand what's happening on your screen.

## What's New

### 1. **Real-Time Monitoring**
- Continuous screen capture and analysis
- Video streaming support for smooth real-time vision
- Automatic fallback to screenshot mode
- Memory-safe operation on 16GB systems

### 2. **Autonomous Behaviors**
- Detects patterns: notifications, errors, dialogs, loading states
- Suggests appropriate actions based on screen content
- Handles specific behaviors like message checking, error handling

### 3. **Enhanced Integration**
- `claude_vision_analyzer_main.py` - Now includes all real-time features
- `claude_vision_analyzer.py` - Clean wrapper with JARVIS-specific methods
- Works seamlessly with existing JARVIS components

## How to Use

### Basic Usage

```python
from vision.claude_vision_analyzer import ClaudeVisionAnalyzer

# Initialize with real-time capabilities
jarvis_vision = ClaudeVisionAnalyzer(api_key, enable_realtime=True)

# Start real-time vision
await jarvis_vision.start_jarvis_vision()

# JARVIS can now see everything happening on your screen!
```

### Key Methods

#### 1. See and Respond to Commands
```python
# JARVIS sees the screen and responds based on visual context
response = await jarvis_vision.see_and_respond("What's in that window?")
print(f"JARVIS: {response['response']}")
```

#### 2. Get Real-Time Context
```python
# Get current screen state with behavior insights
context = await jarvis_vision.get_real_time_context()
if context.get('behavior_insights'):
    print(f"Detected: {context['behavior_insights']['detected_patterns']}")
    print(f"Suggestions: {context['behavior_insights']['suggested_actions']}")
```

#### 3. Monitor for Events
```python
# Monitor for notifications
async def on_notification(event):
    print(f"📬 New notification: {event['description']}")

await jarvis_vision.start_jarvis_vision(on_notification)
```

#### 4. Watch for Changes
```python
# Watch screen for 60 seconds and collect changes
changes = await jarvis_vision.watch_for_changes(duration=60.0)
for change in changes:
    print(f"Change detected: {change['description']}")
```

## Integration with JARVIS

### Update Your Command Handler

```python
class JARVISAssistant:
    def __init__(self):
        self.vision = ClaudeVisionAnalyzer(api_key)
        
    async def handle_command(self, command: str):
        # JARVIS sees the screen before responding
        visual_response = await self.vision.see_and_respond(command)
        
        if visual_response['success']:
            # Use visual context to enhance response
            context = visual_response['visual_context']
            analysis = visual_response['command_analysis']
            
            # Take action based on what's visible
            if 'click' in command.lower():
                # Find clickable elements in the analysis
                pass
            elif 'read' in command.lower():
                # Extract text from the screen
                pass
                
            return visual_response['response']
```

### Enable Continuous Monitoring

```python
# Start JARVIS with eyes always open
async def start_jarvis_with_vision():
    jarvis = JARVISAssistant()
    
    # Callback for vision events
    async def on_screen_event(event):
        # React to screen changes
        if 'notification' in event.get('insights', {}).get('detected_patterns', []):
            await jarvis.handle_notification(event)
    
    # Start real-time vision
    await jarvis.vision.start_jarvis_vision(on_screen_event)
    
    # JARVIS is now continuously aware of the screen
```

## Configuration

### Environment Variables
```bash
# Enable/disable components
export VISION_CONTINUOUS_ENABLED=true
export VISION_VIDEO_STREAMING_ENABLED=true
export VISION_SWIFT_ENABLED=true

# Performance tuning
export VISION_MAX_CONCURRENT=10
export VISION_MONITOR_INTERVAL=3.0
export VISION_CACHE_SIZE_MB=100

# Memory safety
export VISION_MEMORY_SAFETY=true
export VISION_PROCESS_LIMIT_MB=2048
```

### Memory Safety

The system automatically:
- Monitors memory usage
- Adjusts quality when memory is low
- Rejects requests if memory critical
- Provides health status

```python
# Check memory health
health = await jarvis_vision.check_memory_health()
if not health['healthy']:
    print(f"Warning: {health['recommendations']}")
```

## Testing

### Basic Test
```bash
python test_vision_basic.py
```

### Real-Time Test
```bash
python test_realtime_vision.py
```

### Integration Test
```bash
python jarvis_vision_example.py
```

## Troubleshooting

### Issue: "Could not import Swift vision"
- **Solution**: This is just a warning. Swift integration is optional.

### Issue: Real-time monitoring fails
- **Solution**: Check memory with `check_memory_health()`
- Ensure screen recording permissions are granted

### Issue: High latency
- **Solution**: Reduce `max_concurrent_requests` or image dimensions
- Enable compression: `compression_enabled=True`

## Performance Tips

1. **Use Video Mode**: For smooth real-time monitoring
   ```python
   await jarvis_vision.switch_to_video_mode()
   ```

2. **Cache Results**: Automatic caching reduces API calls

3. **Batch Analysis**: Analyze multiple things in one call

4. **Memory Management**: Monitor and clear cache regularly

## Example: Complete Integration

```python
import asyncio
from vision.claude_vision_analyzer import ClaudeVisionAnalyzer

class VisualJARVIS:
    def __init__(self, api_key):
        self.vision = ClaudeVisionAnalyzer(api_key)
        self.is_watching = False
        
    async def start(self):
        """Start JARVIS with visual awareness"""
        # Start real-time monitoring
        result = await self.vision.start_jarvis_vision(self.on_screen_change)
        
        if result['success']:
            self.is_watching = True
            print("👁️ JARVIS vision activated!")
            
    async def on_screen_change(self, event):
        """React to screen changes"""
        insights = event.get('insights', {})
        
        for pattern in insights.get('detected_patterns', []):
            if pattern == 'notification':
                print("📬 I see a new notification!")
            elif pattern == 'error':
                print("⚠️ I notice an error on screen")
            elif pattern == 'dialog':
                print("💬 There's a dialog box")
                
    async def process_command(self, command: str):
        """Process commands with visual context"""
        response = await self.vision.see_and_respond(command)
        return response['response']
        
    async def stop(self):
        """Stop visual monitoring"""
        await self.vision.stop_jarvis_vision()
        await self.vision.cleanup_all_components()

# Usage
async def main():
    jarvis = VisualJARVIS(os.getenv('ANTHROPIC_API_KEY'))
    await jarvis.start()
    
    # JARVIS can now see!
    response = await jarvis.process_command("What do you see on my screen?")
    print(f"JARVIS: {response}")
    
    await jarvis.stop()

asyncio.run(main())
```

## Summary

With these enhancements, JARVIS now has:
- ✅ Real-time vision capabilities
- ✅ Autonomous behavior detection
- ✅ Memory-safe operation
- ✅ Seamless integration with existing code
- ✅ Visual context for all commands

Your JARVIS can now truly see and understand what's happening on your screen in real-time!