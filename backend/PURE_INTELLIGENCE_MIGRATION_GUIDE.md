# Pure Intelligence Migration Guide

This guide shows how to migrate from the template-based system to pure Claude Vision intelligence.

## Step 1: Update WebSocket Handler

### Old WebSocket Handler (jarvis_voice_api.py)
```python
# ❌ OLD - Complex routing with templates
async def jarvis_stream(self, websocket: WebSocket):
    # ... lots of routing logic ...
    
    # Use unified command processor
    try:
        from .unified_command_processor import get_unified_processor
        processor = get_unified_processor(self.api_key)
        result = await processor.process_command(command_text, websocket)
        
    except Exception as e:
        # Fall back to original logic if unified processor fails
        
    # LEGACY ROUTING (kept as fallback)
    vision_keywords = ['see', 'screen', 'monitor', 'vision']
    is_vision_command = any(word in command_text.lower() for word in vision_keywords)
```

### New WebSocket Handler
```python
# ✅ NEW - Pure intelligence, no routing needed
async def jarvis_stream(self, websocket: WebSocket):
    # ... initialization ...
    
    # Use pure unified processor
    from .unified_command_processor_pure import get_pure_unified_processor
    processor = get_pure_unified_processor(self.api_key)
    
    # Process command with pure intelligence
    result = await processor.process_command(command_text, websocket)
    
    # Send natural response
    await websocket.send_json({
        "type": "response",
        "text": result.get('response'),
        "pure_intelligence": True,
        "timestamp": datetime.now().isoformat()
    })
```

## Step 2: Update Vision Command Handler

### Old Import
```python
from .vision_command_handler import vision_command_handler
```

### New Import
```python
from .vision_command_handler_refactored import vision_command_handler
```

## Step 3: Remove ALL Hardcoded Responses

### Find and Remove These Patterns:

```python
# ❌ REMOVE ALL OF THESE:

# Template strings
response = "I can see your screen..."
response = "Screen monitoring activated..."
response = "Your battery is at..."

# Response dictionaries
RESPONSES = {
    'success': "Command executed successfully",
    'error': "An error occurred"
}

# Formatted strings with placeholders
return f"Your {item} is {status}"

# Conditional templates
if error:
    response = "I encountered an error"
else:
    response = "Operation successful"
```

### Replace With:

```python
# ✅ PURE CLAUDE INTELLIGENCE:
response = await intelligence.understand_and_respond(screenshot, query)
```

## Step 4: Update Error Handling

### Old Error Handling
```python
# ❌ OLD
except Exception as e:
    return {
        "response": f"Error: {str(e)}",
        "error": True
    }
```

### New Error Handling
```python
# ✅ NEW - Even errors are natural
except Exception as e:
    error_response = await vision_handler._get_error_response(
        error_type="processing_error",
        command=query,
        details=str(e)
    )
    return error_response
```

## Step 5: Configuration Updates

### Add to .env
```bash
# Pure Intelligence Settings
ENABLE_PURE_INTELLIGENCE=true
CLAUDE_VISION_API_KEY=your_api_key_here
TEMPORAL_MEMORY_SIZE=50
CONVERSATION_HISTORY_SIZE=20
ENABLE_PROACTIVE_INTELLIGENCE=true
PROACTIVE_CHECK_INTERVAL=3
```

### Update VisionConfig
```python
@dataclass
class VisionConfig:
    # Add pure intelligence settings
    enable_pure_intelligence: bool = True
    temporal_memory_enabled: bool = True
    proactive_enabled: bool = True
    emotional_intelligence: bool = True
    workflow_understanding: bool = True
```

## Step 6: Testing the Migration

### Test Script
```python
"""Test pure intelligence system"""
import asyncio
from api.vision_command_handler_refactored import vision_command_handler

async def test_pure_intelligence():
    queries = [
        "What's my battery level?",
        "Can you see my terminal?", 
        "What am I working on?",
        "Any errors?",
        "How's my system doing?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = await vision_command_handler.handle_command(query)
        print(f"Response: {result['response']}")
        print(f"Pure Intelligence: {result.get('pure_intelligence', False)}")

asyncio.run(test_pure_intelligence())
```

## Step 7: Gradual Migration Strategy

### Phase 1: Parallel Running (Week 1)
```python
# Run both systems in parallel
try:
    # Try pure intelligence first
    pure_result = await pure_processor.process_command(command)
    use_pure = True
except:
    # Fall back to old system
    old_result = await old_processor.process_command(command)
    use_pure = False
```

### Phase 2: Pure Intelligence Primary (Week 2)
```python
# Pure intelligence is primary, old is fallback only
result = await pure_processor.process_command(command)
```

### Phase 3: Complete Migration (Week 3)
```python
# Remove all old code
# Only pure intelligence remains
```

## Common Migration Issues

### Issue 1: Missing API Key
```python
# Solution: Ensure ANTHROPIC_API_KEY is set
if not os.getenv('ANTHROPIC_API_KEY'):
    raise ValueError("ANTHROPIC_API_KEY required for pure intelligence")
```

### Issue 2: Response Format Changes
```python
# Old format expected specific fields
# New format is more flexible
# Add compatibility layer if needed:
if not result.get('response'):
    result['response'] = result.get('text', '')
```

### Issue 3: WebSocket Updates
```python
# Update frontend to handle pure intelligence responses
// In JarvisVoice.js
if (data.pure_intelligence) {
    // Handle natural responses without parsing
    displayResponse(data.text);
}
```

## Validation Checklist

- [ ] All template strings removed
- [ ] All hardcoded responses eliminated
- [ ] Vision handler uses pure intelligence
- [ ] Error responses are natural
- [ ] Proactive monitoring implemented
- [ ] Temporal intelligence working
- [ ] Emotional variations present
- [ ] Workflow understanding active
- [ ] WebSocket updated
- [ ] Frontend compatible

## Benefits After Migration

1. **Every response is unique** - No more repetitive answers
2. **Natural conversation** - Feels like talking to a real assistant
3. **Temporal awareness** - Knows what changed and when
4. **Proactive help** - Notices and communicates important changes
5. **Workflow understanding** - Comprehends what you're trying to do
6. **Zero maintenance** - No templates to update
7. **Automatic improvement** - Gets better as Claude improves

## Final Verification

Run this to verify pure intelligence:
```bash
# Check for any remaining templates
grep -r "I can see\|It appears\|Your screen shows" backend/api/

# Should return: No results (all templates removed)
```

The migration transforms JARVIS from a scripted bot to a truly intelligent assistant.