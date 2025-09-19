# Real-Time Vision Fix - No More Generic Responses

## Problem
JARVIS was giving generic, repetitive responses instead of actually analyzing what's on screen:
- "what is the battery percentage?" → Generic screen description
- "can you see my terminal?" → Same generic response
- "what windows are open?" → Same generic response

## Root Causes Fixed

### 1. **Unified Command Processor** - Wrong routing
- Vision queries were all going to `handle_command` instead of `analyze_screen`
- Fixed to route analysis queries properly

### 2. **Generic Prompts** - Not specific to user's question  
- Was using same generic "describe the screen" prompt for everything
- Now uses specific prompts based on query type:
  - Battery → "find the battery percentage"
  - Terminal → "describe what you see in the terminal"
  - Windows → "list all open windows"

### 3. **Response Processing** - Added intelligence
- Battery queries now extract just the percentage
- Removed redundant "Yes, I can see your screen" prefix
- Direct, specific answers to questions

## How It Works Now

```
User: "what is the battery percentage?"
  ↓
Unified Processor → Vision Query (not monitoring command)
  ↓  
Vision Handler → Specific prompt: "find the battery percentage"
  ↓
Claude Vision → Analyzes screen for battery indicator
  ↓
Response: "Your battery is at 51%, Sir."
```

## Key Changes

### unified_command_processor.py
```python
# OLD: All vision commands went to handle_command
result = await handler.handle_command(command_text)

# NEW: Queries go to analyze_screen
if any(word in command_text.lower() for word in ['start', 'stop', 'monitor']):
    result = await handler.handle_command(command_text)
else:
    result = await handler.analyze_screen(command_text)
```

### vision_command_handler.py
```python
# OLD: Generic prompt
"Describe what you see on screen..."

# NEW: Specific prompts
if 'battery' in query:
    prompt = "Find the battery percentage..."
elif 'terminal' in query:
    prompt = "Describe the terminal content..."
# etc.
```

## Result
- **Specific answers** to specific questions
- **Real-time analysis** of what's actually on screen
- **No hardcoded responses** - everything from Claude Vision
- **Natural conversation** - JARVIS personality preserved

Now JARVIS actually sees and understands your screen in real-time!