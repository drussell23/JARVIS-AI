# JARVIS Vision Integration Guide

## Overview

This guide explains how to properly integrate the Claude Vision Analyzer with JARVIS to enable screen understanding capabilities.

## The Problem

JARVIS was unable to use vision because:
1. The import path was incorrect (`vision.claude_vision_analyzer` vs `vision.claude_vision_analyzer_main`)
2. The analyzer returns a tuple `(result, metrics)` but JARVIS expected just the result dict
3. The vision analysis was commented out in the autonomous behaviors

## The Solution

Created `claude_vision_analyzer.py` wrapper that:
- Provides the expected import path
- Extracts just the result dict from the tuple
- Adds helpful methods like `get_screen_context()`
- Handles errors gracefully

## How to Use Vision in JARVIS

### 1. Basic Import and Setup
```python
from vision.claude_vision_analyzer import ClaudeVisionAnalyzer

# Initialize with API key
api_key = os.getenv('ANTHROPIC_API_KEY')
analyzer = ClaudeVisionAnalyzer(api_key)
```

### 2. Get Current Screen Context
```python
# Simple method to understand what's on screen
context = await analyzer.get_screen_context()
print(f"Screen shows: {context['description']}")
```

### 3. Analyze Specific Screenshots
```python
# Capture a screenshot (various methods)
import numpy as np
screenshot = np.array(...)  # Your screenshot data

# Analyze with custom prompt
result = await analyzer.analyze_screenshot(
    screenshot,
    "What application is in focus? What actions can I take?"
)

# Access results
description = result['description']
entities = result.get('entities', {})
actions = result.get('actions', [])
```

### 4. Integration with Autonomous Behaviors

Update `autonomous_behaviors.py`:

```python
async def _extract_message_content(self, window: WindowInfo) -> str:
    """Extract message content from window using vision"""
    try:
        if self.vision_analyzer:
            # Get current screen context
            context = await self.vision_analyzer.get_screen_context()
            
            # Or analyze specific window area if available
            # screenshot = capture_window_screenshot(window)
            # result = await self.vision_analyzer.analyze_screenshot(
            #     screenshot, 
            #     "Extract all text content from this window"
            # )
            
            return context.get('description', window.window_title)
        else:
            return window.window_title
    except Exception as e:
        logger.debug(f"Vision analysis failed: {e}")
        return window.window_title
```

## Testing Vision Integration

### Quick Test Script
```python
import asyncio
from vision.claude_vision_analyzer import ClaudeVisionAnalyzer

async def test():
    analyzer = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    
    # Test screen understanding
    result = await analyzer.get_screen_context()
    print(f"I can see: {result['description']}")
    
    # Cleanup
    await analyzer.cleanup_all_components()

asyncio.run(test())
```

### Run Diagnostic
```bash
cd backend/vision
python diagnose_vision_capture.py
```

## Common Issues and Fixes

### 1. Import Error
- **Issue**: `ImportError: No module named 'vision.claude_vision_analyzer'`
- **Fix**: Ensure you're importing from the correct path and the wrapper exists

### 2. Tuple Unpacking Error
- **Issue**: `TypeError: cannot unpack non-iterable NoneType object`
- **Fix**: The wrapper now handles this by extracting just the result dict

### 3. API Key Not Set
- **Issue**: Vision returns errors or empty results
- **Fix**: Set `export ANTHROPIC_API_KEY='your-key'` in your environment

### 4. Memory Issues
- **Issue**: High memory usage or crashes
- **Fix**: The analyzer has built-in memory safety for 16GB systems

## Performance Tips

1. **Cache Results**: The analyzer caches results automatically
2. **Resize Large Images**: Images are automatically resized to max 1536px
3. **Use Specific Prompts**: More specific prompts = better results
4. **Batch Analysis**: Analyze multiple elements in one call when possible

## Example: Making JARVIS See and Respond

```python
# In your JARVIS command handler
async def handle_visual_command(self, command: str):
    """Handle commands that require visual understanding"""
    
    # Get what's on screen
    context = await self.vision_analyzer.get_screen_context()
    
    # Understand the command in context
    if "click" in command.lower():
        # Find clickable elements
        result = await self.vision_analyzer.analyze_screenshot(
            current_screenshot,
            f"User wants to: {command}. What specific UI element should I click?"
        )
        
        # Extract coordinates or element description
        for action in result.get('actions', []):
            if action['type'] == 'click':
                # Perform the click
                await self.perform_click(action['target'])
                
    elif "read" in command.lower():
        # Extract text
        result = await self.vision_analyzer.analyze_screenshot(
            current_screenshot,
            "Extract and list all text content visible on screen"
        )
        
        # Speak or return the text
        return result['description']
```

## Conclusion

With this wrapper and integration guide, JARVIS can now:
- ✅ See and understand what's on screen
- ✅ Extract text and identify UI elements
- ✅ Understand context for commands
- ✅ Provide visual feedback to users

The vision system is fully integrated and ready for use!