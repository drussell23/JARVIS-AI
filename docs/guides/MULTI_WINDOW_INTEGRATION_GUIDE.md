# 🖥️ JARVIS Multi-Window Integration Guide

## Overview

JARVIS now has **Multi-Window Awareness** - the ability to see and understand your entire workspace, not just a single window. This guide explains how to use the new Phase 1 features.

## 🚀 Quick Start

### 1. Start JARVIS with Multi-Window Support

```bash
python start_system.py
```

The system will automatically detect and initialize the workspace intelligence module.

### 2. Try These Commands

#### Basic Workspace Queries
- **"Hey JARVIS, what am I working on?"**
  - Response: Analyzes your focused window and related context
  - Example: "Sir, you're working on start_system.py in the JARVIS-AI-Agent project..."

- **"What windows do I have open?"**
  - Response: Lists all open applications and window count
  - Example: "Sir, you have 46 windows open across 16 applications..."

- **"Describe my workspace"**
  - Response: Provides overall workspace context
  - Example: "Sir, your workspace shows active development on JARVIS-AI-Agent..."

#### Specific Queries
- **"Do I have any messages?"**
  - Checks for Discord, Slack, Messages, Mail windows
  - Even if they're not in focus

- **"Are there any errors?"**
  - Scans terminals and development windows
  - Identifies error patterns across windows

## 📊 How It Works

### Window Detection
- Detects all open windows (typically 40-50+)
- Tracks focused window in real-time
- Filters out system UI elements

### Multi-Window Capture
- Captures up to 5 windows simultaneously
- Focused window: Full resolution (100%)
- Background windows: Half resolution (50%)
- Total capture time: <1 second

### Intelligent Analysis
- Uses Claude Vision API when available
- Understands window relationships
- Provides context-aware responses

## 🔧 Technical Details

### Architecture
```
Voice Command → JARVIS Agent → Workspace Intelligence
                                      ↓
                              Window Detection
                                      ↓
                              Multi-Window Capture
                                      ↓
                              Claude Vision Analysis
                                      ↓
                              Context-Aware Response
```

### Performance
- Window detection: <100ms
- Multi-window capture: 0.3-0.7 seconds
- Total response time: 1-3 seconds
- API cost: <$0.05 per query

## 🎯 Use Cases

### Development Workflow
```
You: "What am I working on?"
JARVIS: "Sir, you're working on start_system.py in the JARVIS-AI-Agent project. 
         I can see you have the Problems panel open showing several issues to resolve, 
         and Chrome windows with relevant documentation."
```

### Communication Check
```
You: "Any messages?"
JARVIS: "I don't see Discord or Slack open, but you have WhatsApp in the background. 
         Would you like me to check for other communication apps?"
```

### Error Detection
```
You: "Any errors I should look at?"
JARVIS: "Sir, I can see error indicators in your Terminal window. 
         The Problems panel in Cursor shows 5 issues that need attention."
```

## 🛠️ Troubleshooting

### If JARVIS says "I'm having trouble analyzing your workspace"
1. Check that screen recording permission is granted
2. Verify Claude API key is set in `backend/.env`
3. Try reducing the number of open windows

### If window detection seems slow
- Close unnecessary system windows
- Restart JARVIS to refresh the window cache

### If capture quality is poor
- Ensure your display resolution is standard (not scaled)
- Check that windows aren't minimized

## 🚀 Coming in Phase 2

- **Window Relationship Detection**: Understanding how windows work together
- **Smart Query Routing**: Automatically checking relevant windows
- **Workflow Learning**: Recognizing your work patterns
- **Proactive Insights**: Alerting you to important changes

## 📝 API Reference

### Available Commands
```python
# In your code, you can use:
from backend.vision.jarvis_workspace_integration import JARVISWorkspaceIntelligence

workspace = JARVISWorkspaceIntelligence()
response = await workspace.handle_workspace_command("What am I working on?")
```

### Window Info Structure
```python
@dataclass
class WindowInfo:
    window_id: int
    app_name: str
    window_title: str
    is_focused: bool
    bounds: Dict[str, int]  # x, y, width, height
    layer: int
    is_visible: bool
    process_id: int
```

## 🎉 Summary

JARVIS can now:
- See all your open windows
- Understand what you're working on
- Provide context-aware assistance
- Track your workspace in real-time

This makes JARVIS the first AI assistant with true **Workspace Intelligence**!