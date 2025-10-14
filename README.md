# JARVIS AI Assistant

An intelligent voice-activated AI assistant with advanced multi-space desktop awareness and Claude Vision integration.

## Features

### 🖥️ Multi-Space Desktop Intelligence

JARVIS provides comprehensive awareness of all your desktop spaces (Mission Control) with detailed analysis:

**Initial Query:**
```
You: "What's happening across my desktop spaces?"

JARVIS: "You're currently working on multiple different tasks across your 7 desktop spaces.

Space breakdown:
Space 1: Empty
Space 2: Google Chrome (GitHub - Pull Request #123)
Space 3: Google Chrome (Documentation)
Space 4 (current): Cursor (vision_command_handler.py — JARVIS-AI-Agent)
Space 5: VS Code (Main Development)
Space 6: Google Chrome (Research)
Space 7: Terminal (Backend Server)

Would you like me to explain any specific part in more detail?"
```

**Follow-Up with Claude Vision:**
```
You: "Yes, tell me more about Space 4"

JARVIS: "Sir, analyzing Space 4 in detail:

DEVELOPMENT ENVIRONMENT:
- Running VS Code with dark theme
- Multiple files open related to JARVIS vision components
- Current file: 'vision_command_handler.py' in JARVIS-AI-Agent project

ACTIVE WORK:
- Three modified files staged for commit:
  1. pure_vision_intelligence.py
  2. vision_command_handler.py (with 9+ changes)
  3. multi_space_intelligence.py

CURRENT EXECUTION:
- Test script running with error messages visible
- Terminal shows Python 3.9.4 64-bit environment

ERROR STATE:
- Critical issue with VISION component detected
- Error trace visible in terminal output
[Detailed analysis continues...]"
```

### 🎯 Key Capabilities

1. **Multi-Space Overview**
   - Detects all desktop spaces via Yabai/Mission Control
   - Lists applications and window titles in each space
   - Identifies current active space
   - Provides contextual workflow analysis

2. **Follow-Up Intelligence**
   - Remembers multi-space context for follow-up queries
   - Detects responses like "yes", "tell me more", "explain"
   - Uses Claude Vision for detailed space analysis
   - Provides specific, actionable information

3. **Window-Level Detail**
   - Captures exact window titles (not just app names)
   - Understands what you're working on based on titles
   - Identifies specific files, documents, or web pages
   - Recognizes workflow patterns

4. **Protected Component Loading**
   - Vision component stays loaded (never unloaded during memory pressure)
   - Ensures multi-space queries always work
   - No degraded responses from missing components

## Technical Implementation

### Architecture

```
User Query → Smart Router → Multi-Space Handler
                ↓
          Yabai Integration (Window Metadata)
                ↓
          Claude Vision (Screenshot Analysis)
                ↓
          Enhanced Response (Context + Vision)
                ↓
          Follow-Up Context Storage
```

### Components

- **Vision Component**: Protected CORE component (never unloaded)
- **Yabai Integration**: Real-time desktop space detection
- **Claude Vision API**: Deep screenshot analysis
- **Smart Router**: Intent classification and routing
- **Context Manager**: Persistent follow-up context

### Configuration

Vision component is configured as CORE priority in `backend/config/components.json`:

```json
{
  "vision": {
    "priority": "CORE",
    "estimated_memory_mb": 300,
    "intent_keywords": ["screen", "see", "look", "desktop", "space", "window"]
  }
}
```

Protected from unloading in `dynamic_component_manager.py`:
- Excluded from idle component unloading
- Excluded from memory pressure cleanup
- Always included in CORE component list at startup

## Usage Examples

### Basic Queries
- "What's happening across my desktop spaces?"
- "What am I working on?"
- "Show me all my workspaces"
- "What's in my other spaces?"

### Follow-Up Queries
- "Yes" (after multi-space overview)
- "Tell me more about Space 3"
- "What about the Chrome window?"
- "Explain Space 5"
- "Show me the terminal"

### Specific Space Analysis
- "Analyze Space 2"
- "What's happening in Space 4?"
- "Tell me about the coding space"

## Requirements

- macOS with Mission Control
- Yabai window manager (recommended)
- Anthropic Claude API key
- Python 3.8+
- FastAPI backend

## Installation

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Start backend
cd backend
python main.py --port 8010

# Start frontend
cd frontend
npm install
npm start
```

## System Status

The system displays component health:

```json
{
  "components": {
    "chatbots": true,
    "vision": true,     // ✅ Protected CORE component
    "memory": true,
    "voice": true
  }
}
```

## Implementation Details

### Follow-Up Detection
Follow-up indicators: `["yes", "sure", "okay", "tell me more", "explain", "what about", "show me", "describe", "analyze"]`

### Context Storage
```python
self._last_multi_space_context = {
    'spaces': spaces,           # All space metadata
    'window_data': window_data, # Window titles and details
    'timestamp': datetime.now() # For context expiry
}
```

### Claude Vision Integration
- Direct API calls for detailed analysis
- Context-aware prompts with space information
- Structured analysis (Environment, Work, Errors)
- Natural language responses

## macOS Compatibility

### Memory Pressure Detection (Fixed: 2025-10-14)

JARVIS now includes macOS-aware memory pressure detection throughout the entire codebase. This was a critical fix that resolved startup issues where the system would incorrectly enter EMERGENCY mode on macOS.

**The Problem:**
- Original logic used Linux-style percentage-based thresholds (>75% = EMERGENCY)
- macOS shows 70-90% RAM usage as NORMAL due to aggressive caching
- System at 81% usage with 3GB available was flagged as EMERGENCY (incorrect)
- This blocked component loading and made the backend non-functional

**The Solution:**
All memory detection now uses **available memory** instead of percentage:

| Memory Pressure | Available Memory | System Behavior |
|----------------|------------------|-----------------|
| LOW | > 4GB | Normal operation, all features enabled |
| MEDIUM | 2-4GB | Healthy operation (typical on macOS) |
| HIGH | 1-2GB | Start optimizing, reduce background tasks |
| CRITICAL | 500MB-1GB | Aggressive cleanup, limit new operations |
| EMERGENCY | < 500MB | Maximum cleanup, block non-essential features |

**Files Updated (9 total):**
1. `backend/core/dynamic_component_manager.py` - Core memory pressure detection
2. `start_system.py` - Startup cleanup triggers
3. `backend/process_cleanup_manager.py` - System recommendations
4. `backend/resource_manager.py` - Emergency handling
5. `backend/smart_startup_manager.py` - Resource monitoring
6. `backend/voice/model_manager.py` - Model loading decisions
7. `backend/voice/resource_monitor.py` - Adaptive management
8. `backend/voice/optimized_voice_system.py` - Wake word detection
9. `backend/voice_unlock/ml/ml_integration.py` - Health checks

**Impact:**
- ✅ Backend starts reliably every time on macOS
- ✅ No false memory alarms at normal usage (70-90%)
- ✅ Components load correctly in MEDIUM pressure mode
- ✅ System only takes action when truly low on memory (<2GB)

**Technical Details:**
```python
# OLD (Linux-style - incorrect for macOS)
if memory.percent > 75:
    return MemoryPressure.EMERGENCY

# NEW (macOS-aware - correct)
available_gb = memory.available / (1024 ** 3)
if available_gb < 0.5:
    return MemoryPressure.EMERGENCY
```

This fix accounts for macOS's memory management where high percentage usage is normal and "available memory" includes cache that can be instantly freed.

## Fixes Applied

1. ✅ Vision component set to CORE priority
2. ✅ Protected from auto-unloading during idle
3. ✅ Protected from memory pressure cleanup
4. ✅ Window titles included in multi-space data
5. ✅ Enhanced Claude prompts for detailed analysis
6. ✅ Follow-up context storage and detection
7. ✅ Space-specific screenshot capture
8. ✅ Comprehensive debug logging
9. ✅ macOS-aware memory detection (system-wide)

## License

MIT License - see LICENSE file for details
