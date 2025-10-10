# Dynamic Follow-Up Query Feature Demo

## Overview

This feature enables JARVIS to have **conversational follow-up interactions** where users can ask for detailed explanations after an initial query. All responses are **100% dynamically generated** from actual terminal/app context - **no hardcoded responses**.

## Example Conversation Flow

### Scenario: Terminal Error in Another Space

**User:** "can you see my terminal in the other window?"

**JARVIS:**
```
Yes, I can see Terminal in Space 2.

I notice there's an error in Terminal (Space 2):
  ModuleNotFoundError: No module named 'requests'

Would you like me to explain what's happening in detail?
```

**User:** "explain what's happening in detail"

**JARVIS:**
```
**Terminal (Space 2)**
Working directory: `/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent`

Recent commands:
  • `cd backend`
  • `python app.py`

Last command: `python app.py`

**Error Analysis:**

ModuleNotFoundError: No module named 'requests'

**Suggested Fix:**
1. `pip install requests`
   Purpose: Install missing Python module 'requests'
   Safety: YELLOW
   Impact: Installs Python package 'requests'
```

## Key Features

### 1. Conversational Memory
- Remembers what was discussed in the last 2 minutes
- Understands follow-up context without re-asking
- Automatically times out for fresh context

### 2. Dynamic Explanations (Zero Hardcoding)
All information is extracted from actual context:

**Terminal Apps:**
- Working directory
- Recent command history
- Last command executed
- Full error messages
- Command output
- Fix suggestions (via TerminalCommandIntelligence)

**Browser Apps:**
- Current URL and page title
- Recent search queries
- Research topics
- Active tabs

**IDE/Editor Apps:**
- Project name
- Active file
- Open files list
- Recent edits

### 3. Multi-App Analysis
When multiple apps are discussed, JARVIS:
- Explains each app's context
- Finds cross-space relationships
- Provides unified understanding

### 4. Intelligent Fix Suggestions
Uses existing TerminalCommandIntelligence to:
- Detect error patterns
- Suggest fix commands
- Classify command safety
- Estimate impact

## Supported Follow-Up Phrases

JARVIS recognizes these natural follow-up queries:
- "explain in detail"
- "more detail"
- "tell me more"
- "what's happening"
- "explain what's happening"
- "what is happening"
- "give me details"
- "explain it"
- "explain that"
- "what's going on"
- "what is going on"

## Technical Implementation

### Architecture

```
User: "can you see my terminal?"
    ↓
answer_query() in ContextIntegrationBridge
    ↓
_handle_visibility_query()
    ├─ Checks all spaces for terminal
    ├─ Detects errors/significant content
    └─ Proactively offers to explain
    ↓
_save_conversation_context()
    └─ Saves: apps discussed, space IDs, timestamp

User: "explain in detail"
    ↓
answer_query() detects follow-up keywords
    ↓
_handle_detail_followup()
    ├─ Retrieves _last_context (what was discussed)
    ├─ For each app in context:
    │   ├─ Terminal → _explain_terminal_context()
    │   │   ├─ Extract: command, output, errors, dir
    │   │   └─ TerminalCommandIntelligence.suggest_fix_commands()
    │   ├─ Browser → _explain_browser_context()
    │   └─ IDE → _explain_ide_context()
    └─ _find_cross_space_relationships()
        └─ CrossSpaceIntelligence.find_correlations()
```

### Files Modified

**backend/core/context/context_integration_bridge.py**
- Added conversational state tracking:
  - `_last_query`: Last user query
  - `_last_response`: Last JARVIS response
  - `_last_context`: Apps/spaces discussed
  - `_conversation_timestamp`: When conversation started

- New methods:
  - `_handle_detail_followup()`: Main follow-up handler
  - `_explain_terminal_context()`: Dynamic terminal explanations
  - `_explain_browser_context()`: Dynamic browser explanations
  - `_explain_ide_context()`: Dynamic IDE explanations
  - `_save_conversation_context()`: Save what was discussed
  - `_find_cross_space_relationships()`: Cross-app correlation

- Enhanced `answer_query()`:
  - Detects follow-up keywords
  - Checks conversation timestamp (2-minute window)
  - Routes to appropriate handler

### Integration Points

1. **TerminalCommandIntelligence** (`backend/vision/handlers/terminal_command_intelligence.py`)
   - Analyzes terminal errors
   - Suggests fix commands
   - Provides safety classification

2. **MultiSpaceContextGraph** (`backend/core/context/multi_space_context_graph.py`)
   - Stores rich terminal/browser/IDE context
   - Provides context via `spaces` attribute
   - Tracks activity across all desktop spaces

3. **CrossSpaceIntelligence** (`backend/core/intelligence/cross_space_intelligence.py`)
   - Finds semantic relationships between apps
   - Correlates activity across spaces

## Testing

### Run Tests
```bash
# Test follow-up queries
python backend/tests/test_followup_detail_queries.py

# Expected output:
# ✅ Dynamic follow-up query: PASS
# ✅ Conversation timeout: PASS
```

### Test Coverage
```
test_followup_detail_query():
  ✓ Conversational memory works
  ✓ Follow-up detected correctly
  ✓ Terminal context extracted
  ✓ Error analysis included
  ✓ Fix suggestions provided
  ✓ No hardcoded responses

test_followup_timeout():
  ✓ Conversation expires after 2 minutes
  ✓ Fallback to generic response
```

## Usage Examples

### Example 1: Terminal Error
```python
# User sees terminal error but isn't at their desk
User: "can you see my terminal in the other window?"
JARVIS: "Yes... I notice there's an error... Would you like me to explain?"
User: "yes, what's happening?"
JARVIS: [Full dynamic explanation with fix suggestion]
```

### Example 2: Multi-App Workflow
```python
# User working across multiple spaces
User: "can you see what I'm working on?"
JARVIS: "Yes, I can see 3 windows:
         • Terminal (Space 1)
         • Chrome (Space 2)
         • VS Code (Space 3)"
User: "explain what's happening in detail"
JARVIS: [Explains all 3 apps + finds relationships between them]
```

### Example 3: Browser Research
```python
User: "can you see my browser?"
JARVIS: "Yes, I can see Chrome in Space 2..."
User: "tell me more"
JARVIS: [Explains current page, search queries, research topic]
```

## Benefits

1. **Natural Conversation**: Users don't need to repeat context
2. **Time-Saving**: Get details on-demand, not always upfront
3. **Comprehensive**: Combines multiple data sources dynamically
4. **Actionable**: Includes fix suggestions and impact analysis
5. **Safe**: Respects conversation timeout for fresh context

## Future Enhancements

Potential improvements:
- Extend conversation window based on complexity
- Multi-turn conversations (more than 2 turns)
- Conversation history/threading
- Cross-conversation learning
- Voice tone detection ("urgent" vs "casual")
