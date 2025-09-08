# Temporal Context Engine Guide

## Overview

The Temporal Context Engine provides JARVIS with the ability to understand events across time, detecting patterns, predicting future events, and building causal relationships. It maintains context across four temporal layers.

## Architecture

### Temporal Layers

1. **Immediate Context** (Last 30 seconds)
   - Frame-by-frame changes
   - User interactions
   - System responses
   - State transitions

2. **Short-term Context** (Last 5 minutes)
   - Task sequences
   - Error occurrences
   - Navigation patterns
   - Focus changes

3. **Long-term Context** (Hours/days)
   - Workflow patterns
   - Recurring issues
   - Solution history
   - Productivity patterns

4. **Persistent Context** (Permanent)
   - Learned preferences
   - Common workflows
   - Personal patterns
   - Historical solutions

### Core Components

#### 1. Event Stream Processor
- Captures all visual changes
- Timestamps every event
- Classifies event types
- Links related events

#### 2. Pattern Extractor
- **Sequence Patterns**: Repeated event sequences
- **Periodic Patterns**: Events that occur at regular intervals
- **Causality Patterns**: Cause-effect relationships
- **Workflow Patterns**: Complete task sequences

#### 3. Context Builder
- Maintains sliding windows for each temporal layer
- Updates pattern database
- Scores pattern significance
- Prunes irrelevant data

## Event Types

```python
class EventType(Enum):
    # Visual events
    SCREENSHOT_CAPTURED = auto()
    STATE_CHANGE = auto()
    ELEMENT_INTERACTION = auto()
    WINDOW_FOCUS = auto()
    CONTENT_CHANGE = auto()
    
    # User events
    MOUSE_CLICK = auto()
    KEYBOARD_INPUT = auto()
    SCROLL = auto()
    GESTURE = auto()
    
    # System events
    APPLICATION_LAUNCH = auto()
    APPLICATION_CLOSE = auto()
    ERROR_OCCURRED = auto()
    NOTIFICATION = auto()
    
    # Workflow events
    TASK_START = auto()
    TASK_COMPLETE = auto()
    WORKFLOW_STEP = auto()
    CONTEXT_SWITCH = auto()
```

## Integration with VSMS

The Temporal Context Engine is automatically integrated with VSMS Core:

1. **Automatic Event Recording**: State changes are recorded as temporal events
2. **Screenshot Events**: Each visual observation creates a SCREENSHOT_CAPTURED event
3. **State Transitions**: Tracked as STATE_CHANGE events with timing information

## Usage in claude_vision_analyzer_main.py

### Configuration

```python
config = VisionConfig(
    temporal_context_enabled=True,
    temporal_pattern_extraction=True,
    temporal_prediction_enabled=True
)
```

### Access Temporal Context

```python
# Get temporal context
context = await analyzer.get_temporal_context(app_id="chrome")

# Get predictions
predictions = await analyzer.get_temporal_predictions(lookahead_seconds=60)
```

### Temporal Data in Analysis Results

When analyzing screenshots, temporal context is automatically included:

```python
result = await analyzer.analyze_screenshot(screenshot, prompt)

if 'temporal_context' in result:
    context = result['temporal_context']
    # Access immediate, short-term, long-term, and persistent contexts
    # View active patterns and predictions
```

## Pattern Types Detected

### 1. Sequence Patterns
- Repeated sequences of 3+ events
- Example: Open app → Navigate → Edit → Save

### 2. Periodic Patterns
- Events occurring at regular intervals
- Example: Auto-save every 2 minutes

### 3. Causality Patterns
- Consistent cause-effect relationships
- Example: Click save → Save dialog appears

### 4. Workflow Patterns
- Complete task sequences from start to finish
- Example: Email workflow (Open → Read → Reply → Send)

## Memory Allocation (200MB)

- **Event Buffer**: 100MB
  - Stores recent events in memory
  - Automatic compaction of old events

- **Pattern Storage**: 50MB
  - Detected patterns and their metadata
  - Pattern significance scores

- **Context Index**: 50MB
  - Temporal layer data
  - Prediction models

## Key Features

### 1. Real-time Event Processing
- Events are processed as they occur
- Related events are automatically linked
- Patterns are extracted every 30 seconds

### 2. Intelligent Predictions
- Predicts next likely events based on patterns
- Time-based predictions for periodic events
- Workflow continuation suggestions

### 3. Causality Understanding
- Detects cause-effect relationships
- Builds causality chains
- Helps understand user intent

### 4. Memory-Optimized
- Automatic cleanup of old events
- Compression of historical data
- Prioritizes significant patterns

## Example Use Cases

### 1. Workflow Automation
```python
# Detect repeated workflows
patterns = await analyzer.vsms_core.temporal_engine.get_active_patterns()
workflow_patterns = [p for p in patterns if p['type'] == 'workflow']
```

### 2. Error Prevention
```python
# Predict and prevent errors
context = await analyzer.get_temporal_context()
if any('error' in str(p) for p in context.get('predictions', {}).get('next_likely_events', [])):
    # Warn user about potential error
```

### 3. Productivity Analysis
```python
# Analyze productivity patterns
context = await analyzer.get_temporal_context()
productivity = context['long_term']['productivity_metrics']
peak_hours = productivity.get('peak_hours', [])
```

## Testing

Run the temporal context tests:
```bash
cd backend/vision/intelligence
python test_temporal_context.py
```

## Important Notes

1. **Privacy**: All temporal data is stored locally
2. **Learning**: Patterns improve over time with usage
3. **Performance**: Background tasks run every 30 seconds
4. **Persistence**: Learned patterns are saved between sessions

The Temporal Context Engine transforms JARVIS from understanding "what is happening now" to understanding "what happened before, what's happening now, and what's likely to happen next".