# Workflow Pattern Engine Guide

## Overview

The Workflow Pattern Engine is a sophisticated component of the JARVIS Vision Intelligence System that learns, optimizes, and automates user work patterns. It uses dynamic pattern discovery with zero hardcoding to understand how users work and suggest improvements.

## Key Features

### 1. Dynamic Pattern Learning
- **Sequence Mining**: Discovers frequent action sequences using FP-Growth algorithm
- **Pattern Clustering**: Groups similar workflows using DBSCAN and hierarchical clustering
- **Temporal Analysis**: Understands time-based patterns and routines
- **Variation Tolerance**: Recognizes patterns even with minor variations

### 2. Pattern Types
- **Daily Routines**: Morning startup, email checking, end-of-day workflows
- **Task Patterns**: Debug cycles, document reviews, code-test-commit sequences
- **Problem-Solving Patterns**: Error research, solution testing, documentation lookup
- **Context Switch Patterns**: How users transition between tasks
- **Adaptive Patterns**: Patterns that change based on context

### 3. Advanced Capabilities
- **Pattern Optimization**: Removes redundancies and suggests parallel execution
- **Workflow Automation**: Generates automation scripts for repetitive patterns
- **Predictive Assistance**: Predicts next actions with confidence scores
- **Performance Tracking**: Monitors pattern execution success rates

## Architecture

### Memory Allocation (120MB Total)

1. **Pattern Database (60MB)**
   - Stores discovered patterns
   - Pattern metadata and statistics
   - Variation tracking

2. **Sequence Buffer (30MB)**
   - Recent event sequences
   - Sliding window of activities
   - Event correlation data

3. **Matching Engine (30MB)**
   - Real-time pattern matching
   - Prediction models
   - Optimization algorithms

### Multi-Language Components

#### Python (Core Engine)
- Pattern formation and management
- Integration with other IUS components
- API and orchestration

#### Rust (High-Performance Mining)
- `pattern_mining.rs`: FP-Growth implementation
- Parallel sequence similarity calculation
- Memory-efficient pattern storage

#### Swift (System Automation)
- `workflow_automation.swift`: Native macOS event capture
- System-level automation execution
- Application launching and control

## Configuration

```bash
# Enable workflow pattern learning
export WORKFLOW_PATTERN_ENABLED=true

# Set minimum pattern support (0.0-1.0)
export WORKFLOW_MIN_SUPPORT=0.2

# Maximum pattern length
export WORKFLOW_MAX_PATTERN_LENGTH=20

# Enable automation suggestions
export WORKFLOW_AUTOMATION_ENABLED=true

# Pattern clustering method (dbscan, hierarchical, hybrid)
export WORKFLOW_CLUSTERING_METHOD=hybrid
```

## API Usage

### Recording Workflow Events

```python
from workflow_pattern_engine import WorkflowEvent, get_workflow_pattern_engine

engine = get_workflow_pattern_engine()

# Record an event
event = WorkflowEvent(
    timestamp=datetime.now(),
    event_type='app_launch',
    source_system='system',
    event_data={'app': 'vscode'}
)
await engine.record_event(event)
```

### Mining Patterns

```python
# Mine patterns with minimum support
patterns = await engine.mine_patterns(min_support=0.2)

for pattern in patterns:
    print(f"Pattern: {' → '.join(pattern.action_sequence)}")
    print(f"Frequency: {pattern.frequency}, Confidence: {pattern.confidence}")
```

### Predicting Next Actions

```python
# Get predictions for current sequence
current_sequence = ['open_file', 'edit_code', 'save_file']
predictions = await engine.predict_next_actions(current_sequence, top_k=5)

for action, probability in predictions:
    print(f"Next action: {action} (probability: {probability:.2f})")
```

### Getting Automation Suggestions

```python
# Get automation suggestions for current context
context = {
    'active_app': 'vscode',
    'recent_actions': ['edit_code', 'test_code'],
    'time_of_day': 'afternoon'
}

suggestions = engine.suggest_automation(context)
for suggestion in suggestions:
    print(f"Automate: {suggestion['description']}")
    print(f"Time saved: {suggestion['estimated_time_saved']}s")
```

## Pattern Examples

### 1. Development Workflow
```
Pattern: open_vscode → open_file → edit_code → save_file → switch_terminal → run_test → switch_vscode → fix_error → save_file → git_commit
Type: TASK_PATTERN
Frequency: 47
Confidence: 0.92
```

### 2. Morning Routine
```
Pattern: system_wake → check_email → check_slack → review_calendar → start_deep_work
Type: ROUTINE_PATTERN
Frequency: 23
Confidence: 0.85
Time Constraint: 8:00 AM - 9:30 AM
```

### 3. Debug Cycle
```
Pattern: error_detected → search_stackoverflow → try_solution → test_code → [repeat]
Type: PROBLEM_SOLVING_PATTERN
Variations: 12
Confidence: 0.78
```

## Advanced Features

### Pattern Optimization

The engine automatically optimizes discovered patterns:
- Removes redundant actions (e.g., duplicate saves)
- Identifies parallelizable actions
- Reorders for efficiency while maintaining dependencies

### Clustering and Similarity

Uses advanced algorithms to group similar patterns:
- DBSCAN for density-based clustering
- Hierarchical clustering for nested patterns
- Hybrid approach combining both methods
- Custom similarity metrics for sequences

### Neural Prediction (Enhanced Engine)

The enhanced engine includes a neural network for improved predictions:
- LSTM with attention mechanism
- Learns from pattern execution history
- Temperature-controlled sampling for variety

## Integration with IUS

### With Activity Recognition
- Activities provide high-level context for patterns
- Pattern completion updates activity progress
- Activities group related patterns

### With Goal Inference
- Patterns provide evidence for goal inference
- Goals influence pattern selection
- Pattern success affects goal confidence

### With VSMS
- State transitions feed pattern detection
- Patterns predict state changes
- State context influences pattern matching

## Performance Optimization

### Memory Management
- Efficient sequence storage with deduplication
- Sliding window for recent events
- Automatic cleanup of old patterns

### Processing Optimization
- Rust-based mining for 10x speed improvement
- Parallel pattern matching
- Incremental learning updates

## Use Cases

### 1. Developer Productivity
- Automates repetitive coding workflows
- Suggests optimal testing sequences
- Predicts next debugging steps

### 2. Communication Management
- Learns email processing patterns
- Automates message categorization
- Suggests response templates

### 3. Research Workflows
- Tracks information gathering patterns
- Suggests relevant resources
- Automates note organization

### 4. Task Switching
- Learns context switch patterns
- Minimizes switching overhead
- Preserves working state

## Troubleshooting

### Low Pattern Detection
- Increase observation time window
- Lower minimum support threshold
- Check event recording is working

### Poor Predictions
- Need more training data
- Increase pattern confidence threshold
- Check for noisy events

### High Memory Usage
- Reduce pattern retention period
- Increase minimum support
- Enable pattern pruning

## Future Enhancements

- Cross-device pattern synchronization
- Collaborative pattern learning
- Natural language pattern queries
- Integration with external automation tools
- Pattern marketplace for sharing workflows