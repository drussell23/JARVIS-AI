# Intervention Decision Engine Guide

## Overview

The Intervention Decision Engine is a sophisticated component of the Proactive Intelligence System (PIS) that determines when and how JARVIS should proactively assist users. It implements multi-factor decision making with zero hardcoding, learning from user responses to optimize future interventions.

## Architecture

### Memory Allocation (80MB Total)

1. **Decision Models (30MB)**
   - User state classifiers
   - Intervention predictors
   - Timing optimization models

2. **Intervention History (25MB)**
   - Past intervention records
   - User response tracking
   - Effectiveness metrics

3. **Learning Data (25MB)**
   - Training examples
   - Feature buffers
   - Model updates

### Multi-Language Components

1. **Python (Core Engine)**
   - Decision logic and orchestration
   - Machine learning models
   - API and integration

2. **Rust (High-Performance Detection)**
   - Real-time user state detection
   - Signal processing
   - Timing optimization

3. **Swift (Native macOS)**
   - Natural break detection
   - Keyboard/mouse monitoring
   - Application context tracking

## Decision Framework

### User States

The engine tracks eight distinct user states:
- **Focused**: Deep concentration, high productivity
- **Frustrated**: High error rate, repeated attempts
- **Productive**: Good progress, task completion
- **Struggling**: Seeking help, slow progress
- **Stressed**: Erratic behavior, time pressure
- **Idle**: Low activity, potential break
- **Learning**: Documentation viewing, exploration
- **Confused**: Navigation loops, uncertainty

### Situation Assessment

Four key factors evaluated:
1. **Problem Severity**: How critical is the issue?
2. **Time Criticality**: How urgent is resolution?
3. **Solution Availability**: Can we help effectively?
4. **Success Probability**: Will user accept help?

### Intervention Types

Five levels of intervention:
1. **Silent Monitoring**: Observe without interrupting
2. **Subtle Indication**: Small visual cue
3. **Suggestion Offer**: Non-intrusive popup
4. **Direct Assistance**: Active help panel
5. **Autonomous Action**: Execute with confirmation

### Timing Strategies

Optimal timing selection:
- **Immediate**: For critical issues
- **Natural Break**: During idle time
- **Task Boundary**: After completions
- **Low Cognitive Load**: When user is relaxed
- **User Request Likely**: When help-seeking detected

## Usage Examples

### Basic Integration

```python
from intervention_decision_engine import get_intervention_decision_engine

engine = get_intervention_decision_engine()

# Process user signals
signal = UserStateSignal(
    signal_type="error_rate",
    value=0.7,  # High error rate
    confidence=0.9,
    timestamp=datetime.now(),
    source="error_detector"
)
await engine.process_user_signal(signal)

# Assess situation
situation_data = {
    'has_error': True,
    'error_type': 'blocking',
    'failure_count': 3,
    'known_issue': True,
    'documentation_available': True
}
situation = await engine.assess_situation(situation_data)

# Get intervention decision
opportunity = await engine.decide_intervention()
if opportunity:
    print(f"Intervene with: {opportunity.intervention_type}")
    print(f"Timing: {opportunity.timing_strategy}")
    
    # Execute intervention
    result = await engine.execute_intervention(opportunity)
```

### Signal Processing

```python
# Keyboard signals
await engine.process_user_signal(UserStateSignal(
    signal_type="typing_pattern",
    value=0.8,  # Consistent typing
    confidence=0.9,
    timestamp=datetime.now(),
    source="keyboard_monitor",
    metadata={"wpm": 65, "accuracy": 0.95}
))

# Mouse signals
await engine.process_user_signal(UserStateSignal(
    signal_type="mouse_movement",
    value=0.9,  # Erratic movement
    confidence=0.8,
    timestamp=datetime.now(),
    source="mouse_monitor",
    metadata={"velocity": 2.5, "clicks": 15}
))

# Task signals
await engine.process_user_signal(UserStateSignal(
    signal_type="task_completion",
    value=1.0,
    confidence=0.95,
    timestamp=datetime.now(),
    source="task_tracker"
))
```

### Custom Intervention Content

```python
# Prepare custom content for different intervention types
if opportunity.intervention_type == InterventionType.SUGGESTION_OFFER:
    opportunity.content['suggestion'] = {
        'title': 'Need Help?',
        'message': 'I noticed you might be having trouble. Would you like assistance?',
        'options': ['Yes, help me', 'No thanks', 'Ask me later'],
        'display': {
            'position': 'bottom-right',
            'style': 'minimal',
            'auto_dismiss': 10
        }
    }
```

## Configuration

### Environment Variables

```bash
# User state detection
export INTERVENTION_MIN_CONFIDENCE=0.7
export INTERVENTION_STATE_BUFFER_SIZE=1000

# Timing optimization
export INTERVENTION_NATURAL_BREAK_THRESHOLD=5  # seconds
export INTERVENTION_COGNITIVE_LOAD_WEIGHT=0.7

# Learning system
export INTERVENTION_LEARNING_ENABLED=true
export INTERVENTION_MODEL_UPDATE_FREQUENCY=50  # interventions

# Cooldown settings
export INTERVENTION_COOLDOWN_MINUTES=5
export INTERVENTION_MAX_DAILY=20
```

## Integration with Other Components

### With Anomaly Detection

```python
# When anomaly detected, create intervention opportunity
anomaly = anomaly_detector.detect_anomaly(observation)

if anomaly.severity == AnomalySeverity.HIGH:
    # Create error signal
    signal = UserStateSignal(
        signal_type="error_rate",
        value=0.8,
        confidence=anomaly.confidence,
        timestamp=datetime.now(),
        source="anomaly_detector",
        metadata={"anomaly_type": anomaly.anomaly_type.value}
    )
    await engine.process_user_signal(signal)
```

### With Workflow Patterns

```python
# Detect workflow interruption
if workflow_engine.is_workflow_interrupted():
    signal = UserStateSignal(
        signal_type="repeated_actions",
        value=0.7,
        confidence=0.8,
        timestamp=datetime.now(),
        source="workflow_engine"
    )
    await engine.process_user_signal(signal)
```

### With VSMS Core

```python
# Use state transitions for intervention timing
state_change = vsms_core.get_last_state_change()

if state_change.is_error_state:
    situation_data = {
        'has_error': True,
        'error_context': state_change.context,
        'state_duration': state_change.duration
    }
    await engine.assess_situation(situation_data)
```

## Advanced Features

### Learning System

The engine continuously learns from intervention outcomes:

```python
# After intervention
result = await engine.execute_intervention(opportunity)

# Engine automatically:
# - Records effectiveness
# - Updates models when buffer is full
# - Adjusts future decisions

# Manual model update
await engine.update_models()

# Check learning progress
stats = engine.get_statistics()
print(f"Model version: {stats['model_version']}")
print(f"Average effectiveness: {stats['overall_effectiveness']}")
```

### Timing Optimization (Rust)

```rust
// High-performance timing detection
let mut detector = UserStateDetector::new(1000);

// Add signals
detector.add_signal(UserSignal {
    signal_type: SignalType::TypingPattern,
    value: 0.8,
    confidence: 0.9,
    timestamp: Utc::now(),
    metadata: HashMap::new(),
});

// Get current state
let (state, confidence) = detector.detect_state();

// Calculate timing score
let timing = TimingOptimizer::new();
let score = timing.calculate_timing_score(&signals);
```

### Natural Break Detection (Swift)

```swift
// Native macOS timing detection
let timing = NaturalInterventionTiming()

// Subscribe to timing opportunities
timing.timingOpportunities.sink { opportunity in
    print("Timing opportunity: \(opportunity.reason)")
    print("Score: \(opportunity.score)")
    print("Cognitive load: \(opportunity.cognitiveLoad)")
}

// Get current activity
if let activity = timing.getCurrentActivity() {
    print("User is: \(activity.type)")
}
```

## Best Practices

### 1. Signal Quality

- Provide high-confidence signals when possible
- Include relevant metadata
- Update signals regularly (at least every few seconds)

### 2. Situation Context

- Always provide problem severity and solution availability
- Include deadline information when relevant
- Specify the context type (coding, debugging, etc.)

### 3. Intervention Execution

- Respect user responses
- Don't re-intervene too quickly
- Track effectiveness for learning

### 4. Performance

- Use Rust components for real-time signal processing
- Batch signals when possible
- Keep intervention history pruned

## Troubleshooting

### Low Intervention Acceptance

- Check timing quality scores
- Ensure natural breaks are detected
- Verify cognitive load estimation
- Review intervention content

### Incorrect User State Detection

- Increase signal frequency
- Add more signal types
- Check signal confidence values
- Verify state models are trained

### Poor Timing

- Calibrate natural break threshold
- Check task boundary detection
- Verify application monitoring (Swift)
- Review cognitive load weights

## Metrics and Monitoring

### Key Metrics

```python
stats = engine.get_statistics()

# User state distribution
print("Time in each state:", stats['state_distribution'])

# Intervention effectiveness
print("Success by type:", stats['effectiveness_by_type'])

# Timing quality
print("Average timing score:", stats['avg_timing_score'])

# Response rates
print("Acceptance rate:", stats['acceptance_rate'])
print("Rejection rate:", stats['rejection_rate'])
```

### Performance Monitoring

```python
# Memory usage
memory = engine.get_memory_usage()
print(f"Total memory: {memory['total'] / 1024 / 1024:.1f} MB")

# Processing latency
print(f"Avg decision time: {stats['avg_decision_time_ms']} ms")

# Model performance
print(f"Model accuracy: {stats['model_accuracy']}")
```

## Future Enhancements

1. **Personalization**
   - User-specific intervention preferences
   - Adaptive timing based on individual patterns
   - Customized content generation

2. **Multi-Modal Interventions**
   - Voice-based assistance
   - Gesture recognition
   - Eye tracking integration

3. **Predictive Interventions**
   - Anticipate problems before they occur
   - Preemptive resource preparation
   - Workflow optimization suggestions

4. **Collaborative Learning**
   - Share effectiveness data across users
   - Crowdsourced intervention strategies
   - Community-driven improvements