# Anomaly Detection Framework Guide

## Overview

The Anomaly Detection Framework is a sophisticated component of the Proactive Intelligence System (PIS) that identifies unusual situations requiring intervention. It implements multi-layer detection for Visual, Behavioral, and System anomalies with zero hardcoding.

## Architecture

### Memory Allocation (70MB Total)

1. **Baseline Models (30MB)**
   - Statistical baselines for normal behavior
   - Feature distributions and thresholds
   - Confidence scores

2. **Detection Rules (20MB)**
   - Pattern-based rules
   - Threshold rules
   - ML model rules

3. **Anomaly History (20MB)**
   - Recent anomaly records
   - Pattern tracking
   - Resolution history

### Multi-Language Components

1. **Python (Core Framework)**
   - Main orchestration and API
   - Statistical analysis
   - Machine learning integration

2. **Rust (High-Performance Detection)**
   - Real-time pattern detection
   - Statistical computations
   - Memory-efficient processing

3. **Swift (Native macOS)**
   - System-level anomaly detection
   - Window and application monitoring
   - Native event capture

## Anomaly Types

### Visual Anomalies
- **Unexpected Popups**: Modal dialogs, alerts appearing unexpectedly
- **Error Dialogs**: Error messages, warnings, failures
- **Unusual Layouts**: Missing elements, overlapping UI, corruption
- **Missing Elements**: Expected UI components not found
- **Performance Issues**: Visual lag, rendering problems

### Behavioral Anomalies
- **Repeated Failed Attempts**: Multiple failures in sequence
- **Unusual Navigation**: Unexpected user flow patterns
- **Stuck States**: User stuck in same state for extended time
- **Circular Patterns**: Loops in navigation or actions
- **Time Anomalies**: Unusual duration for activities

### System Anomalies
- **Resource Warnings**: High CPU, memory, disk usage
- **Network Issues**: Connectivity problems, high latency
- **Permission Problems**: Access denied, security issues
- **Crash Indicators**: Application crashes, freezes
- **Data Inconsistencies**: Unexpected data patterns

## Detection Strategy

### 1. Baseline Establishment

```python
# Establish baseline from normal observations
framework = get_anomaly_detection_framework()

# Collect normal observations
normal_observations = [
    {
        'category': 'visual',
        'layout': {'elements': [...], 'anomaly_score': 0.2},
        'colors': {'dominant_hue': 200},
        'window': {'windows': ['main'], 'has_modal': False}
    }
    # ... more observations
]

# Establish baseline
baseline = await framework.establish_baseline(normal_observations, 'visual')
```

### 2. Real-time Monitoring

```python
# Monitor for anomalies
observation = {
    'category': 'visual',
    'window': {
        'has_modal': True,
        'modal_unexpected': True,
        'new_windows': [{'type': 'popup', 'expected': False}]
    }
}

anomalies = await framework.monitor_realtime(observation)
for anomaly in anomalies:
    print(f"Detected: {anomaly.description}")
    print(f"Severity: {anomaly.severity.name}")
    print(f"Suggested actions: {anomaly.suggested_actions}")
```

### 3. Anomaly Response

```python
# Respond to anomaly
if anomalies:
    anomaly = anomalies[0]
    
    # Automatic response
    response = await framework.respond_to_anomaly(anomaly, "auto")
    
    # Investigate
    investigation = await framework.respond_to_anomaly(anomaly, "investigate")
    print(f"Related anomalies: {investigation['investigation']['related_anomalies']}")
```

## Configuration

### Environment Variables

```bash
# Enable anomaly detection
export ANOMALY_DETECTION_ENABLED=true

# Set thresholds
export ANOMALY_LOW_THRESHOLD=2.0
export ANOMALY_MEDIUM_THRESHOLD=3.0
export ANOMALY_HIGH_THRESHOLD=4.0
export ANOMALY_CRITICAL_THRESHOLD=5.0

# ML detection
export ANOMALY_ML_ENABLED=true
export ANOMALY_ML_CONTAMINATION=0.1

# History settings
export ANOMALY_HISTORY_SIZE=1000
export ANOMALY_BASELINE_SAMPLES=50
```

## Integration Examples

### With VSMS Core

```python
# Integrate with Visual State Management System
vsms_state = await vsms_core.get_current_state()

# Create observation from VSMS state
observation = {
    'category': 'visual',
    'state': {
        'state_id': vsms_state.state_id,
        'duration_seconds': vsms_state.duration.total_seconds()
    },
    'layout': vsms_state.scene_graph,
    'confidence': vsms_state.confidence
}

# Check for anomalies
anomalies = await framework.monitor_realtime(observation)
```

### With Workflow Pattern Engine

```python
# Detect anomalies in workflow patterns
workflow_sequence = ['action1', 'action2', 'action1', 'action2']

for action in workflow_sequence:
    behavioral_obs = {
        'category': 'behavioral',
        'action_sequence': [action],
        'navigation': {'loop_score': 0.0}
    }
    
    anomalies = await framework.monitor_realtime(behavioral_obs)
    if anomalies:
        # Workflow contains anomalous patterns
        break
```

### With Activity Recognition

```python
# Monitor activity for anomalies
activity = activity_engine.get_current_activity()

if activity.confidence < 0.5:
    # Low confidence might indicate anomaly
    obs = {
        'category': 'behavioral',
        'activity': {
            'type': activity.activity_type,
            'confidence': activity.confidence,
            'duration': activity.duration
        }
    }
    
    anomalies = await framework.monitor_realtime(obs)
```

## Detection Rules

### Creating Custom Rules

```python
from anomaly_detection_framework import DetectionRule, AnomalyType, AnomalySeverity

# Define custom detection condition
def check_custom_anomaly(observation):
    if 'custom_metric' in observation:
        return observation['custom_metric'] > 100
    return False

# Define severity calculator
def calculate_custom_severity(observation):
    value = observation.get('custom_metric', 0)
    if value > 200:
        return AnomalySeverity.CRITICAL
    elif value > 150:
        return AnomalySeverity.HIGH
    return AnomalySeverity.MEDIUM

# Create rule
custom_rule = DetectionRule(
    rule_id="custom_detector",
    rule_type="threshold",
    condition=check_custom_anomaly,
    severity_calculator=calculate_custom_severity,
    anomaly_type=AnomalyType.DATA_INCONSISTENCY
)

# Add to framework
framework.detection_rules.append(custom_rule)
```

## Performance Optimization

### Rust Integration

```rust
// High-performance anomaly detection
let detector = AnomalyDetector::new();

// Establish baseline
detector.establish_baseline("visual", features_list)?;

// Quick check for critical anomalies
if let Some((anomaly_type, severity)) = detector.quick_check("system", &[
    ("cpu_usage", 95.0),
    ("memory_usage", 88.0)
]) {
    // Handle critical anomaly immediately
}
```

### Batch Processing

```python
# Process multiple observations efficiently
observations = [...]  # Large batch

# Batch processing
anomaly_batches = []
for i in range(0, len(observations), 100):
    batch = observations[i:i+100]
    
    # Process batch in parallel
    tasks = [framework.monitor_realtime(obs) for obs in batch]
    results = await asyncio.gather(*tasks)
    anomaly_batches.extend(results)
```

## Best Practices

1. **Baseline Quality**
   - Collect at least 50 samples for reliable baselines
   - Update baselines periodically
   - Separate baselines by context (app, time of day)

2. **Threshold Tuning**
   - Start with default thresholds
   - Adjust based on false positive rate
   - Consider context-specific thresholds

3. **Response Strategy**
   - Critical anomalies: Immediate user notification
   - High severity: Attempt automatic recovery
   - Medium/Low: Monitor and log

4. **Memory Management**
   - Limit anomaly history size
   - Prune old baselines
   - Use Rust for memory-intensive operations

## Troubleshooting

### High False Positive Rate
- Increase baseline sample size
- Adjust detection thresholds
- Check for environmental changes

### Missing Anomalies
- Lower detection thresholds
- Add more specific rules
- Check if baselines are too broad

### Performance Issues
- Enable Rust acceleration
- Reduce history size
- Use batch processing

## Future Enhancements

1. **Advanced ML Models**
   - Deep learning for complex patterns
   - Transfer learning from similar systems
   - Online learning capabilities

2. **Predictive Anomalies**
   - Predict anomalies before they occur
   - Trend analysis
   - Early warning system

3. **Cross-System Correlation**
   - Correlate anomalies across components
   - Root cause analysis
   - Impact prediction

4. **Automated Recovery**
   - Self-healing capabilities
   - Rollback mechanisms
   - Preventive actions