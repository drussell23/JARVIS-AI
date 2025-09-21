# JARVIS Vision Integrated Intelligence System

## Overview

The JARVIS Vision Integrated Intelligence System combines three powerful components to create a proactive, learning assistant that understands user behavior and provides timely help:

1. **Workflow Pattern Engine** - Learns and optimizes user work patterns
2. **Anomaly Detection Framework** - Identifies unusual situations requiring attention
3. **Intervention Decision Engine** - Determines when and how to offer assistance

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Claude Vision Analyzer                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────┐  ┌─────────────────┐  ┌───────────┐│
│  │ Workflow Pattern   │  │ Anomaly         │  │Intervention││
│  │ Engine            │  │ Detection       │  │ Decision   ││
│  │                   │  │ Framework       │  │ Engine     ││
│  │ • Pattern Mining  │  │ • Visual        │  │ • User     ││
│  │ • Sequence Learn  │  │ • Behavioral    │  │   States   ││
│  │ • Predictions     │  │ • System        │  │ • Timing   ││
│  │ • Automation     │  │ • ML Detection  │  │ • Decisions││
│  └─────────┬─────────┘  └────────┬────────┘  └─────┬─────┘│
│            │                      │                  │      │
│            └──────────────────────┴──────────────────┘      │
│                              ↓                              │
│                    Integrated Analysis                      │
└─────────────────────────────────────────────────────────────┘
```

## Memory Allocation

Total: 270MB distributed across components:

- **Workflow Pattern Engine**: 120MB
  - Pattern database: 60MB
  - Sequence buffer: 30MB
  - Matching engine: 30MB

- **Anomaly Detection**: 70MB
  - Baseline models: 30MB
  - Detection rules: 20MB
  - History: 20MB

- **Intervention Engine**: 80MB
  - Decision models: 30MB
  - Intervention history: 25MB
  - Learning data: 25MB

## Multi-Language Implementation

### Python (Core Logic)
- Main orchestration and AI logic
- Integration with Claude Vision API
- Machine learning models

### Rust (Performance)
- High-speed pattern mining (FP-Growth)
- Real-time anomaly detection
- User state signal processing

### Swift (Native macOS)
- Natural break detection
- Keyboard/mouse monitoring
- Application context tracking

## Integration Points

### 1. Screenshot Analysis Flow

```python
# When analyzing a screenshot:
result = await analyzer.analyze_screenshot(image, prompt)

# Automatically:
# 1. Records workflow events
# 2. Detects anomalies
# 3. Processes intervention signals
# 4. Returns enhanced results with intelligence insights
```

### 2. Anomaly → Intervention Pipeline

When an anomaly is detected with HIGH or CRITICAL severity:

1. Anomaly detector creates observation
2. Converts to user state signal
3. Intervention engine assesses situation
4. Decision made on intervention type and timing

### 3. Workflow → Anomaly Detection

Workflow disruptions trigger anomaly detection:

1. Workflow engine detects pattern break
2. Creates behavioral observation
3. Anomaly detector evaluates
4. May trigger intervention

## Usage Examples

### Basic Integration

```python
from backend.vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer

analyzer = ClaudeVisionAnalyzer()

# The intelligence components are automatically integrated
# Just use the analyzer normally:
result, metrics = await analyzer.analyze_screenshot(
    screenshot, 
    "What is the user doing?"
)

# Result includes intelligence insights:
if 'anomaly_detected' in result:
    print(f"Anomaly: {result['anomaly_detected']['type']}")

if 'intervention_suggested' in result:
    print(f"Help offered: {result['intervention_suggested']['type']}")

if 'workflow_predictions' in result:
    print(f"Next likely action: {result['workflow_predictions'][0]}")
```

### Direct Component Access

```python
# Access workflow engine
workflow_engine = await analyzer.get_workflow_engine()
patterns = await analyzer.get_workflow_patterns()
predictions = await analyzer.predict_workflow(["open_file", "edit"], top_k=3)

# Access anomaly detector  
anomaly_result = await analyzer.detect_anomalies_in_screenshot(screenshot)
history = await analyzer.get_anomaly_history(limit=10)

# Access intervention engine
await analyzer.process_intervention_signal("error_rate", 0.7, 0.9)
intervention = await analyzer.check_intervention_opportunity()
stats = await analyzer.get_intervention_stats()
```

### Testing Scenarios

```python
# Test different user scenarios
result = await analyzer.test_intervention_system("frustrated_user")
# Also: "productive_user", "struggling_user"
```

## Configuration

### Environment Variables

```bash
# Workflow Pattern Engine
export WORKFLOW_PATTERNS_ENABLED=true
export WORKFLOW_MIN_SUPPORT=0.3
export WORKFLOW_NEURAL_PREDICTIONS=true
export WORKFLOW_AUTOMATION_THRESHOLD=0.8

# Anomaly Detection
export ANOMALY_DETECTION_ENABLED=true
export ANOMALY_ML_ENABLED=true
export ANOMALY_BASELINE_SAMPLES=50
export ANOMALY_COOLDOWN_SECONDS=30

# Intervention Engine
export INTERVENTION_ENABLED=true
export INTERVENTION_MIN_CONFIDENCE=0.7
export INTERVENTION_COOLDOWN_MINUTES=5
export INTERVENTION_LEARNING_ENABLED=true
```

## Advanced Features

### 1. Continuous Learning

The system continuously learns from:
- User responses to interventions
- Pattern success rates
- Anomaly detection accuracy

### 2. Timing Optimization

Native macOS integration detects:
- Natural breaks in workflow
- Task boundaries
- Cognitive load levels
- Application switches

### 3. Multi-Modal Detection

Combines multiple signals:
- Visual analysis (screenshots)
- Behavioral patterns (actions)
- System metrics (performance)
- Temporal context (time patterns)

## Performance Considerations

1. **Lazy Loading**: Components only initialize when needed
2. **Async Operations**: Non-blocking analysis pipeline  
3. **Memory Safety**: Automatic cleanup and limits
4. **Caching**: Results cached to reduce redundant processing

## Testing

Run the comprehensive test suite:

```bash
python backend/vision/test_integrated_intelligence.py
```

Run the simple demo:

```bash
python backend/vision/demo_intervention_decisions.py
```

## Future Enhancements

1. **Voice Integration**: Natural language interventions
2. **Predictive Interventions**: Anticipate problems before they occur
3. **Collaborative Learning**: Share patterns across users
4. **Custom Interventions**: User-defined assistance rules

## Troubleshooting

### Components Not Loading

Check environment variables and ensure all dependencies are installed:

```bash
pip install -r requirements.txt
cd backend/vision/jarvis-rust-core && cargo build --release
```

### High Memory Usage

Adjust memory allocations in environment variables or use:

```python
# Reduce memory footprint
os.environ['WORKFLOW_MEMORY_LIMIT_MB'] = '80'
os.environ['ANOMALY_MEMORY_LIMIT_MB'] = '50'
os.environ['INTERVENTION_MEMORY_LIMIT_MB'] = '60'
```

### Intervention Not Triggering

Check cooldown settings and confidence thresholds:

```python
# Get current state
stats = await analyzer.get_intervention_stats()
print(f"Last intervention: {stats.get('last_intervention_time')}")
print(f"Cooldown active: {stats.get('cooldown_active')}")
```

## Conclusion

The Integrated Intelligence System transforms JARVIS from a reactive tool to a proactive assistant that:
- Learns from user behavior
- Detects problems early
- Offers help at the right moment
- Continuously improves

This creates a more natural, helpful interaction that enhances productivity while reducing frustration.