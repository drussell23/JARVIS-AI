# VSMS Integration Guide

## Visual State Management System (VSMS) - Complete Integration

### Overview

The Visual State Management System (VSMS) has been fully integrated into JARVIS's Claude Vision Analyzer, providing autonomous visual intelligence with zero hardcoding. All states, patterns, and behaviors are learned dynamically through observation.

### Architecture

```
claude_vision_analyzer_main.py
    ├── Vision Intelligence Bridge
    │   ├── Python VSMS
    │   ├── Rust Pattern Matcher
    │   └── Swift Vision Framework
    └── VSMS Core
        ├── Application State Model
        ├── State Detection Pipeline
        └── State Intelligence
```

### Key Components

#### 1. **VSMS Core** (`vsms_core.py`)
- **Application State Model**: Complete 4-layer architecture
  - Identity Layer: Application identification and versioning
  - State Layer: Current operational state tracking
  - Content Layer: What's being worked on
  - Historical Layer: State history and patterns
- **Memory Management**: Optimized for 150MB allocation (50MB per component)
- **Dynamic Learning**: No hardcoded states - everything is learned

#### 2. **State Detection Pipeline** (`state_detection_pipeline.py`)
- **Multi-Strategy Detection**: 
  - Layout-based detection
  - Color signature matching
  - Text density analysis
  - UI element detection
  - Modal state detection
- **Ensemble Voting**: Consensus-based state determination
- **Visual Signatures**: Perceptual hashing for state identification

#### 3. **State Intelligence** (`state_intelligence.py`)
- **Personal Pattern Learning**:
  - Frequently visited states
  - Stuck state detection
  - Error-prone state identification
  - Time-based preferences
  - Workflow sequence detection
- **Predictive Capabilities**: Next state prediction with personalized scoring
- **Productivity Insights**: Usage patterns and improvement suggestions

### Integration Points in claude_vision_analyzer_main.py

#### Configuration
```python
config = VisionConfig(
    vision_intelligence_enabled=True,  # Enable Vision Intelligence
    vsms_core_enabled=True,           # Enable VSMS Core
    # ... other settings
)
```

#### Key Methods Added

1. **`get_vsms_core()`** - Lazy loading of VSMS Core
2. **`get_vsms_insights()`** - Get comprehensive VSMS insights
3. **`get_state_recommendations(state_id)`** - Get intelligent state recommendations
4. **`save_vsms_states()`** - Persist learned states
5. **`create_state_definition(...)`** - Define new states dynamically

#### Enhanced analyze_screenshot()

The `analyze_screenshot()` method now includes:

```python
# Vision Intelligence enhancement
if self._vision_intelligence_config['enabled']:
    vi_result = await vision_intelligence.analyze_visual_state(...)
    parsed_result['vision_intelligence'] = vi_result

# VSMS Core enhancement
if self._vsms_core_config['enabled']:
    vsms_result = await vsms_core.process_visual_observation(...)
    parsed_result['vsms_core'] = vsms_result
```

### Usage Examples

#### Basic Screenshot Analysis with VSMS
```python
analyzer = ClaudeVisionAnalyzer(config)
result = await analyzer.analyze_screenshot(screenshot, "Analyze this Chrome screenshot")

# Access VSMS results
if 'vsms_core' in result:
    state = result['vsms_core']['detected_state']
    confidence = result['vsms_core']['confidence']
    warnings = result['vsms_core'].get('warnings', [])
```

#### Get State Insights
```python
insights = await analyzer.get_vsms_insights()
print(f"Tracked apps: {insights['tracked_applications']}")
print(f"Personalization: {insights['personalization_score']:.1%}")
```

#### Create Custom State Definition
```python
await analyzer.create_state_definition(
    app_id="chrome",
    state_id="custom_state",
    category="active",
    name="Custom State",
    visual_signatures=[{...}]
)
```

### Memory Optimization

The system is optimized for 16GB RAM constraints:
- **50MB** for state definitions
- **50MB** for transition history
- **50MB** for pattern storage
- Automatic cleanup of old/rarely used data
- Memory-mapped files for efficient storage

### Multi-Language Integration

1. **Python**: Core orchestration and ML algorithms
2. **Rust**: High-performance pattern matching and feature extraction
3. **Swift**: Native macOS Vision framework integration

### Dynamic Learning Features

1. **Zero Hardcoding**: All states are learned through observation
2. **Adaptive Thresholds**: Confidence thresholds adjust based on usage
3. **Personal Preferences**: Learns user-specific patterns over time
4. **Workflow Detection**: Identifies common task sequences
5. **Anomaly Detection**: Flags unusual state transitions

### Testing

Run the integration test:
```bash
cd backend/vision/intelligence
python test_vsms_integration.py
```

### Next Steps

The VSMS integration provides the foundation for:
1. Autonomous task detection and assistance
2. Predictive UI interactions
3. Workflow automation suggestions
4. Performance optimization recommendations
5. Cross-application state tracking

### Important Notes

- The system starts with no knowledge and learns everything dynamically
- Personal patterns improve accuracy over time
- All learned data is persisted between sessions
- The more you use it, the smarter it becomes

### Troubleshooting

If VSMS is not working:
1. Check that both `vision_intelligence_enabled` and `vsms_core_enabled` are True
2. Ensure the `learned_states` directory exists
3. Check logs for any initialization errors
4. Verify Rust/Swift components are built (if available)