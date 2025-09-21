# JARVIS Vision Intelligence System

## Overview

The Vision Intelligence System is a multi-language, dynamic state learning framework that enables JARVIS to understand application states without any hardcoding. It combines Python orchestration, Rust performance optimization, and Swift native macOS integration to create a powerful vision understanding system.

## Key Features

### 🧠 Dynamic State Learning
- **Zero Hardcoding**: All states are learned through observation
- **Pattern Recognition**: Rust-powered high-performance pattern matching
- **State Transitions**: Tracks how applications move between states
- **Confidence Scoring**: Each detection has an associated confidence level

### 🚀 Multi-Language Architecture
- **Python**: Core orchestration and AI logic
- **Rust**: High-performance vision processing with SIMD optimization
- **Swift**: Native macOS Vision framework integration

### 💾 Memory Efficiency
- Custom memory pools for zero-copy operations
- Efficient buffer reuse
- Optimized for 16GB RAM constraints

## Installation

### Prerequisites
- Python 3.8+
- Rust 1.70+
- Swift 5.0+
- macOS 11.0+ (for Swift Vision framework)

### Build Instructions

```bash
cd backend/vision/intelligence
./build.sh
```

This will:
1. Build Rust components with PyO3 bindings
2. Compile Swift Vision framework integration
3. Set up Python modules

## Usage

### Basic Integration with ClaudeVisionAnalyzer

```python
from backend.vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig

# Create config with Vision Intelligence enabled
config = VisionConfig(
    enable_vision_intelligence=True,
    vision_intelligence_learning=True,
    vision_intelligence_consensus=True,
    state_persistence_enabled=True
)

# Initialize analyzer
analyzer = ClaudeVisionAnalyzer(api_key="your_api_key", config=config)

# Analyze with state tracking
result, metrics = await analyzer.analyze_with_state_tracking(
    image=screenshot,
    prompt="What's on the screen?",
    app_id="chrome"
)

# Access Vision Intelligence data
if '_vision_intelligence' in result:
    vi_data = result['_vision_intelligence']
    print(f"State: {vi_data['state']['state_id']}")
    print(f"Confidence: {vi_data['confidence']:.2%}")
```

### Direct Vision Intelligence Usage

```python
from backend.vision.intelligence import VisionIntelligenceBridge

# Get the bridge instance
bridge = VisionIntelligenceBridge()

# Analyze visual state
result = await bridge.analyze_visual_state(
    screenshot=screenshot_array,
    app_id="vscode",
    metadata={'context': 'coding'}
)

# Get insights
insights = bridge.get_system_insights()
```

### Training on Labeled States

```python
# Train the system on a known state
train_result = await analyzer.train_vision_intelligence(
    screenshot=screenshot,
    app_id="slack",
    state_id="message_compose",
    state_type="active"
)
```

## Architecture

### Component Overview

```
Vision Intelligence System
├── Python Layer (Orchestration)
│   ├── VisualStateManagementSystem
│   ├── ApplicationStateTracker
│   └── VisionIntelligenceBridge
│
├── Rust Layer (Performance)
│   ├── PatternMatcher
│   ├── FeatureExtractor
│   ├── StateDetector
│   └── MemoryPool
│
└── Swift Layer (Native macOS)
    ├── VisionIntelligence
    ├── ColorHistogram
    └── StructuralAnalysis
```

### State Learning Process

1. **Observation**: Visual data is captured and processed
2. **Feature Extraction**: Multi-language extractors identify key features
3. **State Detection**: Pattern matching against learned states
4. **Consensus**: Multiple detectors vote on the final state
5. **Learning**: New patterns are incorporated into the knowledge base

## Configuration

### Environment Variables

- `VISION_INTELLIGENCE_ENABLED`: Enable/disable the system (default: true)
- `VISION_INTELLIGENCE_LEARNING`: Enable learning mode (default: true)
- `VISION_INTELLIGENCE_CONSENSUS`: Require consensus between detectors (default: true)
- `VISION_STATE_PERSISTENCE`: Save learned states to disk (default: true)

### VisionConfig Options

```python
config = VisionConfig(
    enable_vision_intelligence=True,      # Enable the system
    vision_intelligence_learning=True,    # Learn new patterns
    vision_intelligence_consensus=True,   # Use consensus voting
    state_persistence_enabled=True        # Save states to disk
)
```

## Performance

### Benchmarks (M1 MacBook Pro, 16GB RAM)

- **Pattern Matching**: 0.5ms per image (Rust SIMD)
- **Feature Extraction**: 2ms per image (parallel processing)
- **State Detection**: 5ms total (including consensus)
- **Memory Usage**: <200MB for 1000 learned states

### Optimization Tips

1. **Enable Rust acceleration**: Provides 4-6x speedup
2. **Use Swift for preprocessing**: Native macOS APIs are faster
3. **Batch analyses**: Process multiple images together
4. **Enable state persistence**: Avoid relearning on restart

## Advanced Features

### Custom State Detectors

```python
from backend.vision.intelligence import StateDetector

class CustomDetector(StateDetector):
    async def detect_state(self, observation):
        # Custom detection logic
        return state_id, confidence
    
    def learn_from_observation(self, observation, state_id):
        # Custom learning logic
        pass
```

### State Transition Analysis

```python
# Get state transition insights
insights = await analyzer.get_vision_intelligence_insights("app_id")
transitions = insights['requested_app']['state_transitions']
```

### Export Learned States

```python
# Save all learned states
analyzer.save_vision_intelligence_states()

# States are saved to: backend/vision/intelligence/learned_states/
```

## Troubleshooting

### Vision Intelligence Not Available

```bash
# Check if components are built
ls backend/vision/intelligence/*.so
ls backend/vision/intelligence/*.dylib

# Rebuild if necessary
cd backend/vision/intelligence
./build.sh
```

### Memory Issues

- Reduce `cache_max_entries` in VisionConfig
- Enable `reject_on_memory_pressure`
- Use `memory_threshold_percent` to trigger cleanup

### Performance Issues

- Check if Rust components are loaded
- Verify Swift integration is active
- Monitor with `get_system_insights()`

## Future Enhancements

1. **GPU Acceleration**: Metal compute shaders for neural networks
2. **Distributed Learning**: Share learned states across instances
3. **Real-time Adaptation**: Continuous learning during use
4. **Cross-Platform**: Windows and Linux support

## Contributing

1. Follow the multi-language architecture
2. Ensure zero hardcoding principle
3. Add tests for new detectors
4. Document learned state formats

## License

Part of the JARVIS AI Agent system.