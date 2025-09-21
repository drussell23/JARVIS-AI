"""
JARVIS Vision Intelligence System
Multi-language vision understanding without hardcoding
"""

from .visual_state_management_system import (
    VisualStateManagementSystem,
    ApplicationStateTracker,
    StateObservation,
    ApplicationState,
    VisualSignature,
    StateType,
    PatternBasedStateDetector
)

from .vision_intelligence_bridge import (
    VisionIntelligenceBridge,
    get_vision_intelligence_bridge,
    analyze_screenshot,
    SwiftBridge
)

__all__ = [
    'VisualStateManagementSystem',
    'ApplicationStateTracker',
    'StateObservation',
    'ApplicationState',
    'VisualSignature',
    'StateType',
    'PatternBasedStateDetector',
    'VisionIntelligenceBridge',
    'get_vision_intelligence_bridge',
    'analyze_screenshot',
    'SwiftBridge'
]

# Try to import Rust components if available
try:
    import vision_intelligence as rust_vi
    __all__.extend(['rust_vi'])
    RUST_AVAILABLE = True
except ImportError:
    rust_vi = None
    RUST_AVAILABLE = False

print("✨ JARVIS Vision Intelligence System initialized")
if RUST_AVAILABLE:
    print("  ✅ Rust acceleration available")
else:
    print("  ⚠️  Rust acceleration not available - run build.sh to enable")