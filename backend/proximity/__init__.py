"""
Proximity-Aware Display Connection System
==========================================

Intelligent spatial awareness layer that bridges Bluetooth proximity detection
with multi-monitor display management for contextual, environmentally-aware
display connection and command routing.

Features:
- Bluetooth LE proximity detection (Apple Watch, iPhone, AirPods)
- RSSI-based distance estimation with Kalman filtering
- Dynamic display location mapping and configuration
- Proximity scoring with adaptive thresholds
- Context-aware display selection
- Async/await architecture throughout
- Zero hardcoding - fully configurable

Author: Derek Russell
Date: 2025-10-14
"""

from .proximity_display_context import (
    ProximityData,
    DisplayLocation,
    ProximityDisplayContext,
    ConnectionDecision,
    ProximityZone
)

from .bluetooth_proximity_service import (
    BluetoothProximityService,
    get_proximity_service
)

from .proximity_display_bridge import (
    ProximityDisplayBridge,
    get_proximity_display_bridge
)

from .auto_connection_manager import (
    AutoConnectionManager,
    DisplayMode,
    ConnectionResult,
    get_auto_connection_manager
)

from .proximity_command_router import (
    ProximityCommandRouter,
    get_proximity_command_router
)

from .voice_prompt_manager import (
    VoicePromptManager,
    PromptState,
    get_voice_prompt_manager
)

from .display_availability_detector import (
    DisplayAvailabilityDetector,
    get_availability_detector
)

__all__ = [
    "ProximityData",
    "DisplayLocation",
    "ProximityDisplayContext",
    "ConnectionDecision",
    "ProximityZone",
    "BluetoothProximityService",
    "get_proximity_service",
    "ProximityDisplayBridge",
    "get_proximity_display_bridge",
    "AutoConnectionManager",
    "DisplayMode",
    "ConnectionResult",
    "get_auto_connection_manager",
    "ProximityCommandRouter",
    "get_proximity_command_router",
    "VoicePromptManager",
    "PromptState",
    "get_voice_prompt_manager",
    "DisplayAvailabilityDetector",
    "get_availability_detector"
]

__version__ = "1.0.0"
