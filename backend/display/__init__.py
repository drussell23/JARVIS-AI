"""
Display Module
==============

Advanced display monitoring and management with multi-method detection.

Author: Derek Russell
Date: 2025-10-15
"""

# Import from advanced display monitor (Component #9)
from .advanced_display_monitor import (
    get_display_monitor,
    set_app_display_monitor,
    AdvancedDisplayMonitor,
    DisplayInfo,
    MonitoredDisplay,
    DetectionMethod,
    DisplayType,
    ConnectionMode
)

# Legacy imports for backward compatibility
from .display_monitor_service import (
    DisplayMonitorService,
    DisplayMonitorConfig,
)

__all__ = [
    # Advanced display monitor (primary)
    "get_display_monitor",
    "set_app_display_monitor",
    "AdvancedDisplayMonitor",
    "DisplayInfo",
    "MonitoredDisplay",
    "DetectionMethod",
    "DisplayType",
    "ConnectionMode",
    # Legacy (backward compatibility)
    "DisplayMonitorService",
    "DisplayMonitorConfig",
]

__version__ = "2.0.0"
