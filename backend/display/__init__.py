"""
Display Module
==============

Simple display monitoring and management.
No proximity detection needed - just monitors Screen Mirroring menu.

Author: Derek Russell
Date: 2025-10-15
"""

from .display_monitor_service import (
    DisplayMonitorService,
    DisplayMonitorConfig,
    get_display_monitor
)

__all__ = [
    "DisplayMonitorService",
    "DisplayMonitorConfig",
    "get_display_monitor"
]

__version__ = "1.0.0"
