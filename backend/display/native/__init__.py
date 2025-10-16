"""
Native AirPlay Control Module
==============================

Production-grade native control for AirPlay displays.

Author: Derek Russell
Date: 2025-10-15
"""

from .native_airplay_controller import (
    NativeAirPlayController,
    ConnectionResult,
    ConnectionMethod,
)

__all__ = [
    "NativeAirPlayController",
    "ConnectionResult",
    "ConnectionMethod",
]
