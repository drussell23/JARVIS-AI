"""
Situational Awareness Intelligence (SAI) Module
================================================

Production-grade environmental awareness system for JARVIS.

Enables real-time perception and adaptation to dynamic UI changes.
"""

from .core_engine import (
    SituationalAwarenessEngine,
    UIElementMonitor,
    SystemUIElementTracker,
    AdaptiveCacheManager,
    EnvironmentHasher,
    MultiDisplayAwareness,
    get_sai_engine,
    # Data models
    UIElementDescriptor,
    UIElementPosition,
    EnvironmentalSnapshot,
    ChangeEvent,
    # Enums
    ElementType,
    ChangeType,
    ConfidenceLevel
)

__all__ = [
    'SituationalAwarenessEngine',
    'UIElementMonitor',
    'SystemUIElementTracker',
    'AdaptiveCacheManager',
    'EnvironmentHasher',
    'MultiDisplayAwareness',
    'get_sai_engine',
    'UIElementDescriptor',
    'UIElementPosition',
    'EnvironmentalSnapshot',
    'ChangeEvent',
    'ElementType',
    'ChangeType',
    'ConfidenceLevel'
]

__version__ = '1.0.0'
