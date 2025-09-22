"""
ML Module for Voice Unlock System
=================================

Memory-optimized machine learning components for voice authentication
on 16GB RAM systems.
"""

from .ml_manager import MLModelManager, get_ml_manager
from .optimized_voice_auth import OptimizedVoiceAuthenticator
from .performance_monitor import PerformanceMonitor, get_monitor
from .ml_integration import VoiceUnlockMLSystem

__all__ = [
    'MLModelManager',
    'get_ml_manager',
    'OptimizedVoiceAuthenticator',
    'PerformanceMonitor',
    'get_monitor',
    'VoiceUnlockMLSystem'
]