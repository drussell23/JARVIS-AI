"""
JARVIS Wake Word Detection System
================================

Advanced wake word detection for hands-free JARVIS activation.
"""

from .config import get_config, WakeWordConfig
from .core.detector import WakeWordDetector
from .core.audio_processor import AudioProcessor
from .services.wake_service import WakeWordService

__all__ = [
    'get_config',
    'WakeWordConfig', 
    'WakeWordDetector',
    'AudioProcessor',
    'WakeWordService'
]

__version__ = '1.0.0'