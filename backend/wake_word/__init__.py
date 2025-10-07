"""
JARVIS Wake Word Detection System
================================

Advanced wake word detection for hands-free JARVIS activation.
"""

from .config import get_config, WakeWordConfig

# Try to import core components, fallback if not available
try:
    from .core.detector import WakeWordDetector
    from .core.audio_processor import AudioProcessor
    CORE_AVAILABLE = True
except ImportError:
    WakeWordDetector = None
    AudioProcessor = None
    CORE_AVAILABLE = False

try:
    from .services.wake_service import WakeWordService
    SERVICE_AVAILABLE = True
except ImportError:
    WakeWordService = None
    SERVICE_AVAILABLE = False

__all__ = [
    'get_config',
    'WakeWordConfig',
    'WakeWordDetector',
    'AudioProcessor',
    'WakeWordService',
    'CORE_AVAILABLE',
    'SERVICE_AVAILABLE'
]

__version__ = '1.0.0'