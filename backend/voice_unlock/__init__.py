"""
JARVIS Voice Unlock Module
========================

Voice-based biometric authentication system for macOS unlocking.
Provides hands-free, secure access to Mac devices using voice recognition.

Features:
- Voice enrollment and profile management
- Real-time voice authentication
- Anti-spoofing protection
- System integration (screensaver, PAM)
- Multi-user support
- Apple Watch proximity detection
- ML optimization for 16GB RAM systems
"""

__version__ = "0.2.0"
__author__ = "JARVIS Team"

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Lazy imports to reduce startup time
_voice_unlock_system = None
_ml_system = None


def get_voice_unlock_system():
    """Get or create the voice unlock system instance"""
    global _voice_unlock_system
    
    if _voice_unlock_system is None:
        try:
            from .voice_unlock_integration import VoiceUnlockSystem
            _voice_unlock_system = VoiceUnlockSystem()
            logger.info("Voice Unlock System initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Voice Unlock System: {e}")
            # Fallback to old system if available
            try:
                from .services.mac_unlock_service import MacUnlockService
                return MacUnlockService()
            except:
                pass
            return None
            
    return _voice_unlock_system


def get_ml_system():
    """Get or create the ML system instance"""
    global _ml_system
    
    if _ml_system is None:
        try:
            from .ml import VoiceUnlockMLSystem
            _ml_system = VoiceUnlockMLSystem()
            logger.info("ML System initialized with optimization")
        except Exception as e:
            logger.error(f"Failed to initialize ML System: {e}")
            return None
            
    return _ml_system


async def initialize_voice_unlock():
    """Initialize the voice unlock system asynchronously"""
    try:
        system = get_voice_unlock_system()
        if system and hasattr(system, 'start'):
            await system.start()
            logger.info("Voice Unlock System started successfully")
            return system
    except Exception as e:
        logger.error(f"Failed to start Voice Unlock System: {e}")
        return None


async def cleanup_voice_unlock():
    """Cleanup voice unlock system resources"""
    global _voice_unlock_system, _ml_system
    
    try:
        if _voice_unlock_system and hasattr(_voice_unlock_system, 'stop'):
            await _voice_unlock_system.stop()
            _voice_unlock_system = None
            
        if _ml_system and hasattr(_ml_system, 'cleanup'):
            _ml_system.cleanup()
            _ml_system = None
            
        logger.info("Voice Unlock System cleaned up")
    except Exception as e:
        logger.error(f"Error during Voice Unlock cleanup: {e}")


# Status helpers
def get_voice_unlock_status() -> Dict[str, Any]:
    """Get current voice unlock status"""
    try:
        system = get_voice_unlock_system()
        if system and hasattr(system, 'get_status'):
            return system.get_status()
        else:
            return {
                'available': False,
                'error': 'Voice Unlock System not initialized'
            }
    except Exception as e:
        logger.error(f"Failed to get Voice Unlock status: {e}")
        return {
            'available': False,
            'error': str(e)
        }


# Check if all dependencies are available
def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available"""
    dependencies = {
        'numpy': False,
        'scipy': False,
        'scikit-learn': False,
        'librosa': False,
        'sounddevice': False,
        'bleak': False,
        'speech_recognition': False
    }
    
    for dep in dependencies:
        try:
            if dep == 'scikit-learn':
                __import__('sklearn')
            elif dep == 'speech_recognition':
                __import__('speech_recognition')
            else:
                __import__(dep.replace('-', '_'))
            dependencies[dep] = True
        except ImportError:
            pass
            
    return dependencies


# Legacy imports for backward compatibility
try:
    from .core.enrollment import VoiceEnrollmentManager
    from .core.authentication import VoiceAuthenticator
    from .services.mac_unlock_service import MacUnlockService
    
    __all__ = [
        'VoiceEnrollmentManager',
        'VoiceAuthenticator', 
        'MacUnlockService',
        'get_voice_unlock_system',
        'get_ml_system',
        'initialize_voice_unlock',
        'cleanup_voice_unlock',
        'get_voice_unlock_status',
        'check_dependencies'
    ]
except ImportError:
    # New system only
    __all__ = [
        'get_voice_unlock_system',
        'get_ml_system',
        'initialize_voice_unlock',
        'cleanup_voice_unlock',
        'get_voice_unlock_status',
        'check_dependencies'
    ]