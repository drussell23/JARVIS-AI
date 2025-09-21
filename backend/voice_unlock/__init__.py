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
"""

__version__ = "0.1.0"
__author__ = "JARVIS Team"

from .core.enrollment import VoiceEnrollmentManager
from .core.authentication import VoiceAuthenticator
from .services.mac_unlock_service import MacUnlockService

__all__ = [
    'VoiceEnrollmentManager',
    'VoiceAuthenticator', 
    'MacUnlockService'
]