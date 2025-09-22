"""
JARVIS Backend Module
====================

Core backend services for the JARVIS AI Assistant.
"""

import logging

logger = logging.getLogger(__name__)

# Version info
__version__ = "13.4.0"
__author__ = "JARVIS Team"

# Import voice unlock if available
try:
    from .voice_unlock import (
        get_voice_unlock_system,
        initialize_voice_unlock,
        cleanup_voice_unlock,
        get_voice_unlock_status,
        check_dependencies as check_voice_dependencies
    )
    VOICE_UNLOCK_AVAILABLE = True
    logger.info("Voice Unlock module available")
except ImportError as e:
    VOICE_UNLOCK_AVAILABLE = False
    logger.debug(f"Voice Unlock module not available: {e}")

# Import vision if available
try:
    from .vision import lazy_vision_engine
    VISION_AVAILABLE = True
    logger.info("Vision module available")
except ImportError as e:
    VISION_AVAILABLE = False
    logger.debug(f"Vision module not available: {e}")

# Import process cleanup manager
try:
    from .process_cleanup_manager import ProcessCleanupManager
    CLEANUP_AVAILABLE = True
except ImportError as e:
    CLEANUP_AVAILABLE = False
    logger.debug(f"Process cleanup not available: {e}")

# Import resource manager for strict control on 16GB systems
try:
    from .resource_manager import get_resource_manager
    RESOURCE_MANAGER_AVAILABLE = True
    _resource_manager = None  # Lazy initialization
    logger.info("Resource manager available for memory control")
except ImportError as e:
    RESOURCE_MANAGER_AVAILABLE = False
    _resource_manager = None
    logger.debug(f"Resource manager not available: {e}")


def get_backend_status():
    """Get status of all backend modules"""
    status = {
        'version': __version__,
        'modules': {
            'voice_unlock': VOICE_UNLOCK_AVAILABLE,
            'vision': VISION_AVAILABLE,
            'cleanup': CLEANUP_AVAILABLE,
            'resource_manager': RESOURCE_MANAGER_AVAILABLE
        }
    }
    
    if VOICE_UNLOCK_AVAILABLE:
        status['voice_unlock'] = get_voice_unlock_status()
        
    if RESOURCE_MANAGER_AVAILABLE and _resource_manager:
        status['resources'] = _resource_manager.get_status()
        
    return status


async def initialize_backend():
    """Initialize all backend services with 30% memory target (4.8GB on 16GB systems)"""
    global _resource_manager
    logger.info("Initializing JARVIS backend services...")
    logger.info("Memory target: 30% of system RAM - ultra-aggressive optimization enabled")
    
    # Initialize resource manager first (CRITICAL for 30% memory target)
    if RESOURCE_MANAGER_AVAILABLE:
        try:
            _resource_manager = get_resource_manager()
            logger.info("Resource manager initialized - enforcing 30% memory limit")
            
            # Get initial memory status
            status = _resource_manager.get_status()
            logger.info(f"Initial memory: {status['memory_percent']:.1f}% of system RAM")
            
            # Prepare for proximity + voice unlock if we're using it
            if VOICE_UNLOCK_AVAILABLE:
                logger.info("Preparing memory allocation for Proximity + Voice Unlock")
                _resource_manager.request_voice_unlock_resources()
        except Exception as e:
            logger.error(f"Failed to initialize resource manager: {e}")
    
    # Initialize voice unlock if available (with proximity detection)
    if VOICE_UNLOCK_AVAILABLE:
        try:
            await initialize_voice_unlock()
            logger.info("Voice Unlock initialized with proximity detection")
            logger.info("Apple Watch proximity: 3m unlock, 10m auto-lock")
        except Exception as e:
            logger.error(f"Failed to initialize Voice Unlock: {e}")
            
    logger.info("Backend services initialized with memory optimization")


async def cleanup_backend():
    """Cleanup all backend services"""
    logger.info("Cleaning up JARVIS backend services...")
    
    # Cleanup voice unlock if available
    if VOICE_UNLOCK_AVAILABLE:
        try:
            await cleanup_voice_unlock()
        except Exception as e:
            logger.error(f"Error cleaning up Voice Unlock: {e}")
            
    logger.info("Backend services cleaned up")


__all__ = [
    'get_backend_status',
    'initialize_backend',
    'cleanup_backend',
    'VOICE_UNLOCK_AVAILABLE',
    'VISION_AVAILABLE',
    'CLEANUP_AVAILABLE'
]