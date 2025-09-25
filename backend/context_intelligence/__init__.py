"""
JARVIS Context Intelligence System
==================================

Provides context awareness for intelligent command processing
"""

__version__ = "2.0.0"

# Core components - new enhanced versions
from .core.context_manager import get_context_manager
from .core.screen_state import get_screen_state_detector
from .core.command_queue import get_command_queue
from .core.policy_engine import get_policy_engine
from .core.unlock_manager import get_unlock_manager
from .core.feedback_manager import get_feedback_manager

# Keep legacy imports for backward compatibility
from .core.system_state_monitor import get_system_monitor
from .detectors.screen_lock_detector import get_screen_lock_detector
from .handlers.context_aware_handler import get_context_aware_handler

# Integration
from .integrations.jarvis_integration import (
    get_jarvis_integration,
    handle_voice_command,
    handle_queue_status,
    handle_system_status
)

# Wrapper for drop-in replacement
from .integrations.enhanced_context_wrapper import (
    wrap_with_enhanced_context,
    EnhancedContextIntelligenceHandler
)

__all__ = [
    # New core components
    'get_context_manager',
    'get_screen_state_detector', 
    'get_command_queue',
    'get_policy_engine',
    'get_unlock_manager',
    'get_feedback_manager',
    # Legacy components
    'get_system_monitor',
    'get_screen_lock_detector',
    'get_context_aware_handler',
    # Integration
    'get_jarvis_integration',
    'handle_voice_command',
    'handle_queue_status',
    'handle_system_status',
    # Wrapper
    'wrap_with_enhanced_context',
    'EnhancedContextIntelligenceHandler'
]