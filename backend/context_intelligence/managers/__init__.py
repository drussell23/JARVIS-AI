"""
Context Intelligence Managers
==============================

Managers for system state and resource management
"""

from .space_state_manager import (
    SpaceStateManager,
    SpaceValidator,
    SpaceTransitionHandler,
    SpaceState,
    WindowState,
    WindowInfo,
    SpaceStateInfo,
    EdgeCaseResult,
    get_space_state_manager,
    initialize_space_state_manager
)

__all__ = [
    'SpaceStateManager',
    'SpaceValidator',
    'SpaceTransitionHandler',
    'SpaceState',
    'WindowState',
    'WindowInfo',
    'SpaceStateInfo',
    'EdgeCaseResult',
    'get_space_state_manager',
    'initialize_space_state_manager',
]
