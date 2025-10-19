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

from .window_capture_manager import (
    WindowCaptureManager,
    PermissionChecker,
    ImageProcessor,
    CaptureRetryHandler,
    CaptureStatus,
    CaptureResult,
    WindowBounds,
    get_window_capture_manager,
    initialize_window_capture_manager
)

__all__ = [
    # Space State Management
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

    # Window Capture Management
    'WindowCaptureManager',
    'PermissionChecker',
    'ImageProcessor',
    'CaptureRetryHandler',
    'CaptureStatus',
    'CaptureResult',
    'WindowBounds',
    'get_window_capture_manager',
    'initialize_window_capture_manager',
]
