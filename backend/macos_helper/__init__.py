"""
JARVIS macOS Helper - Phase 1 Real-Time AI OS Layer

Apple-Compliant macOS Integration for JARVIS AGI OS.

This module provides:
- Real-time macOS event monitoring (NSWorkspace, FSEvents, UserNotifications)
- Permission management and onboarding
- Menu bar status indicator
- LaunchAgent background operation
- Unified event bridge to AGI OS

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    JARVIS macOS Helper Layer                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │              System Event Monitor (NSWorkspace)             │    │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐    │    │
    │  │  │   App   │ │ Window  │ │  Space  │ │   File System   │    │    │
    │  │  │ Events  │ │ Events  │ │ Events  │ │    (FSEvents)   │    │    │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘    │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                              │                                      │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │              Notification Monitor (UserNotifications)       │    │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐    │    │
    │  │  │ System  │ │   App   │ │Calendar │ │    Reminders    │    │    │
    │  │  │ Notifs  │ │ Notifs  │ │ Events  │ │                 │    │    │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘    │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                              │                                      │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │                  Unified Event Bridge                       │    │
    │  │  ┌──────────────────────────────────────────────────────┐   │    │
    │  │  │    Event → AGI OS Coordinator → Decision → Action    │   │    │
    │  │  └──────────────────────────────────────────────────────┘   │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                              │                                      │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │                  Permission Manager                         │    │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐    │    │
    │  │  │Accessib.│ │ Screen  │ │  Mic    │ │  Full Disk      │    │    │
    │  │  │         │ │Recording│ │ Access  │ │    Access       │    │    │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘    │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                              │                                      │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │                  Menu Bar Status Indicator                  │    │
    │  │  ┌──────────────────────────────────────────────────────┐   │    │
    │  │  │    Status │ Pause │ Settings │ Activity │ Quit       │   │    │
    │  │  └──────────────────────────────────────────────────────┘   │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    from macos_helper import get_macos_helper, start_macos_helper

    # Start the helper
    helper = await start_macos_helper()

    # Subscribe to events
    helper.subscribe(MacOSEventType.APP_LAUNCHED, my_handler)

    # Check permissions
    status = await helper.permission_manager.check_all_permissions()

    # Stop when done
    await stop_macos_helper()

Apple Compliance:
    - Uses only public macOS APIs (NSWorkspace, EventKit, UserNotifications)
    - Respects TCC (Transparency, Consent, Control) permissions
    - Shows menu bar indicator when active
    - All user data stays local unless explicitly shared
    - No private API usage

Version: 1.0.0
Author: JARVIS AGI OS Team
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "JARVIS AGI OS Team"

# Core exports
from .event_types import (
    MacOSEventType,
    MacOSEventPriority,
    MacOSEvent,
    AppEvent,
    WindowEvent,
    SpaceEvent,
    NotificationEvent,
    FileSystemEvent,
    PermissionEvent,
)

from .event_bus import (
    MacOSEventBus,
    MacOSEventSubscription,
    get_macos_event_bus,
)

from .permission_manager import (
    PermissionType,
    PermissionStatus,
    PermissionManager,
    get_permission_manager,
)

from .system_event_monitor import (
    SystemEventMonitor,
    get_system_event_monitor,
)

from .notification_monitor import (
    NotificationMonitor,
    NotificationCategory,
    get_notification_monitor,
)

from .macos_helper_coordinator import (
    MacOSHelperCoordinator,
    MacOSHelperState,
    get_macos_helper,
    start_macos_helper,
    stop_macos_helper,
)

from .menu_bar import (
    MenuBarIndicator,
    MenuBarState,
    StatusIcon,
    get_menu_bar,
    start_menu_bar,
    stop_menu_bar,
)

from .agi_integration import (
    AGIBridge,
    AGIBridgeConfig,
    get_agi_bridge,
    start_agi_bridge,
    stop_agi_bridge,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Event Types
    "MacOSEventType",
    "MacOSEventPriority",
    "MacOSEvent",
    "AppEvent",
    "WindowEvent",
    "SpaceEvent",
    "NotificationEvent",
    "FileSystemEvent",
    "PermissionEvent",
    # Event Bus
    "MacOSEventBus",
    "MacOSEventSubscription",
    "get_macos_event_bus",
    # Permission Manager
    "PermissionType",
    "PermissionStatus",
    "PermissionManager",
    "get_permission_manager",
    # System Event Monitor
    "SystemEventMonitor",
    "get_system_event_monitor",
    # Notification Monitor
    "NotificationMonitor",
    "NotificationCategory",
    "get_notification_monitor",
    # Main Coordinator
    "MacOSHelperCoordinator",
    "MacOSHelperState",
    "get_macos_helper",
    "start_macos_helper",
    "stop_macos_helper",
    # Menu Bar
    "MenuBarIndicator",
    "MenuBarState",
    "StatusIcon",
    "get_menu_bar",
    "start_menu_bar",
    "stop_menu_bar",
    # AGI Integration
    "AGIBridge",
    "AGIBridgeConfig",
    "get_agi_bridge",
    "start_agi_bridge",
    "stop_agi_bridge",
]
