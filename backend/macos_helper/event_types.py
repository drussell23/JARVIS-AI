"""
JARVIS macOS Helper - Event Types

Defines all event types for the macOS helper layer.
These events are emitted by various monitors and consumed by the AGI OS coordinator.

Features:
- Type-safe event definitions
- Priority-based event handling
- Automatic serialization
- Event correlation support
- Rich metadata

Apple Compliance:
- All events represent data from public APIs
- No private system information exposed
- User-controllable event emission
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable, Coroutine, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Event Enums
# =============================================================================

class MacOSEventType(str, Enum):
    """Types of macOS system events."""

    # Application Events
    APP_LAUNCHED = "app_launched"
    APP_TERMINATED = "app_terminated"
    APP_ACTIVATED = "app_activated"
    APP_DEACTIVATED = "app_deactivated"
    APP_HIDDEN = "app_hidden"
    APP_UNHIDDEN = "app_unhidden"

    # Window Events
    WINDOW_CREATED = "window_created"
    WINDOW_CLOSED = "window_closed"
    WINDOW_FOCUSED = "window_focused"
    WINDOW_UNFOCUSED = "window_unfocused"
    WINDOW_MOVED = "window_moved"
    WINDOW_RESIZED = "window_resized"
    WINDOW_MINIMIZED = "window_minimized"
    WINDOW_DEMINIMIZED = "window_deminimized"
    WINDOW_FULLSCREEN_ENTERED = "window_fullscreen_entered"
    WINDOW_FULLSCREEN_EXITED = "window_fullscreen_exited"

    # Space/Desktop Events
    SPACE_CHANGED = "space_changed"
    SPACE_CREATED = "space_created"
    SPACE_DESTROYED = "space_destroyed"
    MISSION_CONTROL_ENTERED = "mission_control_entered"
    MISSION_CONTROL_EXITED = "mission_control_exited"

    # Notification Events
    NOTIFICATION_RECEIVED = "notification_received"
    NOTIFICATION_CLICKED = "notification_clicked"
    NOTIFICATION_DISMISSED = "notification_dismissed"
    NOTIFICATION_ACTION_TAKEN = "notification_action_taken"

    # Calendar/Reminder Events
    CALENDAR_EVENT_UPCOMING = "calendar_event_upcoming"
    CALENDAR_EVENT_STARTED = "calendar_event_started"
    CALENDAR_EVENT_ENDED = "calendar_event_ended"
    REMINDER_DUE = "reminder_due"

    # File System Events
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_RENAMED = "file_renamed"
    FILE_MOVED = "file_moved"
    DIRECTORY_CREATED = "directory_created"
    DIRECTORY_DELETED = "directory_deleted"

    # System Events
    SCREEN_LOCKED = "screen_locked"
    SCREEN_UNLOCKED = "screen_unlocked"
    SYSTEM_SLEEP = "system_sleep"
    SYSTEM_WAKE = "system_wake"
    DISPLAY_CHANGED = "display_changed"
    VOLUME_CHANGED = "volume_changed"
    NETWORK_CHANGED = "network_changed"
    POWER_STATUS_CHANGED = "power_status_changed"

    # Permission Events
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    PERMISSION_REQUESTED = "permission_requested"

    # User Activity Events
    USER_IDLE_STARTED = "user_idle_started"
    USER_IDLE_ENDED = "user_idle_ended"
    USER_PRESENCE_DETECTED = "user_presence_detected"

    # Helper System Events
    HELPER_STARTED = "helper_started"
    HELPER_STOPPED = "helper_stopped"
    HELPER_ERROR = "helper_error"
    HELPER_HEALTH_CHECK = "helper_health_check"


class MacOSEventPriority(Enum):
    """Priority levels for macOS events."""
    DEBUG = 0       # Internal debugging only
    LOW = 1         # Background, informational
    NORMAL = 2      # Standard events
    HIGH = 3        # Important events
    URGENT = 4      # Requires attention
    CRITICAL = 5    # Immediate action needed


class EventCategory(str, Enum):
    """Categories of events for filtering."""
    APPLICATION = "application"
    WINDOW = "window"
    SPACE = "space"
    NOTIFICATION = "notification"
    CALENDAR = "calendar"
    FILE_SYSTEM = "file_system"
    SYSTEM = "system"
    PERMISSION = "permission"
    USER_ACTIVITY = "user_activity"
    HELPER = "helper"


# =============================================================================
# Event Data Classes
# =============================================================================

@dataclass
class MacOSEvent:
    """
    Base class for all macOS events.

    All events share common fields for tracking, correlation, and processing.
    """
    event_type: MacOSEventType
    source: str  # Component that generated the event
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MacOSEventPriority = MacOSEventPriority.NORMAL
    event_id: str = field(default="")
    correlation_id: Optional[str] = None  # Link related events
    category: EventCategory = EventCategory.SYSTEM
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing hints
    requires_agi_processing: bool = False  # Should AGI OS process this
    requires_voice_narration: bool = False  # Should be spoken
    is_actionable: bool = False  # Can trigger autonomous actions
    user_initiated: bool = False  # Was this triggered by user action

    def __post_init__(self):
        """Generate event ID if not provided."""
        if not self.event_id:
            self.event_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique event ID."""
        hash_input = f"{self.event_type.value}{self.source}{self.timestamp.isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "category": self.category.value,
            "data": self.data,
            "metadata": self.metadata,
            "requires_agi_processing": self.requires_agi_processing,
            "requires_voice_narration": self.requires_voice_narration,
            "is_actionable": self.is_actionable,
            "user_initiated": self.user_initiated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MacOSEvent":
        """Create event from dictionary."""
        return cls(
            event_type=MacOSEventType(data["event_type"]),
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=MacOSEventPriority(data.get("priority", 2)),
            event_id=data.get("event_id", ""),
            correlation_id=data.get("correlation_id"),
            category=EventCategory(data.get("category", "system")),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            requires_agi_processing=data.get("requires_agi_processing", False),
            requires_voice_narration=data.get("requires_voice_narration", False),
            is_actionable=data.get("is_actionable", False),
            user_initiated=data.get("user_initiated", False),
        )


@dataclass
class AppEvent(MacOSEvent):
    """Event for application lifecycle changes."""
    app_name: str = ""
    bundle_id: str = ""
    process_id: int = 0
    app_path: str = ""
    is_frontmost: bool = False

    def __post_init__(self):
        self.category = EventCategory.APPLICATION
        super().__post_init__()


@dataclass
class WindowEvent(MacOSEvent):
    """Event for window state changes."""
    app_name: str = ""
    window_id: int = 0
    window_title: str = ""
    window_role: str = ""  # AXRole
    frame: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "width": 0, "height": 0})
    space_id: int = 0
    is_focused: bool = False
    is_minimized: bool = False
    is_fullscreen: bool = False

    def __post_init__(self):
        self.category = EventCategory.WINDOW
        super().__post_init__()


@dataclass
class SpaceEvent(MacOSEvent):
    """Event for space/desktop changes."""
    space_id: int = 0
    space_index: int = 0
    space_label: Optional[str] = None
    previous_space_id: Optional[int] = None
    display_id: int = 0
    visible_windows: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.category = EventCategory.SPACE
        super().__post_init__()


@dataclass
class NotificationEvent(MacOSEvent):
    """Event for system and app notifications."""
    notification_id: str = ""
    title: str = ""
    subtitle: str = ""
    body: str = ""
    app_name: str = ""
    bundle_id: str = ""
    category_identifier: str = ""
    thread_identifier: str = ""
    actions: List[str] = field(default_factory=list)
    user_info: Dict[str, Any] = field(default_factory=dict)
    is_silent: bool = False

    def __post_init__(self):
        self.category = EventCategory.NOTIFICATION
        # Notifications often require AGI processing
        self.requires_agi_processing = True
        super().__post_init__()


@dataclass
class CalendarEvent(MacOSEvent):
    """Event for calendar and reminder updates."""
    event_id: str = ""
    calendar_name: str = ""
    title: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    location: str = ""
    notes: str = ""
    attendees: List[str] = field(default_factory=list)
    is_all_day: bool = False
    minutes_until_start: int = 0

    def __post_init__(self):
        self.category = EventCategory.CALENDAR
        # Calendar events often need voice narration
        self.requires_voice_narration = self.minutes_until_start <= 15
        super().__post_init__()


@dataclass
class FileSystemEvent(MacOSEvent):
    """Event for file system changes."""
    path: str = ""
    old_path: str = ""  # For rename/move
    event_flags: int = 0
    is_directory: bool = False
    is_symlink: bool = False
    file_size: int = 0
    modification_time: Optional[datetime] = None

    def __post_init__(self):
        self.category = EventCategory.FILE_SYSTEM
        super().__post_init__()


@dataclass
class PermissionEvent(MacOSEvent):
    """Event for permission status changes."""
    permission_type: str = ""  # accessibility, screen_recording, microphone, etc.
    status: str = ""  # granted, denied, not_determined
    previous_status: Optional[str] = None
    requires_restart: bool = False

    def __post_init__(self):
        self.category = EventCategory.PERMISSION
        # Permission events may require voice feedback
        self.requires_voice_narration = True
        super().__post_init__()


@dataclass
class UserActivityEvent(MacOSEvent):
    """Event for user activity/presence detection."""
    activity_type: str = ""  # idle_started, idle_ended, presence_detected
    idle_duration_seconds: float = 0.0
    last_input_time: Optional[datetime] = None
    input_type: str = ""  # keyboard, mouse, trackpad

    def __post_init__(self):
        self.category = EventCategory.USER_ACTIVITY
        super().__post_init__()


@dataclass
class SystemStateEvent(MacOSEvent):
    """Event for system state changes (sleep, wake, lock, etc.)."""
    state_type: str = ""  # sleep, wake, screen_locked, screen_unlocked
    previous_state: Optional[str] = None
    power_source: str = ""  # battery, ac
    battery_percentage: Optional[float] = None
    display_count: int = 1

    def __post_init__(self):
        self.category = EventCategory.SYSTEM
        super().__post_init__()


# =============================================================================
# Event Type Handlers
# =============================================================================

# Type alias for event handlers
MacOSEventHandler = Callable[[MacOSEvent], Coroutine[Any, Any, None]]


# =============================================================================
# Event Factory
# =============================================================================

class MacOSEventFactory:
    """
    Factory for creating macOS events with proper defaults.

    Provides convenience methods for creating common event types
    with sensible defaults and validation.
    """

    @staticmethod
    def create_app_launched(
        app_name: str,
        bundle_id: str = "",
        process_id: int = 0,
        source: str = "system_event_monitor"
    ) -> AppEvent:
        """Create app launched event."""
        return AppEvent(
            event_type=MacOSEventType.APP_LAUNCHED,
            source=source,
            app_name=app_name,
            bundle_id=bundle_id,
            process_id=process_id,
            requires_agi_processing=True,
            data={
                "app_name": app_name,
                "bundle_id": bundle_id,
                "process_id": process_id,
            }
        )

    @staticmethod
    def create_app_activated(
        app_name: str,
        bundle_id: str = "",
        source: str = "system_event_monitor"
    ) -> AppEvent:
        """Create app activated (focused) event."""
        return AppEvent(
            event_type=MacOSEventType.APP_ACTIVATED,
            source=source,
            app_name=app_name,
            bundle_id=bundle_id,
            is_frontmost=True,
            requires_agi_processing=True,
            data={
                "app_name": app_name,
                "bundle_id": bundle_id,
            }
        )

    @staticmethod
    def create_window_focused(
        app_name: str,
        window_id: int,
        window_title: str,
        source: str = "system_event_monitor"
    ) -> WindowEvent:
        """Create window focused event."""
        return WindowEvent(
            event_type=MacOSEventType.WINDOW_FOCUSED,
            source=source,
            app_name=app_name,
            window_id=window_id,
            window_title=window_title,
            is_focused=True,
            data={
                "app_name": app_name,
                "window_id": window_id,
                "window_title": window_title,
            }
        )

    @staticmethod
    def create_space_changed(
        new_space_id: int,
        new_space_index: int,
        previous_space_id: Optional[int] = None,
        source: str = "system_event_monitor"
    ) -> SpaceEvent:
        """Create space changed event."""
        return SpaceEvent(
            event_type=MacOSEventType.SPACE_CHANGED,
            source=source,
            space_id=new_space_id,
            space_index=new_space_index,
            previous_space_id=previous_space_id,
            requires_agi_processing=True,
            data={
                "space_id": new_space_id,
                "space_index": new_space_index,
                "previous_space_id": previous_space_id,
            }
        )

    @staticmethod
    def create_notification_received(
        title: str,
        body: str,
        app_name: str,
        bundle_id: str = "",
        source: str = "notification_monitor"
    ) -> NotificationEvent:
        """Create notification received event."""
        return NotificationEvent(
            event_type=MacOSEventType.NOTIFICATION_RECEIVED,
            source=source,
            title=title,
            body=body,
            app_name=app_name,
            bundle_id=bundle_id,
            requires_agi_processing=True,
            requires_voice_narration=True,
            priority=MacOSEventPriority.HIGH,
            data={
                "title": title,
                "body": body,
                "app_name": app_name,
            }
        )

    @staticmethod
    def create_screen_locked(source: str = "system_event_monitor") -> SystemStateEvent:
        """Create screen locked event."""
        return SystemStateEvent(
            event_type=MacOSEventType.SCREEN_LOCKED,
            source=source,
            state_type="screen_locked",
            requires_agi_processing=True,
            priority=MacOSEventPriority.HIGH,
            data={"state": "locked"}
        )

    @staticmethod
    def create_screen_unlocked(source: str = "system_event_monitor") -> SystemStateEvent:
        """Create screen unlocked event."""
        return SystemStateEvent(
            event_type=MacOSEventType.SCREEN_UNLOCKED,
            source=source,
            state_type="screen_unlocked",
            requires_agi_processing=True,
            priority=MacOSEventPriority.HIGH,
            data={"state": "unlocked"}
        )

    @staticmethod
    def create_file_modified(
        path: str,
        source: str = "file_system_monitor"
    ) -> FileSystemEvent:
        """Create file modified event."""
        return FileSystemEvent(
            event_type=MacOSEventType.FILE_MODIFIED,
            source=source,
            path=path,
            data={"path": path}
        )

    @staticmethod
    def create_meeting_upcoming(
        title: str,
        minutes_until_start: int,
        calendar_name: str = "",
        source: str = "calendar_monitor"
    ) -> CalendarEvent:
        """Create meeting upcoming event."""
        return CalendarEvent(
            event_type=MacOSEventType.CALENDAR_EVENT_UPCOMING,
            source=source,
            title=title,
            calendar_name=calendar_name,
            minutes_until_start=minutes_until_start,
            requires_agi_processing=True,
            requires_voice_narration=True,
            priority=MacOSEventPriority.URGENT if minutes_until_start <= 5 else MacOSEventPriority.HIGH,
            data={
                "title": title,
                "minutes_until_start": minutes_until_start,
            }
        )

    @staticmethod
    def create_permission_changed(
        permission_type: str,
        status: str,
        previous_status: Optional[str] = None,
        source: str = "permission_manager"
    ) -> PermissionEvent:
        """Create permission changed event."""
        return PermissionEvent(
            event_type=(
                MacOSEventType.PERMISSION_GRANTED
                if status == "granted"
                else MacOSEventType.PERMISSION_DENIED
            ),
            source=source,
            permission_type=permission_type,
            status=status,
            previous_status=previous_status,
            requires_voice_narration=True,
            priority=MacOSEventPriority.HIGH,
            data={
                "permission_type": permission_type,
                "status": status,
            }
        )

    @staticmethod
    def create_helper_error(
        error_message: str,
        component: str,
        source: str = "macos_helper"
    ) -> MacOSEvent:
        """Create helper error event."""
        return MacOSEvent(
            event_type=MacOSEventType.HELPER_ERROR,
            source=source,
            category=EventCategory.HELPER,
            priority=MacOSEventPriority.URGENT,
            requires_agi_processing=True,
            data={
                "error": error_message,
                "component": component,
            }
        )


# Singleton factory instance
event_factory = MacOSEventFactory()
