"""
Tests for macOS helper event types.
"""

import pytest
from datetime import datetime
from uuid import UUID

from macos_helper.event_types import (
    MacOSEventType,
    MacOSEventPriority,
    MacOSEvent,
    AppEvent,
    WindowEvent,
    SpaceEvent,
    NotificationEvent,
    FileSystemEvent,
    PermissionEvent,
    MacOSEventFactory,
)


class TestMacOSEventType:
    """Tests for MacOSEventType enum."""

    def test_app_events_exist(self):
        """Test that app event types exist."""
        assert MacOSEventType.APP_LAUNCHED
        assert MacOSEventType.APP_TERMINATED
        assert MacOSEventType.APP_ACTIVATED
        assert MacOSEventType.APP_DEACTIVATED
        assert MacOSEventType.APP_HIDDEN
        assert MacOSEventType.APP_UNHIDDEN

    def test_window_events_exist(self):
        """Test that window event types exist."""
        assert MacOSEventType.WINDOW_CREATED
        assert MacOSEventType.WINDOW_CLOSED
        assert MacOSEventType.WINDOW_FOCUSED
        assert MacOSEventType.WINDOW_MOVED
        assert MacOSEventType.WINDOW_RESIZED
        assert MacOSEventType.WINDOW_MINIMIZED
        assert MacOSEventType.WINDOW_FULLSCREEN

    def test_space_events_exist(self):
        """Test that space event types exist."""
        assert MacOSEventType.SPACE_CHANGED
        assert MacOSEventType.SPACE_CREATED
        assert MacOSEventType.SPACE_DESTROYED
        assert MacOSEventType.DISPLAY_CHANGED

    def test_notification_events_exist(self):
        """Test that notification event types exist."""
        assert MacOSEventType.NOTIFICATION_RECEIVED
        assert MacOSEventType.NOTIFICATION_DISMISSED
        assert MacOSEventType.NOTIFICATION_CLICKED
        assert MacOSEventType.NOTIFICATION_ACTION_CLICKED

    def test_filesystem_events_exist(self):
        """Test that filesystem event types exist."""
        assert MacOSEventType.FILE_CREATED
        assert MacOSEventType.FILE_MODIFIED
        assert MacOSEventType.FILE_DELETED
        assert MacOSEventType.FILE_RENAMED
        assert MacOSEventType.DIRECTORY_CREATED

    def test_system_events_exist(self):
        """Test that system event types exist."""
        assert MacOSEventType.SYSTEM_WAKE
        assert MacOSEventType.SYSTEM_SLEEP
        assert MacOSEventType.SCREEN_LOCK
        assert MacOSEventType.SCREEN_UNLOCK
        assert MacOSEventType.VOLUME_MOUNTED
        assert MacOSEventType.VOLUME_UNMOUNTED
        assert MacOSEventType.NETWORK_CHANGED

    def test_permission_events_exist(self):
        """Test that permission event types exist."""
        assert MacOSEventType.PERMISSION_GRANTED
        assert MacOSEventType.PERMISSION_DENIED
        assert MacOSEventType.PERMISSION_REVOKED
        assert MacOSEventType.PERMISSION_CHECK

    def test_idle_events_exist(self):
        """Test that idle event types exist."""
        assert MacOSEventType.IDLE_START
        assert MacOSEventType.IDLE_END


class TestMacOSEventPriority:
    """Tests for MacOSEventPriority enum."""

    def test_priorities_exist(self):
        """Test that all priority levels exist."""
        assert MacOSEventPriority.DEBUG
        assert MacOSEventPriority.LOW
        assert MacOSEventPriority.NORMAL
        assert MacOSEventPriority.HIGH
        assert MacOSEventPriority.CRITICAL

    def test_priority_values(self):
        """Test priority string values."""
        assert MacOSEventPriority.DEBUG.value == "debug"
        assert MacOSEventPriority.LOW.value == "low"
        assert MacOSEventPriority.NORMAL.value == "normal"
        assert MacOSEventPriority.HIGH.value == "high"
        assert MacOSEventPriority.CRITICAL.value == "critical"


class TestMacOSEvent:
    """Tests for MacOSEvent dataclass."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = MacOSEvent(
            event_type=MacOSEventType.APP_LAUNCHED,
            source="test",
            data={"app_name": "TestApp"},
        )

        assert event.event_type == MacOSEventType.APP_LAUNCHED
        assert event.source == "test"
        assert event.data["app_name"] == "TestApp"
        assert event.priority == MacOSEventPriority.NORMAL
        assert isinstance(event.event_id, str)
        assert isinstance(event.timestamp, datetime)

    def test_event_with_priority(self):
        """Test event with custom priority."""
        event = MacOSEvent(
            event_type=MacOSEventType.PERMISSION_DENIED,
            source="permission_manager",
            priority=MacOSEventPriority.HIGH,
        )

        assert event.priority == MacOSEventPriority.HIGH

    def test_event_to_dict(self):
        """Test event serialization to dict."""
        event = MacOSEvent(
            event_type=MacOSEventType.NOTIFICATION_RECEIVED,
            source="notification_monitor",
            data={"title": "Test Notification"},
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "notification_received"
        assert event_dict["source"] == "notification_monitor"
        assert event_dict["data"]["title"] == "Test Notification"
        assert "event_id" in event_dict
        assert "timestamp" in event_dict


class TestAppEvent:
    """Tests for AppEvent dataclass."""

    def test_app_event_creation(self):
        """Test app event creation."""
        event = AppEvent(
            event_type=MacOSEventType.APP_LAUNCHED,
            source="system_monitor",
            app_name="Cursor",
            bundle_id="com.todesktop.230313mzl4w4u92",
            pid=12345,
        )

        assert event.app_name == "Cursor"
        assert event.bundle_id == "com.todesktop.230313mzl4w4u92"
        assert event.pid == 12345


class TestWindowEvent:
    """Tests for WindowEvent dataclass."""

    def test_window_event_creation(self):
        """Test window event creation."""
        event = WindowEvent(
            event_type=MacOSEventType.WINDOW_FOCUSED,
            source="system_monitor",
            window_id=123,
            window_title="main.py - Cursor",
            app_name="Cursor",
            bundle_id="com.todesktop.230313mzl4w4u92",
            frame={"x": 0, "y": 0, "width": 1920, "height": 1080},
        )

        assert event.window_id == 123
        assert event.window_title == "main.py - Cursor"
        assert event.frame["width"] == 1920


class TestSpaceEvent:
    """Tests for SpaceEvent dataclass."""

    def test_space_event_creation(self):
        """Test space event creation."""
        event = SpaceEvent(
            event_type=MacOSEventType.SPACE_CHANGED,
            source="system_monitor",
            space_id=2,
            space_index=1,
            is_fullscreen=False,
        )

        assert event.space_id == 2
        assert event.space_index == 1
        assert not event.is_fullscreen


class TestNotificationEvent:
    """Tests for NotificationEvent dataclass."""

    def test_notification_event_creation(self):
        """Test notification event creation."""
        event = NotificationEvent(
            event_type=MacOSEventType.NOTIFICATION_RECEIVED,
            source="notification_monitor",
            notification_id="abc123",
            title="New Message",
            body="Hello from test!",
            app_name="Messages",
            bundle_id="com.apple.MobileSMS",
        )

        assert event.notification_id == "abc123"
        assert event.title == "New Message"
        assert event.body == "Hello from test!"


class TestFileSystemEvent:
    """Tests for FileSystemEvent dataclass."""

    def test_filesystem_event_creation(self):
        """Test filesystem event creation."""
        event = FileSystemEvent(
            event_type=MacOSEventType.FILE_MODIFIED,
            source="fs_monitor",
            path="/Users/test/code/main.py",
            is_directory=False,
        )

        assert event.path == "/Users/test/code/main.py"
        assert not event.is_directory


class TestPermissionEvent:
    """Tests for PermissionEvent dataclass."""

    def test_permission_event_creation(self):
        """Test permission event creation."""
        event = PermissionEvent(
            event_type=MacOSEventType.PERMISSION_GRANTED,
            source="permission_manager",
            permission_type="accessibility",
            granted=True,
        )

        assert event.permission_type == "accessibility"
        assert event.granted


class TestMacOSEventFactory:
    """Tests for MacOSEventFactory."""

    def test_create_app_event(self):
        """Test creating app events via factory."""
        event = MacOSEventFactory.create_app_event(
            event_type=MacOSEventType.APP_LAUNCHED,
            app_name="Safari",
            bundle_id="com.apple.Safari",
            pid=54321,
        )

        assert isinstance(event, AppEvent)
        assert event.app_name == "Safari"
        assert event.bundle_id == "com.apple.Safari"

    def test_create_window_event(self):
        """Test creating window events via factory."""
        event = MacOSEventFactory.create_window_event(
            event_type=MacOSEventType.WINDOW_CREATED,
            window_id=456,
            window_title="Test Window",
            app_name="Finder",
        )

        assert isinstance(event, WindowEvent)
        assert event.window_id == 456

    def test_create_space_event(self):
        """Test creating space events via factory."""
        event = MacOSEventFactory.create_space_event(
            event_type=MacOSEventType.SPACE_CHANGED,
            space_id=3,
            space_index=2,
        )

        assert isinstance(event, SpaceEvent)
        assert event.space_id == 3

    def test_create_notification_event(self):
        """Test creating notification events via factory."""
        event = MacOSEventFactory.create_notification_event(
            event_type=MacOSEventType.NOTIFICATION_RECEIVED,
            notification_id="xyz789",
            title="Alert",
            body="Test alert",
            app_name="System",
        )

        assert isinstance(event, NotificationEvent)
        assert event.notification_id == "xyz789"

    def test_create_permission_event(self):
        """Test creating permission events via factory."""
        event = MacOSEventFactory.create_permission_event(
            event_type=MacOSEventType.PERMISSION_DENIED,
            permission_type="microphone",
            granted=False,
        )

        assert isinstance(event, PermissionEvent)
        assert event.permission_type == "microphone"
        assert not event.granted
