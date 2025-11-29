"""
Pytest configuration and shared fixtures for macOS helper tests.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Configure pytest-asyncio
pytest_plugins = ['pytest_asyncio']


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus for testing."""
    bus = MagicMock()
    bus.emit = AsyncMock(return_value=True)
    bus.subscribe = MagicMock(return_value=MagicMock())
    bus.get_stats = MagicMock(return_value={
        "running": True,
        "events_emitted": 0,
        "events_processed": 0,
    })
    return bus


@pytest.fixture
def mock_permission_manager():
    """Create a mock permission manager for testing."""
    manager = MagicMock()
    manager.check_all_permissions = AsyncMock(return_value=MagicMock(
        all_required_granted=True,
        results={},
    ))
    manager.check_accessibility = AsyncMock(return_value=MagicMock(
        status=MagicMock(value="granted"),
    ))
    return manager


@pytest.fixture
def mock_system_monitor():
    """Create a mock system monitor for testing."""
    monitor = MagicMock()
    monitor.start = AsyncMock()
    monitor.stop = AsyncMock()
    monitor.get_status = MagicMock(return_value={
        "running": True,
        "app_count": 10,
        "window_count": 25,
    })
    return monitor


@pytest.fixture
def mock_notification_monitor():
    """Create a mock notification monitor for testing."""
    monitor = MagicMock()
    monitor.start = AsyncMock()
    monitor.stop = AsyncMock()
    monitor.get_stats = MagicMock(return_value={
        "running": True,
        "pending_count": 5,
    })
    return monitor


@pytest.fixture
def mock_menu_bar():
    """Create a mock menu bar indicator for testing."""
    menu_bar = MagicMock()
    menu_bar.start = AsyncMock(return_value=True)
    menu_bar.stop = AsyncMock()
    menu_bar.set_state = MagicMock()
    menu_bar.update_stats = MagicMock()
    menu_bar.set_permission_status = MagicMock()
    menu_bar.show_notification = MagicMock()
    return menu_bar


@pytest.fixture
def mock_agi_coordinator():
    """Create a mock AGI OS coordinator for testing."""
    coordinator = MagicMock()
    coordinator.get_component = MagicMock(return_value=None)
    coordinator.speak = AsyncMock()
    return coordinator


@pytest.fixture
def mock_voice_communicator():
    """Create a mock voice communicator for testing."""
    voice = MagicMock()
    voice.speak = AsyncMock(return_value="msg_123")
    voice.get_status = MagicMock(return_value={"running": True})
    return voice


@pytest.fixture
def sample_macos_event():
    """Create a sample macOS event for testing."""
    from macos_helper.event_types import MacOSEvent, MacOSEventType
    return MacOSEvent(
        event_type=MacOSEventType.APP_LAUNCHED,
        source="test",
        data={"app_name": "TestApp", "bundle_id": "com.test.app"},
    )


@pytest.fixture
def sample_app_event():
    """Create a sample app event for testing."""
    from macos_helper.event_types import AppEvent, MacOSEventType
    return AppEvent(
        event_type=MacOSEventType.APP_LAUNCHED,
        source="test",
        app_name="Cursor",
        bundle_id="com.todesktop.230313mzl4w4u92",
        pid=12345,
    )


@pytest.fixture
def sample_window_event():
    """Create a sample window event for testing."""
    from macos_helper.event_types import WindowEvent, MacOSEventType
    return WindowEvent(
        event_type=MacOSEventType.WINDOW_FOCUSED,
        source="test",
        window_id=123,
        window_title="main.py - Cursor",
        app_name="Cursor",
    )


@pytest.fixture
def sample_notification_event():
    """Create a sample notification event for testing."""
    from macos_helper.event_types import NotificationEvent, MacOSEventType
    return NotificationEvent(
        event_type=MacOSEventType.NOTIFICATION_RECEIVED,
        source="test",
        notification_id="notif_123",
        title="New Message",
        body="Hello World",
        app_name="Messages",
    )


@pytest.fixture
def minimal_coordinator_config():
    """Create a minimal coordinator config for testing."""
    from macos_helper.macos_helper_coordinator import MacOSHelperConfig
    return MacOSHelperConfig(
        enable_system_monitor=False,
        enable_notification_monitor=False,
        enable_permission_monitor=False,
        enable_menu_bar=False,
        enable_agi_bridge=False,
        enable_voice_feedback=False,
        announce_startup=False,
    )
