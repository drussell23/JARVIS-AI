"""
Tests for macOS helper coordinator.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from macos_helper.macos_helper_coordinator import (
    MacOSHelperCoordinator,
    MacOSHelperConfig,
    MacOSHelperState,
    ComponentStatus,
    HelperStats,
)


class TestMacOSHelperState:
    """Tests for MacOSHelperState enum."""

    def test_states_exist(self):
        """Test that all states exist."""
        assert MacOSHelperState.OFFLINE
        assert MacOSHelperState.INITIALIZING
        assert MacOSHelperState.ONBOARDING
        assert MacOSHelperState.ONLINE
        assert MacOSHelperState.DEGRADED
        assert MacOSHelperState.PAUSED
        assert MacOSHelperState.ERROR
        assert MacOSHelperState.SHUTTING_DOWN

    def test_state_values(self):
        """Test state string values."""
        assert MacOSHelperState.OFFLINE.value == "offline"
        assert MacOSHelperState.ONLINE.value == "online"
        assert MacOSHelperState.PAUSED.value == "paused"


class TestMacOSHelperConfig:
    """Tests for MacOSHelperConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MacOSHelperConfig()

        assert config.enable_system_monitor is True
        assert config.enable_notification_monitor is True
        assert config.enable_permission_monitor is True
        assert config.enable_menu_bar is True
        assert config.enable_agi_bridge is True
        assert config.enable_voice_feedback is True
        assert config.health_check_interval_seconds == 30.0
        assert config.auto_restart_on_failure is True
        assert config.max_restart_attempts == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MacOSHelperConfig(
            enable_system_monitor=False,
            enable_menu_bar=False,
            health_check_interval_seconds=60.0,
        )

        assert config.enable_system_monitor is False
        assert config.enable_menu_bar is False
        assert config.health_check_interval_seconds == 60.0


class TestComponentStatus:
    """Tests for ComponentStatus dataclass."""

    def test_component_status_creation(self):
        """Test creating component status."""
        status = ComponentStatus(
            name="event_bus",
            running=True,
            healthy=True,
        )

        assert status.name == "event_bus"
        assert status.running
        assert status.healthy
        assert status.error is None
        assert status.restart_count == 0

    def test_component_status_with_error(self):
        """Test component status with error."""
        status = ComponentStatus(
            name="notification_monitor",
            running=False,
            healthy=False,
            error="Permission denied",
        )

        assert not status.running
        assert not status.healthy
        assert status.error == "Permission denied"


class TestHelperStats:
    """Tests for HelperStats dataclass."""

    def test_default_stats(self):
        """Test default statistics values."""
        stats = HelperStats()

        assert stats.started_at is None
        assert stats.uptime_seconds == 0.0
        assert stats.events_processed == 0
        assert stats.permissions_granted == 0
        assert stats.permissions_denied == 0
        assert stats.component_restarts == 0


class TestMacOSHelperCoordinator:
    """Tests for MacOSHelperCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create a coordinator with minimal config for testing."""
        config = MacOSHelperConfig(
            enable_system_monitor=False,
            enable_notification_monitor=False,
            enable_permission_monitor=False,
            enable_menu_bar=False,
            enable_agi_bridge=False,
            enable_voice_feedback=False,
            announce_startup=False,
        )
        return MacOSHelperCoordinator(config)

    def test_coordinator_creation(self, coordinator):
        """Test coordinator creation."""
        assert coordinator.state == MacOSHelperState.OFFLINE
        assert not coordinator.is_running
        assert coordinator.config is not None

    def test_default_config(self):
        """Test coordinator with default config."""
        coord = MacOSHelperCoordinator()
        assert coord.config.enable_system_monitor is True
        assert coord.config.enable_agi_bridge is True

    @pytest.mark.asyncio
    async def test_start_stop(self, coordinator):
        """Test starting and stopping the coordinator."""
        # Mock the event bus initialization
        with patch.object(coordinator, '_init_event_bus', new_callable=AsyncMock):
            with patch.object(coordinator, '_init_permission_manager', new_callable=AsyncMock):
                with patch.object(coordinator, '_check_permissions', new_callable=AsyncMock) as mock_perms:
                    mock_perms.return_value = True

                    result = await coordinator.start()
                    assert result
                    assert coordinator.is_running

                    await coordinator.stop()
                    assert coordinator.state == MacOSHelperState.OFFLINE

    @pytest.mark.asyncio
    async def test_pause_resume(self, coordinator):
        """Test pausing and resuming the coordinator."""
        # Set state to online first
        coordinator._state = MacOSHelperState.ONLINE

        await coordinator.pause()
        assert coordinator.state == MacOSHelperState.PAUSED

        # Need to mock _determine_health_state for resume
        with patch.object(coordinator, '_determine_health_state', return_value=MacOSHelperState.ONLINE):
            await coordinator.resume()
            assert coordinator.state == MacOSHelperState.ONLINE

    def test_get_status(self, coordinator):
        """Test getting coordinator status."""
        status = coordinator.get_status()

        assert "state" in status
        assert "started_at" in status
        assert "uptime_seconds" in status
        assert "components" in status
        assert "stats" in status

        assert status["state"] == "offline"

    def test_get_event_bus(self, coordinator):
        """Test getting event bus."""
        # Event bus should be None before start
        assert coordinator.get_event_bus() is None

    def test_get_permission_manager(self, coordinator):
        """Test getting permission manager."""
        # Permission manager should be None before start
        assert coordinator.get_permission_manager() is None

    @pytest.mark.asyncio
    async def test_state_changed_callback(self, coordinator):
        """Test state change callbacks."""
        callback = AsyncMock()
        coordinator.on_state_changed(callback)

        # Trigger a state change
        await coordinator._set_state(MacOSHelperState.INITIALIZING)

        callback.assert_called_once_with(MacOSHelperState.INITIALIZING)

    def test_is_running_property(self, coordinator):
        """Test is_running property for different states."""
        coordinator._state = MacOSHelperState.OFFLINE
        assert not coordinator.is_running

        coordinator._state = MacOSHelperState.INITIALIZING
        assert not coordinator.is_running

        coordinator._state = MacOSHelperState.ONLINE
        assert coordinator.is_running

        coordinator._state = MacOSHelperState.DEGRADED
        assert coordinator.is_running

        coordinator._state = MacOSHelperState.PAUSED
        assert not coordinator.is_running

    @pytest.mark.asyncio
    async def test_check_permissions_updates_stats(self, coordinator):
        """Test that permission check updates statistics."""
        # Mock permission manager
        mock_overview = MagicMock()
        mock_overview.all_required_granted = True
        mock_overview.results = {
            "accessibility": MagicMock(status=MagicMock(value="granted")),
            "microphone": MagicMock(status=MagicMock(value="denied")),
        }

        mock_perm_manager = MagicMock()
        mock_perm_manager.check_all_permissions = AsyncMock(return_value=mock_overview)
        coordinator._permission_manager = mock_perm_manager

        result = await coordinator._check_permissions()

        assert result is True
        assert coordinator._stats.permissions_granted == 1
        assert coordinator._stats.permissions_denied == 1


class TestMacOSHelperSingleton:
    """Tests for singleton pattern functions."""

    @pytest.mark.asyncio
    async def test_get_macos_helper(self):
        """Test getting the global helper instance."""
        from macos_helper.macos_helper_coordinator import get_macos_helper, _macos_helper

        # Get helper
        config = MacOSHelperConfig(
            enable_system_monitor=False,
            enable_notification_monitor=False,
            enable_permission_monitor=False,
            enable_menu_bar=False,
            enable_agi_bridge=False,
        )
        helper1 = await get_macos_helper(config)
        helper2 = await get_macos_helper()

        # Should be same instance
        assert helper1 is helper2

        # Cleanup
        import macos_helper.macos_helper_coordinator as coord_module
        coord_module._macos_helper = None
