"""
Tests for AGI OS integration bridge.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from macos_helper.agi_integration import (
    AGIBridge,
    AGIBridgeConfig,
    AGIEventMapping,
    BridgeStats,
)


class TestAGIEventMapping:
    """Tests for AGI event mapping."""

    def test_context_event_mapping(self):
        """Test mapping of context change events."""
        context_events = [
            "app_launched",
            "app_activated",
            "app_deactivated",
            "app_terminated",
            "window_created",
            "window_closed",
            "space_changed",
            "idle_start",
            "idle_end",
        ]

        for event_type in context_events:
            result = AGIEventMapping.get_agi_event_type(event_type)
            assert result == "CONTEXT_CHANGED", f"Expected CONTEXT_CHANGED for {event_type}"

    def test_opportunity_event_mapping(self):
        """Test mapping of opportunity events."""
        opportunity_events = [
            "notification_received",
            "notification_action_available",
            "file_created",
            "file_modified",
            "calendar_event",
            "reminder_due",
        ]

        for event_type in opportunity_events:
            result = AGIEventMapping.get_agi_event_type(event_type)
            assert result == "OPPORTUNITY_DETECTED", f"Expected OPPORTUNITY_DETECTED for {event_type}"

    def test_issue_event_mapping(self):
        """Test mapping of issue events."""
        issue_events = [
            "permission_denied",
            "permission_revoked",
            "app_crashed",
            "monitor_error",
        ]

        for event_type in issue_events:
            result = AGIEventMapping.get_agi_event_type(event_type)
            assert result == "ISSUE_DETECTED", f"Expected ISSUE_DETECTED for {event_type}"

    def test_user_action_event_mapping(self):
        """Test mapping of user action events."""
        user_events = [
            "notification_dismissed",
            "notification_clicked",
        ]

        for event_type in user_events:
            result = AGIEventMapping.get_agi_event_type(event_type)
            assert result == "USER_ACTION", f"Expected USER_ACTION for {event_type}"

    def test_unknown_event_mapping(self):
        """Test mapping of unknown events."""
        result = AGIEventMapping.get_agi_event_type("unknown_event")
        assert result == "SYSTEM_STATUS"

    def test_priority_mapping(self):
        """Test priority mapping."""
        assert AGIEventMapping.get_agi_priority("debug") == "LOW"
        assert AGIEventMapping.get_agi_priority("low") == "LOW"
        assert AGIEventMapping.get_agi_priority("normal") == "MEDIUM"
        assert AGIEventMapping.get_agi_priority("high") == "HIGH"
        assert AGIEventMapping.get_agi_priority("critical") == "CRITICAL"
        assert AGIEventMapping.get_agi_priority("unknown") == "MEDIUM"


class TestAGIBridgeConfig:
    """Tests for AGI bridge configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AGIBridgeConfig()

        assert config.bridge_all_events is True
        assert config.min_priority_to_bridge == "low"
        assert config.enable_voice_feedback is True
        assert config.voice_for_important_events is True
        assert config.enable_action_requests is True
        assert config.auto_approve_safe_actions is True
        assert config.send_context_to_uae is True
        assert config.send_events_to_cai is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = AGIBridgeConfig(
            bridge_all_events=False,
            min_priority_to_bridge="high",
            enable_voice_feedback=False,
        )

        assert config.bridge_all_events is False
        assert config.min_priority_to_bridge == "high"
        assert config.enable_voice_feedback is False


class TestBridgeStats:
    """Tests for bridge statistics."""

    def test_default_stats(self):
        """Test default statistics values."""
        stats = BridgeStats()

        assert stats.started_at is None
        assert stats.events_sent_to_agi == 0
        assert stats.events_received_from_agi == 0
        assert stats.actions_requested == 0
        assert stats.actions_completed == 0
        assert stats.voice_announcements == 0


class TestAGIBridge:
    """Tests for AGI integration bridge."""

    @pytest.fixture
    def bridge(self):
        """Create a bridge for testing."""
        config = AGIBridgeConfig(
            enable_voice_feedback=False,
            send_context_to_uae=False,
            send_events_to_cai=False,
        )
        return AGIBridge(config)

    def test_bridge_creation(self, bridge):
        """Test bridge creation."""
        assert bridge is not None
        assert not bridge._running
        assert bridge._agi_coordinator is None
        assert bridge._macos_helper is None

    @pytest.mark.asyncio
    async def test_start_without_components(self, bridge):
        """Test starting bridge without AGI OS or macOS helper."""
        # Should handle missing components gracefully
        with patch.object(bridge, '_connect_to_agi_os', new_callable=AsyncMock):
            with patch.object(bridge, '_connect_to_macos_helper', new_callable=AsyncMock):
                with patch.object(bridge, '_setup_event_forwarding', new_callable=AsyncMock):
                    with patch.object(bridge, '_setup_action_handling', new_callable=AsyncMock):
                        result = await bridge.start()
                        assert result
                        assert bridge._running

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_stop(self, bridge):
        """Test stopping the bridge."""
        bridge._running = True
        await bridge.stop()
        assert not bridge._running

    def test_get_stats(self, bridge):
        """Test getting bridge statistics."""
        stats = bridge.get_stats()

        assert "started_at" in stats
        assert "running" in stats
        assert "events_sent_to_agi" in stats
        assert "actions_requested" in stats
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_request_notification(self, bridge):
        """Test requesting a notification."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = await bridge.request_notification(
                title="Test",
                message="Test message",
            )

            assert result
            assert bridge._stats.actions_requested == 1
            assert bridge._stats.actions_completed == 1

    @pytest.mark.asyncio
    async def test_request_open_url(self, bridge):
        """Test requesting to open a URL."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = await bridge.request_open_url("https://example.com")

            assert result
            assert bridge._stats.actions_requested == 1

    @pytest.mark.asyncio
    async def test_request_open_url_no_url(self, bridge):
        """Test requesting to open URL without URL."""
        result = await bridge.request_open_url("")
        assert not result

    @pytest.mark.asyncio
    async def test_generate_event_description(self, bridge):
        """Test generating event descriptions."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.event_type = MagicMock(value="app_launched")
        mock_event.data = {"app_name": "Safari"}

        description = bridge._generate_event_description(mock_event)
        assert "Safari" in description
        assert "launched" in description.lower()

    @pytest.mark.asyncio
    async def test_event_description_notification(self, bridge):
        """Test event description for notifications."""
        mock_event = MagicMock()
        mock_event.event_type = MagicMock(value="notification_received")
        mock_event.data = {"app_name": "Messages", "title": "New Message"}

        description = bridge._generate_event_description(mock_event)
        assert "Messages" in description
        assert "New Message" in description


class TestAGIBridgeSingleton:
    """Tests for AGI bridge singleton pattern."""

    @pytest.mark.asyncio
    async def test_get_agi_bridge(self):
        """Test getting the global bridge instance."""
        from macos_helper.agi_integration import get_agi_bridge, _agi_bridge

        config = AGIBridgeConfig(enable_voice_feedback=False)
        bridge1 = await get_agi_bridge(config)
        bridge2 = await get_agi_bridge()

        assert bridge1 is bridge2

        # Cleanup
        import macos_helper.agi_integration as bridge_module
        bridge_module._agi_bridge = None
