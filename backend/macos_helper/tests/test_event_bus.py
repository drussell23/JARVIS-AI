"""
Tests for macOS helper event bus.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from macos_helper.event_types import (
    MacOSEventType,
    MacOSEventPriority,
    MacOSEvent,
)
from macos_helper.event_bus import (
    MacOSEventBus,
    MacOSEventSubscription,
)


class TestMacOSEventSubscription:
    """Tests for MacOSEventSubscription."""

    def test_subscription_creation(self):
        """Test subscription creation."""
        handler = AsyncMock()
        sub = MacOSEventSubscription(
            event_type=MacOSEventType.APP_LAUNCHED,
            handler=handler,
            priority=10,
        )

        assert sub.event_type == MacOSEventType.APP_LAUNCHED
        assert sub.handler == handler
        assert sub.priority == 10
        assert sub.enabled
        assert sub.call_count == 0
        assert sub.consecutive_failures == 0

    def test_subscription_disable(self):
        """Test disabling subscription."""
        handler = AsyncMock()
        sub = MacOSEventSubscription(
            event_type=MacOSEventType.APP_LAUNCHED,
            handler=handler,
        )

        sub.disable()
        assert not sub.enabled

    def test_subscription_enable(self):
        """Test enabling subscription."""
        handler = AsyncMock()
        sub = MacOSEventSubscription(
            event_type=MacOSEventType.APP_LAUNCHED,
            handler=handler,
        )

        sub.disable()
        sub.enable()
        assert sub.enabled


class TestMacOSEventBus:
    """Tests for MacOSEventBus."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh event bus for testing."""
        return MacOSEventBus(enable_agi_bridge=False)

    @pytest.mark.asyncio
    async def test_event_bus_creation(self, event_bus):
        """Test event bus creation."""
        assert not event_bus._running
        assert event_bus._queue is not None

    @pytest.mark.asyncio
    async def test_start_stop(self, event_bus):
        """Test starting and stopping the event bus."""
        await event_bus.start()
        assert event_bus._running

        await event_bus.stop()
        assert not event_bus._running

    @pytest.mark.asyncio
    async def test_subscribe(self, event_bus):
        """Test subscribing to events."""
        handler = AsyncMock()
        sub = event_bus.subscribe(MacOSEventType.APP_LAUNCHED, handler)

        assert sub is not None
        assert sub in event_bus._subscriptions[MacOSEventType.APP_LAUNCHED]

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        handler = AsyncMock()
        sub = event_bus.subscribe(MacOSEventType.APP_LAUNCHED, handler)

        result = event_bus.unsubscribe(sub)
        assert result
        assert sub not in event_bus._subscriptions[MacOSEventType.APP_LAUNCHED]

    @pytest.mark.asyncio
    async def test_emit_event(self, event_bus):
        """Test emitting events."""
        await event_bus.start()

        handler = AsyncMock()
        event_bus.subscribe(MacOSEventType.APP_LAUNCHED, handler)

        event = MacOSEvent(
            event_type=MacOSEventType.APP_LAUNCHED,
            source="test",
            data={"app_name": "TestApp"},
        )

        result = await event_bus.emit(event, bridge_to_agi=False)
        assert result

        # Wait for event processing
        await asyncio.sleep(0.1)

        handler.assert_called_once()

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_emit_with_priority(self, event_bus):
        """Test that higher priority handlers are called first."""
        await event_bus.start()

        call_order = []

        async def handler_low(event):
            call_order.append("low")

        async def handler_high(event):
            call_order.append("high")

        # Subscribe with different priorities (lower number = higher priority)
        event_bus.subscribe(MacOSEventType.APP_LAUNCHED, handler_low, priority=10)
        event_bus.subscribe(MacOSEventType.APP_LAUNCHED, handler_high, priority=1)

        event = MacOSEvent(
            event_type=MacOSEventType.APP_LAUNCHED,
            source="test",
        )

        await event_bus.emit(event, bridge_to_agi=False)
        await asyncio.sleep(0.1)

        # High priority should be called first
        assert call_order[0] == "high"
        assert call_order[1] == "low"

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_deduplication(self, event_bus):
        """Test event deduplication."""
        await event_bus.start()

        handler = AsyncMock()
        event_bus.subscribe(MacOSEventType.APP_LAUNCHED, handler)

        event1 = MacOSEvent(
            event_type=MacOSEventType.APP_LAUNCHED,
            source="test",
            data={"app_name": "TestApp"},
        )

        # Emit the same event twice with same data
        await event_bus.emit(event1, deduplicate=True, bridge_to_agi=False)
        await event_bus.emit(event1, deduplicate=True, bridge_to_agi=False)
        await asyncio.sleep(0.1)

        # Handler should only be called once due to deduplication
        assert handler.call_count == 1

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self, event_bus):
        """Test getting event bus statistics."""
        stats = event_bus.get_stats()

        assert "running" in stats
        assert "events_emitted" in stats
        assert "events_processed" in stats
        assert "subscription_count" in stats

    @pytest.mark.asyncio
    async def test_handler_failure_circuit_breaker(self, event_bus):
        """Test circuit breaker on handler failures."""
        await event_bus.start()

        # Handler that always fails
        handler = AsyncMock(side_effect=Exception("Test failure"))
        sub = event_bus.subscribe(MacOSEventType.APP_LAUNCHED, handler)

        # Emit events to trigger failures
        for _ in range(5):
            event = MacOSEvent(
                event_type=MacOSEventType.APP_LAUNCHED,
                source="test",
            )
            await event_bus.emit(event, bridge_to_agi=False)
            await asyncio.sleep(0.05)

        # After multiple failures, subscription should be disabled
        assert sub.consecutive_failures >= event_bus._max_failures

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus):
        """Test subscribing to all events with wildcard."""
        await event_bus.start()

        handler = AsyncMock()
        # Subscribe to all events (None means wildcard)
        event_bus.subscribe(None, handler)

        events = [
            MacOSEvent(event_type=MacOSEventType.APP_LAUNCHED, source="test"),
            MacOSEvent(event_type=MacOSEventType.WINDOW_CREATED, source="test"),
            MacOSEvent(event_type=MacOSEventType.NOTIFICATION_RECEIVED, source="test"),
        ]

        for event in events:
            await event_bus.emit(event, bridge_to_agi=False)
            await asyncio.sleep(0.05)

        assert handler.call_count == 3

        await event_bus.stop()
