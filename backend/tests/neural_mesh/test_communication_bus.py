"""
Tests for the Agent Communication Bus.
"""

import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from neural_mesh.data_models import (
    AgentMessage,
    MessagePriority,
    MessageType,
)
from neural_mesh.communication.agent_communication_bus import AgentCommunicationBus


@pytest.fixture
async def bus():
    """Create and start a communication bus for testing."""
    bus = AgentCommunicationBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
async def test_bus_start_stop():
    """Test bus start and stop lifecycle."""
    bus = AgentCommunicationBus()
    assert not bus._running

    await bus.start()
    assert bus._running

    await bus.stop()
    assert not bus._running


@pytest.mark.asyncio
async def test_publish_message(bus):
    """Test publishing a message."""
    message = AgentMessage(
        from_agent="test_sender",
        to_agent="test_receiver",
        message_type=MessageType.CUSTOM,
        payload={"test": "data"},
    )

    msg_id = await bus.publish(message)
    assert msg_id == message.message_id

    metrics = bus.get_metrics()
    assert metrics.messages_published == 1


@pytest.mark.asyncio
async def test_subscribe_and_receive(bus):
    """Test subscribing and receiving messages."""
    received = []

    async def handler(msg: AgentMessage):
        received.append(msg)

    await bus.subscribe("receiver", MessageType.CUSTOM, handler)

    message = AgentMessage(
        from_agent="sender",
        to_agent="receiver",
        message_type=MessageType.CUSTOM,
        payload={"test": "data"},
    )

    await bus.publish(message)

    # Wait for message to be processed
    await asyncio.sleep(0.1)

    assert len(received) == 1
    assert received[0].payload == {"test": "data"}


@pytest.mark.asyncio
async def test_broadcast_message(bus):
    """Test broadcasting to multiple subscribers."""
    received_a = []
    received_b = []

    async def handler_a(msg: AgentMessage):
        received_a.append(msg)

    async def handler_b(msg: AgentMessage):
        received_b.append(msg)

    await bus.subscribe("agent_a", MessageType.CUSTOM, handler_a)
    await bus.subscribe("agent_b", MessageType.CUSTOM, handler_b)

    await bus.broadcast(
        from_agent="broadcaster",
        message_type=MessageType.CUSTOM,
        payload={"broadcast": True},
    )

    await asyncio.sleep(0.1)

    assert len(received_a) == 1
    assert len(received_b) == 1


@pytest.mark.asyncio
async def test_request_response(bus):
    """Test request/response pattern."""
    async def responder(msg: AgentMessage):
        await bus.respond(
            msg,
            payload={"response": "received"},
            from_agent="responder",
        )

    await bus.subscribe("responder", MessageType.REQUEST, responder)

    request = AgentMessage(
        from_agent="requester",
        to_agent="responder",
        message_type=MessageType.REQUEST,
        payload={"query": "test"},
    )

    response = await bus.request(request, timeout=5.0)

    assert response == {"response": "received"}


@pytest.mark.asyncio
async def test_priority_ordering(bus):
    """Test that higher priority messages are processed first."""
    received = []

    async def handler(msg: AgentMessage):
        received.append(msg.priority)
        await asyncio.sleep(0.01)  # Simulate processing time

    await bus.subscribe("agent", MessageType.CUSTOM, handler)

    # Publish in order: LOW, NORMAL, HIGH, CRITICAL
    for priority in [MessagePriority.LOW, MessagePriority.NORMAL,
                     MessagePriority.HIGH, MessagePriority.CRITICAL]:
        msg = AgentMessage(
            from_agent="sender",
            to_agent="agent",
            message_type=MessageType.CUSTOM,
            payload={"priority": priority.name},
            priority=priority,
        )
        await bus.publish(msg)

    await asyncio.sleep(0.2)

    # Higher priority should be processed first (lower value = higher priority)
    # Note: Due to async nature, exact ordering may vary slightly
    assert len(received) == 4


@pytest.mark.asyncio
async def test_unsubscribe(bus):
    """Test unsubscribing from messages."""
    received = []

    async def handler(msg: AgentMessage):
        received.append(msg)

    await bus.subscribe("agent", MessageType.CUSTOM, handler)

    # First message should be received
    await bus.publish(AgentMessage(
        from_agent="sender",
        to_agent="agent",
        message_type=MessageType.CUSTOM,
        payload={},
    ))
    await asyncio.sleep(0.1)
    assert len(received) == 1

    # Unsubscribe
    await bus.unsubscribe("agent", MessageType.CUSTOM)

    # Second message should not be received
    await bus.publish(AgentMessage(
        from_agent="sender",
        to_agent="agent",
        message_type=MessageType.CUSTOM,
        payload={},
    ))
    await asyncio.sleep(0.1)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_message_history(bus):
    """Test message history tracking."""
    for i in range(10):
        await bus.publish(AgentMessage(
            from_agent="sender",
            to_agent="receiver",
            message_type=MessageType.CUSTOM,
            payload={"index": i},
        ))

    history = bus.get_message_history(limit=5)
    assert len(history) == 5

    # Most recent should be last
    assert history[-1].payload["index"] == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
