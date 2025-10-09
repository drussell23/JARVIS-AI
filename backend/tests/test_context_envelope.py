"""
Test suite for Context Envelope models.
"""
import pytest
from datetime import datetime, timedelta

from backend.core.models.context_envelope import (
    ContextEnvelope,
    ContextMetadata,
    ContextCategory,
    ContextPriority,
    ContextState,
    VisionContextPayload,
    InteractionContextPayload,
)


class TestContextEnvelope:
    """Test ContextEnvelope functionality."""

    def test_envelope_creation(self):
        """Test basic envelope creation."""
        metadata = ContextMetadata(
            category=ContextCategory.VISION,
            priority=ContextPriority.HIGH,
            source="test",
        )

        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test terminal",
        )

        envelope = ContextEnvelope(
            metadata=metadata,
            payload=payload,
            ttl_seconds=60,
        )

        assert envelope.metadata.category == ContextCategory.VISION
        assert envelope.state == ContextState.PENDING
        assert envelope.access_count == 0

    def test_envelope_expiry(self):
        """Test expiry logic."""
        metadata = ContextMetadata()
        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test",
        )

        envelope = ContextEnvelope(
            metadata=metadata,
            payload=payload,
            ttl_seconds=1,  # 1 second
        )

        # Should not be expired immediately
        assert not envelope.is_expired()

        # Should be expired after TTL
        future = datetime.utcnow() + timedelta(seconds=2)
        assert envelope.is_expired(future)

    def test_relevance_score(self):
        """Test relevance scoring."""
        metadata = ContextMetadata(priority=ContextPriority.CRITICAL)
        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test",
        )

        envelope = ContextEnvelope(
            metadata=metadata,
            payload=payload,
            decay_rate=0.01,  # 1% decay per second
        )

        # Fresh envelope should have high relevance
        score1 = envelope.relevance_score()
        assert score1 > 0.9

        # After some time, should decay
        future = datetime.utcnow() + timedelta(seconds=30)
        score2 = envelope.relevance_score(future)
        assert score2 < score1

    def test_access_tracking(self):
        """Test access counting."""
        metadata = ContextMetadata()
        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test",
        )

        envelope = ContextEnvelope(metadata=metadata, payload=payload)

        assert envelope.access_count == 0
        assert envelope.last_accessed is None

        envelope.access()

        assert envelope.access_count == 1
        assert envelope.last_accessed is not None
        assert envelope.state == ContextState.ACTIVE

    def test_consume(self):
        """Test consume operation."""
        metadata = ContextMetadata()
        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test",
        )

        envelope = ContextEnvelope(metadata=metadata, payload=payload)

        envelope.consume()

        assert envelope.state == ContextState.CONSUMED
        assert envelope.access_count == 1

    def test_invalidate(self):
        """Test invalidation."""
        metadata = ContextMetadata()
        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test",
        )

        envelope = ContextEnvelope(metadata=metadata, payload=payload)

        envelope.invalidate("test reason")

        assert envelope.state == ContextState.INVALIDATED
        assert envelope.constraints["invalidation_reason"] == "test reason"
        assert not envelope.is_valid()

    def test_serialization(self):
        """Test to_dict serialization."""
        metadata = ContextMetadata(
            category=ContextCategory.VISION,
            priority=ContextPriority.HIGH,
            tags=("test", "terminal"),
        )

        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test terminal",
            ocr_text="Some text",
        )

        envelope = ContextEnvelope(metadata=metadata, payload=payload)

        data = envelope.to_dict()

        assert data["metadata"]["category"] == "VISION"
        assert data["metadata"]["priority"] == "HIGH"
        assert "test" in data["metadata"]["tags"]
        assert data["payload"]["window_type"] == "terminal"
        assert data["state"] == "pending"


class TestVisionContextPayload:
    """Test VisionContextPayload."""

    def test_payload_creation(self):
        """Test payload creation and validation."""
        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test terminal",
        )

        assert payload.validate()
        assert payload.window_type == "terminal"

    def test_payload_serialization(self):
        """Test serialization roundtrip."""
        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test",
            ocr_text="Sample text",
            detected_elements=[{"type": "button", "text": "Click"}],
        )

        data = payload.to_dict()
        restored = VisionContextPayload.from_dict(data)

        assert restored.window_type == payload.window_type
        assert restored.ocr_text == payload.ocr_text
        assert len(restored.detected_elements) == 1


class TestInteractionContextPayload:
    """Test InteractionContextPayload."""

    def test_interaction_payload(self):
        """Test interaction payload creation."""
        payload = InteractionContextPayload(
            question_text="Would you like more details?",
            expected_response_types=("affirmative", "negative", "inquiry"),
            linked_context_id="ctx123",
        )

        assert payload.validate()
        assert len(payload.expected_response_types) == 3

    def test_interaction_serialization(self):
        """Test serialization."""
        payload = InteractionContextPayload(
            question_text="Test question?",
            expected_response_types=("yes", "no"),
            clarification_needed=True,
        )

        data = payload.to_dict()
        restored = InteractionContextPayload.from_dict(data)

        assert restored.question_text == payload.question_text
        assert restored.clarification_needed is True
