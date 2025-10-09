"""
Test suite for Context Store implementations.
"""
import pytest
import asyncio
from datetime import datetime, timedelta

from backend.core.context.memory_store import InMemoryContextStore
from backend.core.context.store_interface import ContextQuery
from backend.core.models.context_envelope import (
    ContextEnvelope,
    ContextMetadata,
    ContextCategory,
    ContextPriority,
    ContextState,
    VisionContextPayload,
)


@pytest.fixture
def sample_envelope():
    """Create sample envelope for testing."""
    metadata = ContextMetadata(
        category=ContextCategory.VISION,
        priority=ContextPriority.HIGH,
        source="test",
        tags=("terminal", "error"),
    )

    payload = VisionContextPayload(
        window_type="terminal",
        window_id="w1",
        space_id="s1",
        snapshot_id="snap1",
        summary="Test terminal window",
    )

    return ContextEnvelope(
        metadata=metadata,
        payload=payload,
        ttl_seconds=120,
    )


class TestInMemoryContextStore:
    """Test InMemoryContextStore."""

    @pytest.mark.asyncio
    async def test_add_and_get(self, sample_envelope):
        """Test adding and retrieving envelope."""
        store = InMemoryContextStore(max_size=100)

        envelope_id = await store.add(sample_envelope)

        assert envelope_id == sample_envelope.metadata.id

        retrieved = await store.get(envelope_id)

        assert retrieved is not None
        assert retrieved.metadata.id == envelope_id
        assert retrieved.access_count == 1  # Access increments count

    @pytest.mark.asyncio
    async def test_update(self, sample_envelope):
        """Test updating envelope."""
        store = InMemoryContextStore()

        envelope_id = await store.add(sample_envelope)

        # Modify
        sample_envelope.access()

        success = await store.update(sample_envelope)

        assert success is True

        # Verify update
        retrieved = await store.get(envelope_id)
        assert retrieved.access_count == 2  # 1 from add access, 1 from manual access

    @pytest.mark.asyncio
    async def test_delete(self, sample_envelope):
        """Test deleting envelope."""
        store = InMemoryContextStore()

        envelope_id = await store.add(sample_envelope)

        success = await store.delete(envelope_id)
        assert success is True

        # Should not exist
        retrieved = await store.get(envelope_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_query_by_category(self, sample_envelope):
        """Test querying by category."""
        store = InMemoryContextStore()

        await store.add(sample_envelope)

        # Create another envelope with different category
        metadata2 = ContextMetadata(
            category=ContextCategory.COMMAND,
            priority=ContextPriority.NORMAL,
        )
        payload2 = VisionContextPayload(
            window_type="browser",
            window_id="w2",
            space_id="s2",
            snapshot_id="snap2",
            summary="Browser",
        )
        envelope2 = ContextEnvelope(metadata=metadata2, payload=payload2)

        await store.add(envelope2)

        # Query vision only
        query = ContextQuery().with_category("VISION")
        results = await store.query(query)

        assert len(results) == 1
        assert results[0].metadata.category == ContextCategory.VISION

    @pytest.mark.asyncio
    async def test_query_by_tags(self, sample_envelope):
        """Test querying by tags."""
        store = InMemoryContextStore()

        await store.add(sample_envelope)

        results = await store.get_by_tags("terminal")

        assert len(results) == 1
        assert "terminal" in results[0].metadata.tags

    @pytest.mark.asyncio
    async def test_query_by_relevance(self, sample_envelope):
        """Test sorting by relevance."""
        store = InMemoryContextStore()

        # Add high priority envelope
        await store.add(sample_envelope)

        # Add low priority envelope
        metadata_low = ContextMetadata(
            category=ContextCategory.VISION,
            priority=ContextPriority.LOW,
        )
        payload_low = VisionContextPayload(
            window_type="terminal",
            window_id="w2",
            space_id="s2",
            snapshot_id="snap2",
            summary="Low priority",
        )
        envelope_low = ContextEnvelope(metadata=metadata_low, payload=payload_low)

        await store.add(envelope_low)

        # Get by relevance
        query = ContextQuery().sort_by_relevance().limit(2)
        results = await store.query(query)

        assert len(results) == 2
        # High priority should come first
        assert results[0].metadata.priority == ContextPriority.HIGH

    @pytest.mark.asyncio
    async def test_clear_expired(self):
        """Test clearing expired contexts."""
        store = InMemoryContextStore()

        # Add short-lived envelope
        metadata = ContextMetadata()
        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test",
        )
        envelope = ContextEnvelope(metadata=metadata, payload=payload, ttl_seconds=1)

        await store.add(envelope)

        # Wait for expiry
        await asyncio.sleep(1.5)

        removed = await store.clear_expired()

        assert removed == 1
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when max size reached."""
        store = InMemoryContextStore(max_size=2)

        # Add 3 envelopes
        for i in range(3):
            metadata = ContextMetadata()
            payload = VisionContextPayload(
                window_type="terminal",
                window_id=f"w{i}",
                space_id=f"s{i}",
                snapshot_id=f"snap{i}",
                summary=f"Test {i}",
            )
            envelope = ContextEnvelope(metadata=metadata, payload=payload)
            await store.add(envelope)

        # Should only have 2 (oldest evicted)
        count = await store.count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_auto_cleanup(self):
        """Test auto-cleanup background task."""
        store = InMemoryContextStore(auto_cleanup_interval=1)

        # Add short-lived envelope
        metadata = ContextMetadata()
        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test",
        )
        envelope = ContextEnvelope(metadata=metadata, payload=payload, ttl_seconds=1)

        await store.add(envelope)

        # Start auto-cleanup
        await store.start_auto_cleanup()

        # Wait for cleanup to run
        await asyncio.sleep(2)

        # Should be cleaned up
        count = await store.count()
        assert count == 0

        # Stop cleanup
        await store.stop_auto_cleanup()

    @pytest.mark.asyncio
    async def test_get_stats(self, sample_envelope):
        """Test statistics retrieval."""
        store = InMemoryContextStore(max_size=100)

        await store.add(sample_envelope)

        stats = await store.get_stats()

        assert stats["total"] == 1
        assert stats["max_size"] == 100
        assert stats["utilization"] == 0.01
        assert "VISION" in stats["by_category"]
        assert "PENDING" in stats["by_state"]
        assert stats["avg_relevance"] > 0.0


class TestContextQuery:
    """Test ContextQuery builder."""

    def test_query_building(self):
        """Test fluent query building."""
        query = (
            ContextQuery()
            .with_category("VISION")
            .with_tag("terminal")
            .with_min_relevance(0.5)
            .sort_by_relevance()
            .limit(10)
        )

        # Query should have filters
        assert query._limit == 10
        assert query._sort_key is not None
        assert len(query._filters) == 3

    @pytest.mark.asyncio
    async def test_query_application(self):
        """Test applying query to contexts."""
        # Create test envelopes
        metadata1 = ContextMetadata(
            category=ContextCategory.VISION,
            tags=("terminal",),
        )
        payload1 = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test 1",
        )
        env1 = ContextEnvelope(metadata=metadata1, payload=payload1)

        metadata2 = ContextMetadata(
            category=ContextCategory.COMMAND,
            tags=("browser",),
        )
        payload2 = VisionContextPayload(
            window_type="browser",
            window_id="w2",
            space_id="s2",
            snapshot_id="snap2",
            summary="Test 2",
        )
        env2 = ContextEnvelope(metadata=metadata2, payload=payload2)

        contexts = [env1, env2]

        # Query for vision only
        query = ContextQuery().with_category("VISION")
        results = query.apply(contexts)

        assert len(results) == 1
        assert results[0].metadata.category == ContextCategory.VISION
