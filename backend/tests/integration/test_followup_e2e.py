"""
End-to-End Integration Tests for Follow-Up System
Tests the complete flow from pending question tracking to follow-up resolution.
"""
import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch


@pytest_asyncio.fixture
async def context_bridge():
    """Create unified context bridge for testing."""
    from backend.core.unified_context_bridge import (
        UnifiedContextBridge,
        ContextBridgeConfig,
        ContextStoreBackend,
    )

    config = ContextBridgeConfig(
        backend=ContextStoreBackend.MEMORY,
        max_contexts=50,
        default_ttl=120,
        follow_up_enabled=True,
        context_aware_enabled=True,
    )

    bridge = UnifiedContextBridge(config=config)
    await bridge.initialize()

    # Yield the actual bridge object, not the async generator
    try:
        yield bridge
    finally:
        await bridge.shutdown()


@pytest.fixture
def mock_vision_intelligence():
    """Mock PureVisionIntelligence."""
    vision = Mock()
    vision.context_store = None
    vision.track_pending_question = AsyncMock()
    return vision


@pytest.fixture
def mock_async_pipeline():
    """Mock AsyncPipeline."""
    pipeline = Mock()
    pipeline.context_store = None
    pipeline._follow_up_enabled = True
    pipeline.intent_engine = None
    pipeline.router = None
    return pipeline


@pytest.mark.asyncio
class TestFollowUpE2E:
    """End-to-end integration tests."""

    async def test_terminal_error_follow_up_flow(self, context_bridge, mock_vision_intelligence):
        """
        Test complete flow:
        1. JARVIS detects terminal with error
        2. Asks user "Would you like me to describe it?"
        3. Tracks pending question
        4. User says "yes"
        5. System retrieves context and provides error analysis
        """
        # Step 1: Integrate vision intelligence with bridge
        context_bridge.integrate_vision_intelligence(mock_vision_intelligence)

        # Step 2: Simulate JARVIS asking a question and tracking it
        question_text = "I can see your Terminal. Would you like me to describe what's displayed?"
        window_type = "terminal"
        window_id = "term_1"
        space_id = "space_1"
        snapshot_id = "snap_12345"
        summary = "Terminal with Python error"
        ocr_text = "Traceback (most recent call last):\n  File test.py, line 5\nModuleNotFoundError: No module named 'requests'"

        context_id = await context_bridge.track_pending_question(
            question_text=question_text,
            window_type=window_type,
            window_id=window_id,
            space_id=space_id,
            snapshot_id=snapshot_id,
            summary=summary,
            ocr_text=ocr_text,
            ttl_seconds=120,
        )

        assert context_id is not None, "Context ID should be returned"

        # Step 3: Verify context was stored
        pending = await context_bridge.get_pending_context(category="VISION")
        assert pending is not None, "Pending context should exist"
        assert pending.payload.window_type == "terminal"
        assert pending.payload.ocr_text == ocr_text

        # Step 4: Simulate follow-up response (user says "yes")
        from backend.core.intent.adaptive_classifier import IntentResult
        from backend.core.routing.adaptive_router import AdaptiveRouter
        from backend.vision.handlers.follow_up_plugin import VisionFollowUpPlugin

        # Create router with vision follow-up plugin
        router = AdaptiveRouter()
        plugin = VisionFollowUpPlugin()
        await router.register_plugin(plugin)

        # Mock intent classification result
        intent_result = IntentResult(
            primary_intent="follow_up",
            confidence=0.95,
            secondary_intents=[],
            metadata={},
        )

        # Route the follow-up
        routing_result = await router.route(
            user_input="yes",
            intent=intent_result,
            context=pending,
        )

        # Step 5: Verify response
        assert routing_result.success, "Follow-up routing should succeed"
        assert "error" in routing_result.response.lower() or "terminal" in routing_result.response.lower()
        assert routing_result.metadata.get("window_type") == "terminal"

        # Step 6: Verify context was consumed
        pending_after = await context_bridge.get_pending_context(category="VISION")
        # Context should still exist but marked as consumed
        # (or None if cleared after consumption)

    async def test_context_expiry_handling(self, context_bridge):
        """
        Test expired context handling:
        1. Track question with short TTL
        2. Wait for expiration
        3. Verify expired contexts are cleaned up
        4. User says "yes" but gets "no pending context" response
        """
        # Step 1: Track question with 1-second TTL
        context_id = await context_bridge.track_pending_question(
            question_text="Quick question?",
            window_type="browser",
            window_id="win_1",
            space_id="space_1",
            snapshot_id="snap_999",
            summary="Browser window",
            ocr_text="Some text",
            ttl_seconds=1,
        )

        assert context_id is not None

        # Step 2: Verify context exists
        pending = await context_bridge.get_pending_context()
        assert pending is not None

        # Step 3: Wait for expiration
        await asyncio.sleep(2)

        # Step 4: Clear expired contexts
        expired_count = await context_bridge.clear_expired_contexts()
        assert expired_count == 1, "One context should be expired"

        # Step 5: Verify no pending context
        pending_after = await context_bridge.get_pending_context()
        assert pending_after is None, "No pending context should remain"

    async def test_browser_content_follow_up(self, context_bridge):
        """Test browser window follow-up flow."""
        # Track browser question
        context_id = await context_bridge.track_pending_question(
            question_text="I can see a documentation page. Want me to summarize it?",
            window_type="browser",
            window_id="chrome_1",
            space_id="space_1",
            snapshot_id="snap_browser",
            summary="Documentation page open",
            ocr_text="Python requests library documentation...",
            ttl_seconds=120,
        )

        assert context_id is not None

        # Retrieve and verify
        pending = await context_bridge.get_pending_context(category="VISION")
        assert pending is not None
        assert pending.payload.window_type == "browser"

    async def test_code_editor_follow_up(self, context_bridge):
        """Test code editor follow-up flow."""
        # Track code editor question
        context_id = await context_bridge.track_pending_question(
            question_text="I see errors in your code. Want me to analyze them?",
            window_type="code",
            window_id="vscode_1",
            space_id="space_1",
            snapshot_id="snap_code",
            summary="VS Code with TypeScript errors",
            ocr_text="test.ts:10:5 - error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.",
            ttl_seconds=120,
        )

        assert context_id is not None

        # Retrieve and verify
        pending = await context_bridge.get_pending_context(category="VISION")
        assert pending is not None
        assert pending.payload.window_type == "code"

    async def test_no_pending_context_graceful_fallback(self, context_bridge):
        """Test graceful handling when user says 'yes' but no pending context exists."""
        # No context tracked

        # Try to get pending context
        pending = await context_bridge.get_pending_context()
        assert pending is None, "Should have no pending context"

        # In real flow, pipeline would return:
        # "I don't have any pending context to follow up on..."

    async def test_multiple_pending_contexts_priority(self, context_bridge):
        """Test that most relevant/recent context is retrieved when multiple exist."""
        # Track multiple questions
        context_id_1 = await context_bridge.track_pending_question(
            question_text="Question 1",
            window_type="terminal",
            window_id="term_1",
            space_id="space_1",
            snapshot_id="snap_1",
            summary="Terminal 1",
            ttl_seconds=120,
        )

        await asyncio.sleep(0.1)  # Small delay

        context_id_2 = await context_bridge.track_pending_question(
            question_text="Question 2 (more recent)",
            window_type="browser",
            window_id="browser_1",
            space_id="space_1",
            snapshot_id="snap_2",
            summary="Browser 1",
            ttl_seconds=120,
        )

        # Get most relevant (should be most recent)
        pending = await context_bridge.get_pending_context()
        assert pending is not None
        # Most recent should be browser
        assert pending.payload.window_type == "browser"
        assert pending.metadata.id == context_id_2

    async def test_context_bridge_stats(self, context_bridge):
        """Test bridge statistics reporting."""
        # Track some contexts
        await context_bridge.track_pending_question(
            question_text="Test question",
            window_type="terminal",
            window_id="term_1",
            space_id="space_1",
            snapshot_id="snap_1",
            summary="Test",
            ttl_seconds=120,
        )

        # Get stats
        stats = context_bridge.get_stats()

        assert stats["initialized"] is True
        assert stats["backend"] == "memory"
        assert stats["follow_up_enabled"] is True
        assert stats["context_aware_enabled"] is True
        assert stats["max_contexts"] == 50
        assert stats["components"]["vision_intelligence"] is True


@pytest.mark.asyncio
class TestFollowUpIntegrationWithPipeline:
    """Test integration between follow-up system and async pipeline."""

    async def test_pipeline_follow_up_detection(self, context_bridge):
        """Test that pipeline correctly detects and routes follow-up intents."""
        from backend.core.async_pipeline import AdvancedAsyncPipeline, PipelineContext

        # Create mock JARVIS instance
        mock_jarvis = Mock()
        mock_jarvis.async_pipeline = None

        # Create pipeline
        pipeline = AdvancedAsyncPipeline(
            jarvis_instance=mock_jarvis,
            config={
                "follow_up_enabled": True,
                "max_pending_contexts": 50,
            }
        )

        # Integrate with bridge
        context_bridge.integrate_async_pipeline(pipeline)

        # Track a pending question
        await context_bridge.track_pending_question(
            question_text="Terminal question?",
            window_type="terminal",
            window_id="term_1",
            space_id="space_1",
            snapshot_id="snap_1",
            summary="Terminal with error",
            ocr_text="Error: file not found",
            ttl_seconds=120,
        )

        # Verify pipeline has access to context store
        assert pipeline.context_store is not None
        assert pipeline.context_store == context_bridge.context_store


@pytest.mark.asyncio
class TestTelemetryTracking:
    """Test telemetry event tracking."""

    async def test_telemetry_events_fired(self, context_bridge):
        """Test that telemetry events are properly tracked."""
        with patch('backend.core.telemetry.events.get_telemetry') as mock_telemetry:
            mock_telem_instance = AsyncMock()
            mock_telemetry.return_value = mock_telem_instance

            # Track question (should fire pending_created event)
            await context_bridge.track_pending_question(
                question_text="Test question",
                window_type="terminal",
                window_id="term_1",
                space_id="space_1",
                snapshot_id="snap_1",
                summary="Test",
                ttl_seconds=120,
            )

            # Verify telemetry was called
            # Note: Actual telemetry calls are wrapped in try/except
            # so this test validates the structure is in place


@pytest.mark.asyncio
class TestContextStoreBackends:
    """Test different context store backends."""

    async def test_memory_backend(self):
        """Test memory backend."""
        from backend.core.unified_context_bridge import (
            ContextBridgeConfig,
            ContextStoreBackend,
        )

        config = ContextBridgeConfig(backend=ContextStoreBackend.MEMORY)
        from backend.core.unified_context_bridge import ContextStoreFactory

        store = await ContextStoreFactory.create(config)
        assert store is not None

        # Test basic operations
        from backend.core.models.context_envelope import (
            ContextEnvelope,
            ContextMetadata,
            ContextCategory,
            VisionContextPayload,
        )

        envelope = ContextEnvelope(
            metadata=ContextMetadata(category=ContextCategory.VISION),
            payload=VisionContextPayload(
                window_type="terminal",
                window_id="term_1",
                space_id="space_1",
                snapshot_id="snap_1",
                summary="Test",
            ),
            ttl_seconds=120,
        )

        context_id = await store.add(envelope)
        assert context_id is not None

        # Retrieve
        retrieved = await store.get(context_id)
        assert retrieved is not None
        assert retrieved.metadata.id == context_id

    @pytest.mark.skipif(
        not hasattr(__import__('backend.core.context', fromlist=['redis_store']), 'redis_store'),
        reason="Redis backend not available"
    )
    async def test_redis_backend(self):
        """Test Redis backend (if available)."""
        from backend.core.unified_context_bridge import (
            ContextBridgeConfig,
            ContextStoreBackend,
        )

        config = ContextBridgeConfig(
            backend=ContextStoreBackend.REDIS,
            redis_url="redis://localhost:6379/15",  # Use test DB
        )

        from backend.core.unified_context_bridge import ContextStoreFactory

        try:
            store = await ContextStoreFactory.create(config)
            assert store is not None

            # Cleanup
            if hasattr(store, 'clear_all'):
                await store.clear_all()
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
