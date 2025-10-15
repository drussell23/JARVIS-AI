"""
Integration test for complete follow-up flow.
Tests end-to-end: intent detection → context retrieval → routing → response.
"""
import pytest
from backend.core.intent.adaptive_classifier import (
    AdaptiveIntentEngine,
    LexicalClassifier,
    WeightedVotingStrategy,
)
from backend.core.intent.intent_registry import create_default_registry
from backend.core.context.memory_store import InMemoryContextStore
from backend.core.routing.adaptive_router import (
    AdaptiveRouter,
    RouteMatcher,
    RouteConfig,
    BaseRouteHandler,
    RoutingResult,
)
from backend.core.models.context_envelope import (
    ContextEnvelope,
    ContextMetadata,
    ContextCategory,
    ContextPriority,
    VisionContextPayload,
)
from backend.core.telemetry.events import TelemetryManager, InMemoryEventSink, EventType


# Mock handler for testing
class MockFollowUpHandler(BaseRouteHandler):
    """Mock handler for follow-up intents."""

    def __init__(self):
        super().__init__(name="mock_followup", priority=80)
        self.calls = []

    async def handle(self, user_input, intent, context, extras):
        self.calls.append({
            "input": user_input,
            "intent": intent.primary_intent,
            "context_id": context.metadata.id if context else None,
        })

        if not context:
            return RoutingResult(
                success=False,
                response="No pending context",
            )

        return RoutingResult(
            success=True,
            response=f"Handled follow-up for {context.payload.window_type}",
        )


@pytest.fixture
def intent_engine():
    """Create intent engine with follow-up patterns."""
    registry = create_default_registry()
    patterns = registry.get_all_patterns()

    classifier = LexicalClassifier(
        name="lexical",
        patterns=patterns,
        priority=50,
    )

    return AdaptiveIntentEngine(
        classifiers=[classifier],
        strategy=WeightedVotingStrategy(min_confidence=0.5),
    )


@pytest.fixture
def context_store():
    """Create in-memory context store."""
    return InMemoryContextStore(max_size=100)


@pytest.fixture
def router_with_handler():
    """Create router with mock handler."""
    handler = MockFollowUpHandler()
    matcher = RouteMatcher()

    router = AdaptiveRouter(matcher=matcher)

    # Add follow-up route
    route = RouteConfig(
        intent_label="follow_up",
        handler=handler,
        requires_context=False,  # Allow testing both with/without
        min_confidence=0.5,
    )

    router.add_route(route)

    return router, handler


@pytest.fixture
def telemetry():
    """Create telemetry manager with in-memory sink."""
    manager = TelemetryManager()
    sink = InMemoryEventSink()
    manager.add_sink(sink)
    return manager, sink


class TestFollowUpIntegration:
    """Integration tests for follow-up handling."""

    @pytest.mark.asyncio
    async def test_complete_follow_up_flow(
        self, intent_engine, context_store, router_with_handler, telemetry
    ):
        """Test complete flow: create context → user says 'yes' → handler called."""
        router, handler = router_with_handler
        telemetry_manager, event_sink = telemetry

        # 1. Create pending vision context (simulating JARVIS asking a question)
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
            summary="Terminal with errors detected",
            ocr_text="ModuleNotFoundError: No module named 'requests'",
        )

        envelope = ContextEnvelope(metadata=metadata, payload=payload, ttl_seconds=60)

        context_id = await context_store.add(envelope)

        await telemetry_manager.track_context_created(
            context_id=context_id,
            category="VISION",
            priority="HIGH",
            ttl_seconds=60,
        )

        # 2. User responds with "yes"
        user_input = "yes"

        # Classify intent
        intent_result = await intent_engine.classify(user_input)

        assert intent_result.primary_intent == "follow_up"
        assert intent_result.confidence > 0.5

        await telemetry_manager.track_intent_detected(
            intent_label=intent_result.primary_intent,
            confidence=intent_result.confidence,
            classifiers=["lexical"],
        )

        # 3. Retrieve active context
        retrieved_context = await context_store.get(context_id)

        assert retrieved_context is not None

        # 4. Route to handler
        routing_result = await router.route(
            user_input=user_input,
            intent=intent_result,
            context=retrieved_context,
        )

        assert routing_result.success is True
        assert "terminal" in routing_result.response.lower()

        # 5. Verify handler was called
        assert len(handler.calls) == 1
        assert handler.calls[0]["intent"] == "follow_up"
        assert handler.calls[0]["context_id"] == context_id

        # 6. Verify telemetry
        events = event_sink.get_events()
        assert len(events) >= 2  # context_created + intent_detected

    @pytest.mark.asyncio
    async def test_follow_up_without_context(
        self, intent_engine, router_with_handler
    ):
        """Test follow-up when no pending context exists."""
        router, handler = router_with_handler

        user_input = "yes"

        intent_result = await intent_engine.classify(user_input)

        assert intent_result.primary_intent == "follow_up"

        # Route without context
        routing_result = await router.route(
            user_input=user_input,
            intent=intent_result,
            context=None,
        )

        assert routing_result.success is False
        assert "no pending context" in routing_result.response.lower()

    @pytest.mark.asyncio
    async def test_multiple_follow_up_types(self, intent_engine):
        """Test different follow-up response types."""
        test_cases = [
            ("yes", "follow_up"),
            ("no thanks", "follow_up"),
            ("tell me more", "follow_up"),
            ("show me", "follow_up"),
            ("what does it say", "follow_up"),
        ]

        for user_input, expected_intent in test_cases:
            result = await intent_engine.classify(user_input)
            assert result.primary_intent == expected_intent, (
                f"Failed for '{user_input}': got {result.primary_intent}"
            )

    @pytest.mark.asyncio
    async def test_context_expiry_handling(self, context_store, intent_engine, router_with_handler):
        """Test handling of expired context."""
        import asyncio

        router, handler = router_with_handler

        # Create short-lived context
        metadata = ContextMetadata(category=ContextCategory.VISION)
        payload = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="Test",
        )
        envelope = ContextEnvelope(metadata=metadata, payload=payload, ttl_seconds=1)

        context_id = await context_store.add(envelope)

        # Wait for expiry
        await asyncio.sleep(1.5)

        # Try to retrieve
        retrieved = await context_store.get(context_id)

        # Context should still exist but be invalid
        assert retrieved is not None
        assert not retrieved.is_valid()

        # Clean up expired
        removed = await context_store.clear_expired()
        assert removed == 1

    @pytest.mark.asyncio
    async def test_semantic_context_matching(self, context_store):
        """Test finding relevant context when user mentions window type."""
        # Add multiple contexts
        for i, window_type in enumerate(["terminal", "browser", "code"]):
            metadata = ContextMetadata(
                category=ContextCategory.VISION,
                tags=(window_type,),
            )
            payload = VisionContextPayload(
                window_type=window_type,
                window_id=f"w{i}",
                space_id=f"s{i}",
                snapshot_id=f"snap{i}",
                summary=f"Test {window_type}",
            )
            envelope = ContextEnvelope(metadata=metadata, payload=payload)
            await context_store.add(envelope)

        # Search for terminal contexts
        results = await context_store.get_by_tags("terminal")

        assert len(results) == 1
        assert results[0].payload.window_type == "terminal"

    @pytest.mark.asyncio
    async def test_relevance_based_retrieval(self, context_store):
        """Test retrieving most relevant context."""
        from backend.core.context.store_interface import ContextQuery

        # Add high priority context
        metadata_high = ContextMetadata(
            category=ContextCategory.VISION,
            priority=ContextPriority.CRITICAL,
        )
        payload_high = VisionContextPayload(
            window_type="terminal",
            window_id="w1",
            space_id="s1",
            snapshot_id="snap1",
            summary="High priority",
        )
        env_high = ContextEnvelope(metadata=metadata_high, payload=payload_high)

        # Add low priority context
        metadata_low = ContextMetadata(
            category=ContextCategory.VISION,
            priority=ContextPriority.LOW,
        )
        payload_low = VisionContextPayload(
            window_type="browser",
            window_id="w2",
            space_id="s2",
            snapshot_id="snap2",
            summary="Low priority",
        )
        env_low = ContextEnvelope(metadata=metadata_low, payload=payload_low)

        await context_store.add(env_high)
        await context_store.add(env_low)

        # Get most relevant
        most_relevant = await context_store.get_most_relevant(limit=1)

        assert len(most_relevant) == 1
        assert most_relevant[0].metadata.priority == ContextPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_telemetry_tracking(self, telemetry):
        """Test comprehensive telemetry tracking."""
        manager, sink = telemetry

        # Track various events
        await manager.track_context_created(
            context_id="ctx1",
            category="VISION",
            priority="HIGH",
            ttl_seconds=60,
        )

        await manager.track_intent_detected(
            intent_label="follow_up",
            confidence=0.92,
            classifiers=["lexical"],
        )

        await manager.track_followup_resolved(
            context_id="ctx1",
            window_type="terminal",
            response_type="affirmative",
            latency_ms=45.2,
        )

        # Verify all events recorded
        events = sink.get_events()
        assert len(events) == 3

        # Verify event types
        event_types = {e.event_type for e in events}
        assert EventType.CONTEXT_CREATED in event_types
        assert EventType.INTENT_DETECTED in event_types
        assert EventType.FOLLOWUP_RESOLVED in event_types

        # Verify event data
        intent_events = sink.get_events(event_type=EventType.INTENT_DETECTED)
        assert len(intent_events) == 1
        assert intent_events[0].properties["intent_label"] == "follow_up"
        assert intent_events[0].metrics["confidence"] == 0.92
