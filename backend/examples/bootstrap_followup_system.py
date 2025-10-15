"""
Complete Bootstrap Example: Context-Aware Follow-Up System

This example shows how to wire together all components:
- Intent classification
- Context storage
- Routing
- Vision handlers
- Telemetry

Usage:
    python -m backend.examples.bootstrap_followup_system
"""
import asyncio
import logging
from pathlib import Path

# Core components
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
    logging_middleware,
    context_validation_middleware,
)
from backend.vision.handlers.follow_up_plugin import VisionFollowUpPlugin
from backend.core.routing.adaptive_router import PluginRegistry
from backend.core.telemetry.events import init_telemetry, InMemoryEventSink, get_telemetry
from backend.core.models.context_envelope import (
    ContextEnvelope,
    ContextMetadata,
    ContextCategory,
    ContextPriority,
    VisionContextPayload,
    InteractionContextPayload,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class FollowUpSystem:
    """Orchestrates the complete follow-up system."""

    def __init__(self):
        # Initialize telemetry
        self.event_sink = InMemoryEventSink()
        self.telemetry = init_telemetry(
            sinks=[self.event_sink],
            session_id="demo_session",
        )

        # Initialize intent engine
        self.intent_engine = self._create_intent_engine()

        # Initialize context store
        self.context_store = InMemoryContextStore(max_size=1000)

        # Initialize router
        self.router = self._create_router()

        # Register plugins
        self.plugin_registry = PluginRegistry(self.router)

        logger.info("Follow-Up System initialized")

    def _create_intent_engine(self) -> AdaptiveIntentEngine:
        """Create and configure intent classification engine."""
        registry = create_default_registry()
        patterns = registry.get_all_patterns()

        # Create lexical classifier
        lexical = LexicalClassifier(
            name="lexical_primary",
            patterns=patterns,
            priority=50,
        )

        # TODO: Add semantic classifier if embeddings available
        # semantic = SemanticEmbeddingClassifier(...)

        engine = AdaptiveIntentEngine(
            classifiers=[lexical],
            strategy=WeightedVotingStrategy(
                source_weights={"lexical_primary": 1.0},
                min_confidence=0.6,
            ),
        )

        logger.info(f"Intent engine created with {engine.classifier_count} classifier(s)")
        return engine

    def _create_router(self) -> AdaptiveRouter:
        """Create and configure adaptive router."""
        matcher = RouteMatcher()
        router = AdaptiveRouter(matcher=matcher)

        # Add middleware
        router.use_middleware(logging_middleware)
        router.use_middleware(context_validation_middleware)

        logger.info("Router created with middleware")
        return router

    async def start(self):
        """Start the system."""
        # Start auto-cleanup for context store
        await self.context_store.start_auto_cleanup()

        # Register plugins
        await self._register_plugins()

        logger.info("Follow-Up System started")

    async def stop(self):
        """Stop the system."""
        await self.context_store.stop_auto_cleanup()
        logger.info("Follow-Up System stopped")

    async def _register_plugins(self):
        """Register handler plugins."""
        # Register vision follow-up plugin
        vision_plugin = VisionFollowUpPlugin()
        await self.plugin_registry.register_plugin("vision_followup", vision_plugin)

        logger.info(
            f"Registered {len(self.plugin_registry.plugin_names)} plugin(s): "
            f"{self.plugin_registry.plugin_names}"
        )

    async def track_pending_question(
        self,
        question_text: str,
        window_type: str,
        window_id: str,
        space_id: str,
        snapshot_id: str,
        summary: str,
        ocr_text: str | None = None,
    ) -> str:
        """
        Track a pending question after JARVIS asks the user something.
        Returns context ID.
        """
        # Create vision context
        metadata = ContextMetadata(
            category=ContextCategory.VISION,
            priority=ContextPriority.HIGH,
            source="jarvis_vision",
            tags=(window_type, "pending_question"),
        )

        payload = VisionContextPayload(
            window_type=window_type,
            window_id=window_id,
            space_id=space_id,
            snapshot_id=snapshot_id,
            summary=summary,
            ocr_text=ocr_text,
        )

        envelope = ContextEnvelope(
            metadata=metadata,
            payload=payload,
            ttl_seconds=120,  # 2 minutes
            decay_rate=0.01,  # 1% per second
        )

        context_id = await self.context_store.add(envelope)

        # Track telemetry
        await self.telemetry.track_context_created(
            context_id=context_id,
            category="VISION",
            priority="HIGH",
            ttl_seconds=120,
            window_type=window_type,
        )

        logger.info(
            f"Tracked pending question: '{question_text}' (context_id={context_id})"
        )

        return context_id

    async def process_user_input(self, user_input: str) -> str:
        """
        Process user input through the complete pipeline.

        Flow:
        1. Classify intent
        2. Retrieve most relevant context
        3. Route to handler
        4. Return response
        """
        logger.info(f"Processing input: '{user_input}'")

        # 1. Classify intent
        intent_result = await self.intent_engine.classify(user_input)

        await self.telemetry.track_intent_detected(
            intent_label=intent_result.primary_intent,
            confidence=intent_result.confidence,
            classifiers=["lexical_primary"],
        )

        logger.info(
            f"Intent: {intent_result.primary_intent} "
            f"(confidence={intent_result.confidence:.2f})"
        )

        # 2. Retrieve relevant context
        context = await self._get_relevant_context(user_input, intent_result)

        if context:
            logger.info(
                f"Found relevant context: {context.metadata.id} "
                f"(category={context.metadata.category.name}, "
                f"relevance={context.relevance_score():.2f})"
            )

        # 3. Route to handler
        routing_result = await self.router.route(
            user_input=user_input,
            intent=intent_result,
            context=context,
        )

        # 4. Mark context as consumed if used
        if context and routing_result.success:
            await self.context_store.mark_consumed(context.metadata.id)

        logger.info(
            f"Routing result: success={routing_result.success}, "
            f"response_length={len(routing_result.response)}"
        )

        return routing_result.response

    async def _get_relevant_context(self, user_input: str, intent_result):
        """Retrieve most relevant context for input."""
        from backend.core.context.store_interface import ContextQuery

        # For follow-up intents, get most recent valid context
        if intent_result.primary_intent == "follow_up":
            contexts = await self.context_store.get_most_relevant(limit=1)
            return contexts[0] if contexts else None

        # For vision intents, search by tags/category
        if intent_result.primary_intent == "vision":
            query = (
                ContextQuery()
                .with_category("VISION")
                .with_min_relevance(0.5)
                .sort_by_relevance()
                .limit(1)
            )
            contexts = await self.context_store.query(query)
            return contexts[0] if contexts else None

        return None

    async def get_stats(self) -> dict:
        """Get system statistics."""
        store_stats = await self.context_store.get_stats()
        telemetry_stats = {
            "total_events": self.event_sink.count,
        }

        return {
            "context_store": store_stats,
            "telemetry": telemetry_stats,
            "intent_engine": {
                "classifiers": self.intent_engine.classifier_count,
            },
            "router": {
                "plugins": len(self.plugin_registry.plugin_names),
            },
        }


async def demo():
    """Run a demo interaction."""
    system = FollowUpSystem()
    await system.start()

    try:
        # Simulate JARVIS detecting a terminal and asking a question
        print("\n=== Scenario 1: Terminal Error Detection ===\n")

        question = "I can see your Terminal. Would you like me to describe what's displayed?"
        print(f"JARVIS: {question}")

        context_id = await system.track_pending_question(
            question_text=question,
            window_type="terminal",
            window_id="terminal_1",
            space_id="space_1",
            snapshot_id="snap_12345",
            summary="Terminal window with Python error",
            ocr_text="Traceback (most recent call last):\n"
            '  File "app.py", line 42, in <module>\n'
            "    import requests\n"
            "ModuleNotFoundError: No module named 'requests'",
        )

        # User responds
        user_response = "yes"
        print(f"\nYou: {user_response}")

        response = await system.process_user_input(user_response)
        print(f"\nJARVIS: {response}\n")

        # Scenario 2: Follow-up inquiry
        print("\n=== Scenario 2: Negative Response ===\n")

        question2 = "Would you like me to help fix this error?"
        print(f"JARVIS: {question2}")

        await system.track_pending_question(
            question_text=question2,
            window_type="terminal",
            window_id="terminal_1",
            space_id="space_1",
            snapshot_id="snap_12346",
            summary="Offering to fix Python import error",
        )

        user_response2 = "no thanks"
        print(f"\nYou: {user_response2}")

        response2 = await system.process_user_input(user_response2)
        print(f"\nJARVIS: {response2}\n")

        # Show stats
        print("\n=== System Statistics ===\n")
        stats = await system.get_stats()
        print(f"Active contexts: {stats['context_store']['total']}")
        print(f"Total events tracked: {stats['telemetry']['total_events']}")
        print(f"Registered plugins: {stats['router']['plugins']}")

    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(demo())
