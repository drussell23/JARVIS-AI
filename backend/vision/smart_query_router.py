"""
Smart Query Router for JARVIS Vision System
Routes queries to optimal processing pipeline based on classification.

Three routing paths:
- METADATA_ONLY → Yabai CLI only (fast, no screenshots)
- VISUAL_ANALYSIS → Current screen + Claude Vision
- DEEP_ANALYSIS → Multi-space capture + Yabai + Claude Vision
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime

from .intelligent_query_classifier import (
    QueryIntent,
    ClassificationResult,
    get_query_classifier
)
from .adaptive_learning_system import get_learning_system

# Import A/B testing framework
try:
    from .ab_testing_framework import get_ab_test, ABTestingFramework
    ab_testing_available = True
except ImportError:
    ab_testing_available = False

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of query routing"""
    intent: QueryIntent
    response: str
    latency_ms: float
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class SmartQueryRouter:
    """
    Intelligent router that directs queries to optimal processing pipeline.
    Uses confidence thresholds and hybrid fallback strategies.
    """

    def __init__(
        self,
        yabai_handler: Optional[Callable] = None,
        vision_handler: Optional[Callable] = None,
        multi_space_handler: Optional[Callable] = None,
        claude_client=None,
        enable_ab_testing: bool = False
    ):
        """
        Initialize smart router

        Args:
            yabai_handler: Handler for metadata-only queries
            vision_handler: Handler for visual analysis queries
            multi_space_handler: Handler for deep analysis queries
            claude_client: Claude API client for classification
            enable_ab_testing: Enable A/B testing for classification
        """
        self.yabai_handler = yabai_handler
        self.vision_handler = vision_handler
        self.multi_space_handler = multi_space_handler

        # Get classifier and learning system
        self.classifier = get_query_classifier(claude_client)
        self.learning_system = get_learning_system()

        # A/B testing
        self.enable_ab_testing = enable_ab_testing and ab_testing_available
        self.ab_test: Optional[ABTestingFramework] = None
        if self.enable_ab_testing:
            self.ab_test = get_ab_test("classification_variants")
            logger.info("[ROUTER] A/B testing enabled")

        # Confidence thresholds
        self.high_confidence_threshold = 0.85  # Direct routing
        self.medium_confidence_threshold = 0.70  # Route with logging
        self.low_confidence_threshold = 0.60  # Hybrid approach

        # Performance tracking
        self._routing_stats = {
            'metadata_only': 0,
            'visual_analysis': 0,
            'deep_analysis': 0,
            'hybrid_fallback': 0,
            'total_latency_ms': 0,
            'total_queries': 0,
            'ab_test_queries': 0
        }

        logger.info("[ROUTER] Smart query router initialized")

    async def route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_feedback_callback: Optional[Callable] = None
    ) -> RoutingResult:
        """
        Route a query to the optimal processing pipeline

        Args:
            query: User's query
            context: Optional context (active space, recent queries, etc.)
            user_feedback_callback: Optional callback for user feedback prompts

        Returns:
            RoutingResult with response and metadata
        """
        start_time = time.time()

        # Classify the query
        classification = await self.classifier.classify_query(query, context)

        # Record classification for learning
        await self.learning_system.record_classification(query, classification, context)

        logger.info(
            f"[ROUTER] Query classified as {classification.intent.value} "
            f"(confidence: {classification.confidence:.2f})"
        )

        # Route based on confidence and intent
        if classification.confidence >= self.high_confidence_threshold:
            # High confidence - direct routing
            result = await self._route_direct(query, classification, context)

        elif classification.confidence >= self.medium_confidence_threshold:
            # Medium confidence - route but log for review
            result = await self._route_with_monitoring(query, classification, context)

        elif classification.confidence >= self.low_confidence_threshold:
            # Low confidence - use hybrid approach
            result = await self._route_hybrid(query, classification, context, user_feedback_callback)

        else:
            # Very low confidence - ask user or default to visual
            result = await self._route_with_user_input(query, classification, context, user_feedback_callback)

        # Track latency
        total_latency = (time.time() - start_time) * 1000
        result.latency_ms = total_latency

        # Update stats
        self._routing_stats['total_queries'] += 1
        self._routing_stats['total_latency_ms'] += total_latency
        self._routing_stats[classification.intent.value] += 1

        logger.info(
            f"[ROUTER] Query routed successfully "
            f"(intent: {result.intent.value}, latency: {result.latency_ms:.1f}ms)"
        )

        return result

    async def _route_direct(
        self,
        query: str,
        classification: ClassificationResult,
        context: Optional[Dict[str, Any]]
    ) -> RoutingResult:
        """Direct routing with high confidence"""
        intent = classification.intent

        try:
            if intent == QueryIntent.METADATA_ONLY:
                # Yabai-only path - fast metadata query
                response, metadata = await self._handle_metadata_query(query, context)

            elif intent == QueryIntent.VISUAL_ANALYSIS:
                # Current screen + Claude vision
                response, metadata = await self._handle_visual_query(query, context)

            elif intent == QueryIntent.DEEP_ANALYSIS:
                # Multi-space comprehensive analysis
                response, metadata = await self._handle_deep_analysis(query, context)

            else:
                raise ValueError(f"Unknown intent: {intent}")

            return RoutingResult(
                intent=intent,
                response=response,
                latency_ms=0,  # Will be set by caller
                metadata={
                    **metadata,
                    'classification': classification.__dict__,
                    'routing_strategy': 'direct',
                    'confidence': classification.confidence
                },
                success=True
            )

        except Exception as e:
            logger.error(f"[ROUTER] Direct routing failed: {e}", exc_info=True)
            return RoutingResult(
                intent=intent,
                response=f"I encountered an error processing your request: {str(e)}",
                latency_ms=0,
                metadata={'error': str(e), 'classification': classification.__dict__},
                success=False,
                error=str(e)
            )

    async def _route_with_monitoring(
        self,
        query: str,
        classification: ClassificationResult,
        context: Optional[Dict[str, Any]]
    ) -> RoutingResult:
        """Route with medium confidence, log for review"""
        logger.info(
            f"[ROUTER] Medium confidence routing - logging for review "
            f"({classification.intent.value}, {classification.confidence:.2f})"
        )

        # Route normally but flag for review
        result = await self._route_direct(query, classification, context)
        result.metadata['needs_review'] = True
        result.metadata['routing_strategy'] = 'monitored'

        return result

    async def _route_hybrid(
        self,
        query: str,
        classification: ClassificationResult,
        context: Optional[Dict[str, Any]],
        feedback_callback: Optional[Callable]
    ) -> RoutingResult:
        """
        Hybrid routing for low confidence.
        Try cheaper approach first, offer upgrade if insufficient.
        """
        logger.info(
            f"[ROUTER] Low confidence - using hybrid approach "
            f"({classification.intent.value}, {classification.confidence:.2f})"
        )

        self._routing_stats['hybrid_fallback'] += 1

        try:
            # Step 1: Always try metadata-only first (cheap)
            metadata_response, metadata = await self._handle_metadata_query(query, context)

            # Step 2: Check if metadata response is sufficient
            # (In a real system, Claude could evaluate this)
            # For now, we'll offer the user an option to see more detail

            if feedback_callback:
                # Ask user if they want visual analysis
                prompt = f"{metadata_response}\n\nWould you like me to analyze screenshots for more detail?"
                user_wants_visual = await feedback_callback(prompt)

                if user_wants_visual:
                    # User wants more detail - upgrade to visual analysis
                    logger.info("[ROUTER] User requested visual analysis upgrade")

                    if classification.intent == QueryIntent.DEEP_ANALYSIS:
                        # Go straight to deep analysis
                        visual_response, visual_metadata = await self._handle_deep_analysis(query, context)
                    else:
                        # Just current screen
                        visual_response, visual_metadata = await self._handle_visual_query(query, context)

                    # Record feedback: actual intent was visual/deep, not metadata
                    actual_intent = (
                        QueryIntent.DEEP_ANALYSIS
                        if classification.intent == QueryIntent.DEEP_ANALYSIS
                        else QueryIntent.VISUAL_ANALYSIS
                    )

                    await self.learning_system.record_feedback(
                        query=query,
                        classified_intent=classification.intent,
                        actual_intent=actual_intent,
                        confidence=classification.confidence,
                        reasoning="User requested visual analysis after metadata",
                        user_satisfied=True,
                        context=context
                    )

                    return RoutingResult(
                        intent=actual_intent,
                        response=visual_response,
                        latency_ms=0,
                        metadata={
                            **visual_metadata,
                            'routing_strategy': 'hybrid_upgraded',
                            'initial_intent': classification.intent.value
                        },
                        success=True
                    )
                else:
                    # User satisfied with metadata
                    await self.learning_system.record_feedback(
                        query=query,
                        classified_intent=classification.intent,
                        actual_intent=QueryIntent.METADATA_ONLY,
                        confidence=0.85,
                        reasoning="User accepted metadata-only response",
                        user_satisfied=True,
                        context=context
                    )

            # Return metadata response
            return RoutingResult(
                intent=QueryIntent.METADATA_ONLY,
                response=metadata_response,
                latency_ms=0,
                metadata={
                    **metadata,
                    'routing_strategy': 'hybrid',
                    'offered_upgrade': feedback_callback is not None
                },
                success=True
            )

        except Exception as e:
            logger.error(f"[ROUTER] Hybrid routing failed: {e}", exc_info=True)
            # Fallback to visual analysis
            return await self._route_direct(
                query,
                ClassificationResult(
                    intent=QueryIntent.VISUAL_ANALYSIS,
                    confidence=0.7,
                    reasoning="Fallback after hybrid routing failure",
                    features={}
                ),
                context
            )

    async def _route_with_user_input(
        self,
        query: str,
        classification: ClassificationResult,
        context: Optional[Dict[str, Any]],
        feedback_callback: Optional[Callable]
    ) -> RoutingResult:
        """Route with user input when confidence is very low"""
        logger.warning(
            f"[ROUTER] Very low confidence ({classification.confidence:.2f}) - "
            "requesting user input or defaulting to visual"
        )

        # Default to visual analysis if no feedback mechanism
        if not feedback_callback:
            return await self._route_direct(
                query,
                ClassificationResult(
                    intent=QueryIntent.VISUAL_ANALYSIS,
                    confidence=0.7,
                    reasoning="Default to visual analysis (no feedback available)",
                    features={}
                ),
                context
            )

        # Ask user how they want the query handled
        prompt = (
            "I'm not sure how to best answer that. Would you like:\n"
            "1. Quick overview (metadata only)\n"
            "2. Current screen analysis\n"
            "3. Comprehensive multi-space analysis"
        )

        # This would need a more sophisticated feedback mechanism in practice
        # For now, default to visual
        return await self._route_direct(
            query,
            ClassificationResult(
                intent=QueryIntent.VISUAL_ANALYSIS,
                confidence=0.7,
                reasoning="Low confidence fallback",
                features={}
            ),
            context
        )

    async def _handle_metadata_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle metadata-only query with Yabai"""
        if not self.yabai_handler:
            raise ValueError("Yabai handler not configured")

        logger.info("[ROUTER] Executing metadata-only query (Yabai)")
        response = await self.yabai_handler(query, context)

        metadata = {
            'handler': 'yabai',
            'uses_screenshots': False,
            'expected_latency': '<100ms'
        }

        return response, metadata

    async def _handle_visual_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle visual analysis query with current screen"""
        if not self.vision_handler:
            raise ValueError("Vision handler not configured")

        logger.info("[ROUTER] Executing visual analysis query (Current screen)")
        response = await self.vision_handler(query, context, multi_space=False)

        metadata = {
            'handler': 'vision',
            'uses_screenshots': True,
            'screenshot_count': 1,
            'expected_latency': '1-3s'
        }

        return response, metadata

    async def _handle_deep_analysis(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle deep analysis query with multi-space capture"""
        if not self.multi_space_handler:
            # Fallback to vision handler if no multi-space handler
            logger.warning("[ROUTER] No multi-space handler, falling back to vision")
            return await self._handle_visual_query(query, context)

        logger.info("[ROUTER] Executing deep analysis query (Multi-space)")
        response = await self.multi_space_handler(query, context)

        metadata = {
            'handler': 'multi_space',
            'uses_screenshots': True,
            'screenshot_count': 'multiple',
            'expected_latency': '3-10s'
        }

        return response, metadata

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        stats = dict(self._routing_stats)

        if stats['total_queries'] > 0:
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['total_queries']

            # Calculate distribution
            total = stats['total_queries']
            stats['distribution'] = {
                'metadata_only_pct': (stats['metadata_only'] / total) * 100,
                'visual_analysis_pct': (stats['visual_analysis'] / total) * 100,
                'deep_analysis_pct': (stats['deep_analysis'] / total) * 100,
                'hybrid_fallback_pct': (stats['hybrid_fallback'] / total) * 100,
            }

        # Add classifier stats
        stats['classifier_performance'] = self.classifier.get_performance_stats()

        # Add learning stats
        stats['learning_metrics'] = self.learning_system.get_accuracy_report()

        return stats

    def update_handlers(
        self,
        yabai_handler: Optional[Callable] = None,
        vision_handler: Optional[Callable] = None,
        multi_space_handler: Optional[Callable] = None
    ):
        """Update handler callbacks"""
        if yabai_handler:
            self.yabai_handler = yabai_handler
        if vision_handler:
            self.vision_handler = vision_handler
        if multi_space_handler:
            self.multi_space_handler = multi_space_handler

        logger.info("[ROUTER] Handlers updated")

    def enable_ab_test(
        self,
        variant_a_name: str,
        variant_a_classifier: Callable,
        variant_b_name: str,
        variant_b_classifier: Callable,
        traffic_split: float = 0.5
    ):
        """
        Enable A/B testing with two classifier variants

        Args:
            variant_a_name: Name for variant A (control)
            variant_a_classifier: Classifier function for variant A
            variant_b_name: Name for variant B (test)
            variant_b_classifier: Classifier function for variant B
            traffic_split: Traffic allocation for variant A (0.0-1.0)
        """
        if not ab_testing_available:
            logger.warning("[ROUTER] A/B testing not available")
            return

        self.enable_ab_testing = True
        self.ab_test = get_ab_test("classification_variants")

        # Add variants
        self.ab_test.add_variant(
            variant_id="variant_a",
            name=variant_a_name,
            description="Control variant",
            classifier_func=variant_a_classifier,
            traffic_allocation=traffic_split,
            is_control=True
        )

        self.ab_test.add_variant(
            variant_id="variant_b",
            name=variant_b_name,
            description="Test variant",
            classifier_func=variant_b_classifier,
            traffic_allocation=1.0 - traffic_split
        )

        logger.info(
            f"[ROUTER] A/B test configured: {variant_a_name} ({traffic_split*100:.0f}%) vs "
            f"{variant_b_name} ({(1-traffic_split)*100:.0f}%)"
        )

    def get_ab_test_report(self) -> Optional[Dict[str, Any]]:
        """Get A/B test report"""
        if not self.ab_test:
            return None

        return self.ab_test.get_report()


# Singleton instance
_router_instance: Optional[SmartQueryRouter] = None


def get_smart_router(
    yabai_handler: Optional[Callable] = None,
    vision_handler: Optional[Callable] = None,
    multi_space_handler: Optional[Callable] = None,
    claude_client=None
) -> SmartQueryRouter:
    """Get or create the singleton smart router"""
    global _router_instance

    if _router_instance is None:
        _router_instance = SmartQueryRouter(
            yabai_handler=yabai_handler,
            vision_handler=vision_handler,
            multi_space_handler=multi_space_handler,
            claude_client=claude_client
        )
    else:
        # Update handlers if provided
        if any([yabai_handler, vision_handler, multi_space_handler]):
            _router_instance.update_handlers(
                yabai_handler=yabai_handler,
                vision_handler=vision_handler,
                multi_space_handler=multi_space_handler
            )

    return _router_instance
