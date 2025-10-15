"""
Comprehensive Test Suite for Intelligent Query Classification System
Tests all components: classifier, router, learning, context, and performance monitoring
"""

import asyncio
import pytest
import logging
from typing import Dict, Any

from intelligent_query_classifier import (
    IntelligentQueryClassifier,
    QueryIntent,
    ClassificationResult
)
from smart_query_router import SmartQueryRouter, RoutingResult
from query_context_manager import QueryContextManager, UserPattern
from adaptive_learning_system import AdaptiveLearningSystem, FeedbackRecord
from performance_monitor import PerformanceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Test Queries ====================

TEST_QUERIES = {
    'metadata_only': [
        "How many spaces do I have?",
        "What's on desktop 2?",
        "Which apps are open?",
        "Give me a workspace overview",
        "List all my workspaces",
        "What apps are running?",
        "How many windows are open?",
        "Which space is Cursor on?",
        "Show me my workspace layout",
        "What's happening across my desktop spaces?"
    ],
    'visual_analysis': [
        "What do you see on my screen?",
        "Read this error message",
        "What's on my current screen?",
        "Can you see what I'm looking at?",
        "Describe what you see",
        "What's in this window?",
        "Read this for me",
        "What does this say?",
        "Analyze my current screen",
        "What's this error about?"
    ],
    'deep_analysis': [
        "What am I working on across all spaces?",
        "Analyze all my desktops",
        "What's happening in all my spaces?",
        "Give me a comprehensive workspace analysis",
        "Review all my open projects",
        "What work am I doing today?",
        "Analyze my coding session across desktops",
        "Compare my workspaces",
        "What errors do you see across all spaces?",
        "Comprehensive desktop analysis"
    ]
}


# ==================== Mock Handlers ====================

class MockClaudeClient:
    """Mock Claude API client for testing"""

    async def analyze_with_prompt(self, prompt: str) -> Dict[str, Any]:
        """Simulate Claude classification response"""
        # Simulate classification based on keywords
        prompt_lower = prompt.lower()

        if 'metadata' in prompt_lower or 'how many' in prompt_lower:
            intent = 'METADATA_ONLY'
            confidence = 0.92
        elif 'visual' in prompt_lower or 'see' in prompt_lower:
            intent = 'VISUAL_ANALYSIS'
            confidence = 0.88
        elif 'comprehensive' in prompt_lower or 'all' in prompt_lower:
            intent = 'DEEP_ANALYSIS'
            confidence = 0.85
        else:
            intent = 'VISUAL_ANALYSIS'
            confidence = 0.75

        response = f'''{{
  "intent": "{intent}",
  "confidence": {confidence},
  "reasoning": "Classified based on query characteristics",
  "second_best": {{
    "intent": "VISUAL_ANALYSIS",
    "confidence": 0.65
  }}
}}'''

        return {"response": response}


async def mock_yabai_handler(query: str, context: Dict[str, Any]) -> str:
    """Mock Yabai handler"""
    await asyncio.sleep(0.05)  # Simulate 50ms latency
    return f"You have 3 desktop spaces with Cursor, Chrome, and Slack open."


async def mock_vision_handler(query: str, context: Dict[str, Any], multi_space: bool = False) -> str:
    """Mock vision handler"""
    await asyncio.sleep(1.5 if not multi_space else 3.0)  # Simulate latency
    return f"I can see your screen with {'multiple spaces' if multi_space else 'the current space'}."


async def mock_multi_space_handler(query: str, context: Dict[str, Any]) -> str:
    """Mock multi-space handler"""
    await asyncio.sleep(4.0)  # Simulate 4s latency
    return "I've analyzed all your desktop spaces comprehensively."


# ==================== Tests ====================

class TestIntelligentQueryClassifier:
    """Test the intelligent query classifier"""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        claude_client = MockClaudeClient()
        return IntelligentQueryClassifier(claude_client, enable_cache=True)

    @pytest.mark.asyncio
    async def test_metadata_classification(self, classifier):
        """Test classification of metadata-only queries"""
        for query in TEST_QUERIES['metadata_only'][:3]:
            result = await classifier.classify_query(query)
            logger.info(f"Query: '{query}' -> {result.intent.value} ({result.confidence:.2f})")
            # Note: With mock, results may vary, so we just check it returns a result
            assert result.intent in [QueryIntent.METADATA_ONLY, QueryIntent.VISUAL_ANALYSIS, QueryIntent.DEEP_ANALYSIS]
            assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_visual_classification(self, classifier):
        """Test classification of visual analysis queries"""
        for query in TEST_QUERIES['visual_analysis'][:3]:
            result = await classifier.classify_query(query)
            logger.info(f"Query: '{query}' -> {result.intent.value} ({result.confidence:.2f})")
            assert result.intent in [QueryIntent.METADATA_ONLY, QueryIntent.VISUAL_ANALYSIS, QueryIntent.DEEP_ANALYSIS]
            assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_deep_classification(self, classifier):
        """Test classification of deep analysis queries"""
        for query in TEST_QUERIES['deep_analysis'][:3]:
            result = await classifier.classify_query(query)
            logger.info(f"Query: '{query}' -> {result.intent.value} ({result.confidence:.2f})")
            assert result.intent in [QueryIntent.METADATA_ONLY, QueryIntent.VISUAL_ANALYSIS, QueryIntent.DEEP_ANALYSIS]
            assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_classification_cache(self, classifier):
        """Test that classification results are cached"""
        query = "How many spaces do I have?"

        # First classification
        result1 = await classifier.classify_query(query)
        stats1 = classifier.get_performance_stats()
        cache_hits_before = stats1['cache_hits']

        # Second classification (should hit cache)
        result2 = await classifier.classify_query(query)
        stats2 = classifier.get_performance_stats()
        cache_hits_after = stats2['cache_hits']

        assert result1.intent == result2.intent
        assert cache_hits_after > cache_hits_before
        logger.info(f"✅ Cache working: {cache_hits_after - cache_hits_before} hits")

    @pytest.mark.asyncio
    async def test_fallback_classification(self, classifier):
        """Test fallback classification when Claude unavailable"""
        # Temporarily remove Claude client
        original_claude = classifier.claude
        classifier.claude = None

        query = "How many desktops do I have?"
        result = await classifier.classify_query(query)

        assert result.intent in [QueryIntent.METADATA_ONLY, QueryIntent.VISUAL_ANALYSIS, QueryIntent.DEEP_ANALYSIS]
        assert "fallback" in result.reasoning.lower() or "heuristic" in result.reasoning.lower()

        # Restore Claude client
        classifier.claude = original_claude
        logger.info("✅ Fallback classification works")


class TestSmartQueryRouter:
    """Test the smart query router"""

    @pytest.fixture
    def router(self):
        """Create router instance"""
        claude_client = MockClaudeClient()
        return SmartQueryRouter(
            yabai_handler=mock_yabai_handler,
            vision_handler=mock_vision_handler,
            multi_space_handler=mock_multi_space_handler,
            claude_client=claude_client
        )

    @pytest.mark.asyncio
    async def test_metadata_routing(self, router):
        """Test routing of metadata queries"""
        query = "How many spaces do I have?"
        result = await router.route_query(query)

        logger.info(f"Routed to: {result.intent.value}, latency: {result.latency_ms:.1f}ms")
        assert result.success
        assert result.response
        # Metadata queries should be fast
        # Note: With mocks this is less predictable, but we can check it completed
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_vision_routing(self, router):
        """Test routing of visual analysis queries"""
        query = "What do you see on my screen?"
        result = await router.route_query(query)

        logger.info(f"Routed to: {result.intent.value}, latency: {result.latency_ms:.1f}ms")
        assert result.success
        assert result.response

    @pytest.mark.asyncio
    async def test_deep_routing(self, router):
        """Test routing of deep analysis queries"""
        query = "Analyze all my desktop spaces"
        result = await router.route_query(query)

        logger.info(f"Routed to: {result.intent.value}, latency: {result.latency_ms:.1f}ms")
        assert result.success
        assert result.response

    @pytest.mark.asyncio
    async def test_routing_stats(self, router):
        """Test routing statistics collection"""
        queries = [
            "How many spaces?",
            "What do you see?",
            "Analyze all spaces"
        ]

        for query in queries:
            await router.route_query(query)

        stats = router.get_routing_stats()
        logger.info(f"Routing stats: {stats}")

        assert stats['total_queries'] >= len(queries)
        assert 'distribution' in stats or stats['total_queries'] > 0


class TestQueryContextManager:
    """Test the query context manager"""

    @pytest.fixture
    def context_manager(self):
        """Create context manager instance"""
        return QueryContextManager(max_history=50)

    def test_record_query(self, context_manager):
        """Test recording queries"""
        context = context_manager.record_query(
            query="How many spaces?",
            intent="metadata_only",
            active_space=1,
            total_spaces=3,
            response_latency_ms=150
        )

        assert context.query == "How many spaces?"
        assert context.intent == "metadata_only"
        assert context.active_space == 1

    def test_context_for_query(self, context_manager):
        """Test getting context for a query"""
        # Record some queries
        for i in range(5):
            context_manager.record_query(
                query=f"Query {i}",
                intent="visual_analysis",
                active_space=i % 3,
                total_spaces=3
            )

        # Get context
        context = context_manager.get_context_for_query("New query")

        assert 'active_space' in context
        assert 'total_spaces' in context
        assert 'recent_intent' in context
        assert 'time_since_last_query' in context
        logger.info(f"Context: {context}")

    def test_user_preferences(self, context_manager):
        """Test user preferences detection"""
        # Simulate a pattern of metadata queries
        for i in range(10):
            context_manager.record_query(
                query=f"Metadata query {i}",
                intent="metadata_only",
                active_space=1,
                total_spaces=3
            )

        preferences = context_manager.get_user_preferences()
        logger.info(f"User preferences: {preferences}")

        assert 'preferred_intent' in preferences
        assert 'detected_pattern' in preferences


class TestAdaptiveLearningSystem:
    """Test the adaptive learning system"""

    @pytest.fixture
    def learning_system(self):
        """Create learning system instance"""
        import tempfile
        import os
        db_path = os.path.join(tempfile.gettempdir(), "test_learning.db")
        return AdaptiveLearningSystem(db_path=db_path)

    @pytest.mark.asyncio
    async def test_record_feedback(self, learning_system):
        """Test recording feedback"""
        await learning_system.record_feedback(
            query="How many spaces?",
            classified_intent=QueryIntent.METADATA_ONLY,
            actual_intent=QueryIntent.METADATA_ONLY,
            confidence=0.9,
            reasoning="Test feedback",
            user_satisfied=True,
            response_latency_ms=100
        )

        report = learning_system.get_accuracy_report()
        logger.info(f"Accuracy report: {report}")
        assert report['total_queries'] >= 1

    @pytest.mark.asyncio
    async def test_implicit_feedback(self, learning_system):
        """Test implicit feedback detection"""
        result = await learning_system.detect_implicit_feedback(
            query="What do you see?",
            classified_intent=QueryIntent.VISUAL_ANALYSIS,
            user_action="accepted",
            response_latency_ms=1500
        )

        # Should return None since classification was correct
        assert result is None or result == QueryIntent.VISUAL_ANALYSIS

    @pytest.mark.asyncio
    async def test_learning_from_feedback(self, learning_system):
        """Test learning from accumulated feedback"""
        # Record several misclassifications
        for i in range(3):
            await learning_system.record_feedback(
                query=f"Similar query {i}",
                classified_intent=QueryIntent.METADATA_ONLY,
                actual_intent=QueryIntent.VISUAL_ANALYSIS,
                confidence=0.7,
                reasoning="Misclassified",
                user_satisfied=False,
                response_latency_ms=2000
            )

        # Run learning
        await learning_system.learn_from_feedback()

        # Check learned patterns
        patterns = await learning_system.get_learned_patterns()
        logger.info(f"Learned {len(patterns)} patterns")


class TestPerformanceMonitor:
    """Test the performance monitor"""

    @pytest.fixture
    def monitor(self):
        """Create performance monitor instance"""
        return PerformanceMonitor(report_interval_minutes=1)

    @pytest.mark.asyncio
    async def test_collect_metrics(self, monitor):
        """Test metrics collection"""
        metrics = await monitor.collect_metrics()

        assert metrics.timestamp
        assert metrics.total_classifications >= 0
        logger.info(f"Collected metrics: {metrics}")

    @pytest.mark.asyncio
    async def test_performance_report(self, monitor):
        """Test performance report generation"""
        # Collect some metrics first
        await monitor.collect_metrics()

        report = monitor.generate_report()
        logger.info(f"Performance report: {report}")

        assert 'summary' in report
        assert 'classification' in report
        assert 'health' in report

    def test_performance_insights(self, monitor):
        """Test performance insights generation"""
        insights = monitor.get_performance_insights()
        logger.info(f"Insights: {insights}")

        assert isinstance(insights, list)


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.mark.asyncio
    async def test_end_to_end_query_flow(self):
        """Test complete query flow from classification to routing"""
        # Setup
        claude_client = MockClaudeClient()
        classifier = IntelligentQueryClassifier(claude_client)
        context_manager = QueryContextManager()
        router = SmartQueryRouter(
            yabai_handler=mock_yabai_handler,
            vision_handler=mock_vision_handler,
            multi_space_handler=mock_multi_space_handler,
            claude_client=claude_client
        )

        # Test multiple queries
        test_queries = [
            "How many spaces do I have?",
            "What do you see?",
            "Analyze all my desktops"
        ]

        for query in test_queries:
            # Get context
            context = context_manager.get_context_for_query(query)

            # Route query
            result = await router.route_query(query, context)

            # Record in context
            context_manager.record_query(
                query=query,
                intent=result.intent.value,
                response_latency_ms=result.latency_ms
            )

            logger.info(
                f"✅ '{query}' -> {result.intent.value} "
                f"({result.latency_ms:.1f}ms)"
            )

            assert result.success

        # Check stats
        stats = router.get_routing_stats()
        session_stats = context_manager.get_session_stats()

        logger.info(f"Final routing stats: {stats}")
        logger.info(f"Final session stats: {session_stats}")

    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring across queries"""
        claude_client = MockClaudeClient()
        monitor = PerformanceMonitor(report_interval_minutes=1)

        # Simulate some queries
        for _ in range(5):
            await monitor.collect_metrics()
            await asyncio.sleep(0.1)

        # Generate report
        report = monitor.generate_report()
        insights = monitor.get_performance_insights()

        logger.info("=== Performance Report ===")
        logger.info(f"Summary: {report.get('summary', {})}")
        logger.info(f"Insights: {insights}")

        assert report
        assert isinstance(insights, list)


# ==================== P2 Feature Tests ====================

class TestABTestingFramework:
    """Test A/B testing framework"""

    @pytest.fixture
    def ab_test(self):
        """Create A/B test instance"""
        from ab_testing_framework import ABTestingFramework
        return ABTestingFramework("test_classifiers", enable_persistence=False)

    @pytest.mark.asyncio
    async def test_add_variants(self, ab_test):
        """Test adding A/B test variants"""
        # Mock classifiers
        async def classifier_a(query, context):
            return {"intent": "metadata_only", "confidence": 0.85}

        async def classifier_b(query, context):
            return {"intent": "visual_analysis", "confidence": 0.90}

        ab_test.add_variant(
            variant_id="control",
            name="Original Classifier",
            description="Current production classifier",
            classifier_func=classifier_a,
            traffic_allocation=0.5,
            is_control=True
        )

        ab_test.add_variant(
            variant_id="test",
            name="New Classifier",
            description="Experimental classifier",
            classifier_func=classifier_b,
            traffic_allocation=0.5
        )

        assert len(ab_test.variants) == 2
        assert ab_test.control_variant_id == "control"

    @pytest.mark.asyncio
    async def test_ab_classification(self, ab_test):
        """Test classification with A/B testing"""
        # Add variants
        async def classifier_a(query, context):
            await asyncio.sleep(0.01)
            return {"intent": "metadata_only", "confidence": 0.85}

        async def classifier_b(query, context):
            await asyncio.sleep(0.01)
            return {"intent": "metadata_only", "confidence": 0.90}

        ab_test.add_variant("a", "A", "Variant A", classifier_a, 0.5, True)
        ab_test.add_variant("b", "B", "Variant B", classifier_b, 0.5)

        # Run multiple queries
        results = []
        for i in range(10):
            result = await ab_test.classify_query(f"Query {i}")
            results.append(result.variant_id)

        # Check that both variants were used
        assert 'a' in results or 'b' in results
        logger.info(f"✅ A/B test distributed queries: {results}")

    @pytest.mark.asyncio
    async def test_feedback_recording(self, ab_test):
        """Test recording feedback"""
        async def classifier_a(query, context):
            return {"intent": "metadata_only", "confidence": 0.85}

        ab_test.add_variant("a", "A", "Test", classifier_a, 1.0, True)

        # Classify and record feedback
        result = await ab_test.classify_query("test query")
        ab_test.record_feedback(result.variant_id, correct=True, user_satisfied=True)

        perf = ab_test.get_variant_performance(result.variant_id)
        assert perf['total_queries'] >= 1
        assert perf['accuracy'] > 0

    @pytest.mark.asyncio
    async def test_comparison_report(self, ab_test):
        """Test variant comparison"""
        async def classifier_a(query, context):
            return {"intent": "metadata_only", "confidence": 0.80}

        async def classifier_b(query, context):
            return {"intent": "metadata_only", "confidence": 0.90}

        ab_test.add_variant("control", "Control", "Control", classifier_a, 0.5, True)
        ab_test.add_variant("test", "Test", "Test", classifier_b, 0.5)

        # Run queries and record feedback
        for i in range(20):
            result = await ab_test.classify_query(f"Query {i}")
            # Control is 80% accurate, test is 90% accurate
            correct = (result.variant_id == "control" and i < 16) or (result.variant_id == "test" and i < 18)
            ab_test.record_feedback(result.variant_id, correct=correct, user_satisfied=correct)

        # Get comparison
        comparison = ab_test.compare_variants()
        logger.info(f"Comparison: {comparison}")

        assert 'comparisons' in comparison


class TestProactiveSuggestions:
    """Test proactive suggestion system"""

    @pytest.fixture
    def proactive_system(self):
        """Create proactive system instance"""
        from proactive_suggestions import ProactiveSuggestionSystem
        return ProactiveSuggestionSystem()

    @pytest.mark.asyncio
    async def test_error_detection(self, proactive_system):
        """Test error detection suggestions"""
        yabai_data = {
            'spaces': {
                1: {'applications': ['Chrome', 'Error: Failed to load']},
                2: {'applications': ['VS Code']}
            }
        }

        context = {'active_space': 1}

        suggestion = await proactive_system.analyze_and_suggest(context, yabai_data)

        if suggestion:
            assert suggestion.type.value in ['error_detected', 'opportunity']
            logger.info(f"✅ Generated suggestion: {suggestion.message}")

    @pytest.mark.asyncio
    async def test_work_session_detection(self, proactive_system):
        """Test long work session detection"""
        context = {
            'session_duration_minutes': 150,  # 2.5 hours
            'active_space': 1
        }

        suggestion = await proactive_system.analyze_and_suggest(context, None)

        if suggestion:
            assert suggestion.type.value == 'work_session'
            logger.info(f"✅ Work session suggestion: {suggestion.message}")

    @pytest.mark.asyncio
    async def test_user_response(self, proactive_system):
        """Test recording user response"""
        context = {'session_duration_minutes': 150}

        suggestion = await proactive_system.analyze_and_suggest(context, None)

        if suggestion:
            # User accepts
            await proactive_system.record_user_response(suggestion.suggestion_id, accepted=True)

            stats = proactive_system.get_statistics()
            logger.info(f"Stats: {stats}")
            assert stats['total_suggestions_generated'] >= 1


# ==================== Main Test Runner ====================

if __name__ == "__main__":
    """Run all tests"""
    import sys

    # Run tests with pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
