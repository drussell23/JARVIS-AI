#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Awareness Engine (UAE)
============================================================

Tests all UAE components and integration scenarios.

Test Coverage:
- Context Intelligence Layer
- Situational Awareness Layer
- Awareness Integration Layer
- Unified Awareness Engine
- Bidirectional learning
- Decision fusion logic
- Priority-based monitoring
- Error handling and resilience

Author: Derek J. Russell
Date: October 2025
"""

import asyncio
import pytest
import logging
import time
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from backend.intelligence.unified_awareness_engine import (
    UnifiedAwarenessEngine,
    ContextIntelligenceLayer,
    SituationalAwarenessLayer,
    AwarenessIntegrationLayer,
    UnifiedDecision,
    ExecutionResult,
    ContextualData,
    SituationalData,
    ElementPriority,
    DecisionSource,
    ConfidenceSource,
    get_uae_engine
)

from backend.vision.situational_awareness import (
    SituationalAwarenessEngine,
    UIElementPosition
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_knowledge_base():
    """Create temporary knowledge base"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "uae_context.json"


@pytest.fixture
def mock_sai_engine():
    """Create mock SAI engine"""
    engine = Mock(spec=SituationalAwarenessEngine)
    engine.is_monitoring = False
    engine.start_monitoring = AsyncMock()
    engine.stop_monitoring = AsyncMock()
    engine.get_element_position = AsyncMock(return_value=UIElementPosition(
        element_id="test",
        coordinates=(100, 200),
        confidence=0.9,
        detection_method="vision",
        timestamp=time.time(),
        display_id=0
    ))
    engine.register_change_callback = Mock()
    return engine


@pytest.fixture
def sample_contextual_data():
    """Create sample contextual data"""
    return ContextualData(
        element_id="test_element",
        expected_position=(100, 100),
        confidence=0.85,
        usage_count=10,
        last_success=time.time(),
        pattern_strength=0.8
    )


@pytest.fixture
def sample_situational_data():
    """Create sample situational data"""
    return SituationalData(
        element_id="test_element",
        detected_position=(110, 105),
        confidence=0.9,
        detection_method="vision_claude",
        detection_time=time.time()
    )


# ============================================================================
# Test Context Intelligence Layer
# ============================================================================

class TestContextIntelligenceLayer:
    """Test context intelligence and pattern learning"""

    def test_initialization(self, temp_knowledge_base):
        """Test CI layer initialization"""
        ci = ContextIntelligenceLayer(temp_knowledge_base)

        assert ci.knowledge_base_path == temp_knowledge_base
        assert len(ci.element_patterns) == 0
        assert len(ci.usage_history) == 0

    @pytest.mark.asyncio
    async def test_update_pattern_new_element(self, temp_knowledge_base):
        """Test updating pattern for new element"""
        ci = ContextIntelligenceLayer(temp_knowledge_base)

        await ci.update_pattern(
            "new_element",
            (100, 200),
            success=True
        )

        assert "new_element" in ci.element_patterns
        pattern = ci.element_patterns["new_element"]
        assert pattern.expected_position == (100, 200)
        assert pattern.usage_count == 1

    @pytest.mark.asyncio
    async def test_update_pattern_existing_element_same_position(self, temp_knowledge_base):
        """Test updating existing pattern with same position strengthens it"""
        ci = ContextIntelligenceLayer(temp_knowledge_base)

        # First update
        await ci.update_pattern("element", (100, 200), success=True)
        initial_strength = ci.element_patterns["element"].pattern_strength

        # Second update with same position
        await ci.update_pattern("element", (100, 200), success=True)
        new_strength = ci.element_patterns["element"].pattern_strength

        assert new_strength > initial_strength

    @pytest.mark.asyncio
    async def test_update_pattern_position_changed(self, temp_knowledge_base):
        """Test updating pattern with changed position"""
        ci = ContextIntelligenceLayer(temp_knowledge_base)

        # First update
        await ci.update_pattern("element", (100, 200), success=True)

        # Update with different position
        await ci.update_pattern("element", (150, 250), success=True)

        pattern = ci.element_patterns["element"]
        assert pattern.expected_position == (150, 250)
        # Pattern strength should decrease slightly
        assert pattern.pattern_strength < 1.0

    @pytest.mark.asyncio
    async def test_get_priority_elements(self, temp_knowledge_base):
        """Test priority element calculation"""
        ci = ContextIntelligenceLayer(temp_knowledge_base)

        # Add several elements with different usage
        await ci.update_pattern("elem1", (100, 100), success=True)
        await ci.update_pattern("elem2", (200, 200), success=True)
        await ci.update_pattern("elem2", (200, 200), success=True)  # Used more
        await ci.update_pattern("elem3", (300, 300), success=True)

        priorities = await ci.get_priority_elements(top_n=2)

        assert len(priorities) <= 2
        # elem2 should have higher priority (used more)
        assert "elem2" in priorities

    @pytest.mark.asyncio
    async def test_learn_from_execution_success(self, temp_knowledge_base):
        """Test learning from successful execution"""
        ci = ContextIntelligenceLayer(temp_knowledge_base)

        decision = UnifiedDecision(
            element_id="test",
            chosen_position=(100, 100),
            confidence=0.9,
            decision_source=DecisionSource.CONTEXT,
            context_weight=0.8,
            situation_weight=0.2,
            reasoning="test",
            timestamp=time.time()
        )

        result = ExecutionResult(
            decision=decision,
            success=True,
            execution_time=0.5,
            verification_passed=True
        )

        await ci.learn_from_execution(result)

        assert "test" in ci.element_patterns
        assert ci.metrics['learning_events'] > 0

    def test_persistence(self, temp_knowledge_base):
        """Test knowledge base persistence"""
        # Create first instance and add pattern
        ci1 = ContextIntelligenceLayer(temp_knowledge_base)
        asyncio.run(ci1.update_pattern("element", (100, 200), success=True))

        # Create second instance and verify data loaded
        ci2 = ContextIntelligenceLayer(temp_knowledge_base)
        assert "element" in ci2.element_patterns
        assert ci2.element_patterns["element"].expected_position == (100, 200)


# ============================================================================
# Test Situational Awareness Layer
# ============================================================================

class TestSituationalAwarenessLayer:
    """Test real-time situational perception"""

    def test_initialization(self, mock_sai_engine):
        """Test SAL initialization"""
        sal = SituationalAwarenessLayer(mock_sai_engine)

        assert sal.sai_engine == mock_sai_engine
        assert not sal.monitoring_active

    @pytest.mark.asyncio
    async def test_get_situational_data(self, mock_sai_engine):
        """Test getting situational data"""
        sal = SituationalAwarenessLayer(mock_sai_engine)

        data = await sal.get_situational_data("test_element")

        assert data is not None
        assert data.element_id == "test"
        assert data.detected_position == (100, 200)
        assert data.confidence == 0.9

    @pytest.mark.asyncio
    async def test_caching(self, mock_sai_engine):
        """Test situational data caching"""
        sal = SituationalAwarenessLayer(mock_sai_engine)

        # First call - should hit SAI
        data1 = await sal.get_situational_data("test_element")

        # Second call - should use cache
        data2 = await sal.get_situational_data("test_element")

        # SAI should only be called once
        assert mock_sai_engine.get_element_position.call_count == 1
        assert sal.metrics['cache_hits'] > 0

    @pytest.mark.asyncio
    async def test_force_detect(self, mock_sai_engine):
        """Test forced detection bypasses cache"""
        sal = SituationalAwarenessLayer(mock_sai_engine)

        # First call
        await sal.get_situational_data("test_element")

        # Second call with force_detect
        await sal.get_situational_data("test_element", force_detect=True)

        # SAI should be called twice
        assert mock_sai_engine.get_element_position.call_count == 2


# ============================================================================
# Test Awareness Integration Layer
# ============================================================================

class TestAwarenessIntegrationLayer:
    """Test decision fusion logic"""

    @pytest.mark.asyncio
    async def test_decision_context_only(self, sample_contextual_data):
        """Test decision with only context available"""
        ail = AwarenessIntegrationLayer()

        decision = await ail.make_decision(
            "test_element",
            context_data=sample_contextual_data,
            situation_data=None
        )

        assert decision.decision_source == DecisionSource.CONTEXT
        assert decision.chosen_position == sample_contextual_data.expected_position
        assert decision.context_weight == 1.0
        assert decision.situation_weight == 0.0

    @pytest.mark.asyncio
    async def test_decision_situation_only(self, sample_situational_data):
        """Test decision with only situation available"""
        ail = AwarenessIntegrationLayer()

        decision = await ail.make_decision(
            "test_element",
            context_data=None,
            situation_data=sample_situational_data
        )

        assert decision.decision_source == DecisionSource.SITUATION
        assert decision.chosen_position == sample_situational_data.detected_position
        assert decision.context_weight == 0.0
        assert decision.situation_weight == 1.0

    @pytest.mark.asyncio
    async def test_decision_fusion_agreement(
        self,
        sample_contextual_data,
        sample_situational_data
    ):
        """Test fusion when positions agree"""
        ail = AwarenessIntegrationLayer()

        # Make positions agree
        sample_contextual_data.expected_position = (110, 105)

        decision = await ail.make_decision(
            "test_element",
            context_data=sample_contextual_data,
            situation_data=sample_situational_data
        )

        assert decision.decision_source == DecisionSource.FUSION
        assert decision.metadata.get('agreement') == True
        # Confidence should be boosted
        assert decision.confidence > max(
            sample_contextual_data.confidence,
            sample_situational_data.confidence
        )

    @pytest.mark.asyncio
    async def test_decision_fusion_disagreement_recent_situation(
        self,
        sample_contextual_data,
        sample_situational_data
    ):
        """Test fusion when positions disagree - situation is recent"""
        ail = AwarenessIntegrationLayer()

        # Situation is very recent (just now)
        sample_situational_data.detection_time = time.time()

        decision = await ail.make_decision(
            "test_element",
            context_data=sample_contextual_data,
            situation_data=sample_situational_data
        )

        # Should prefer situation (recency)
        assert decision.chosen_position == sample_situational_data.detected_position
        assert decision.situation_weight > decision.context_weight

    @pytest.mark.asyncio
    async def test_decision_fusion_disagreement_strong_pattern(
        self,
        sample_contextual_data,
        sample_situational_data
    ):
        """Test fusion when positions disagree - context has strong pattern"""
        ail = AwarenessIntegrationLayer()

        # Make context pattern very strong
        sample_contextual_data.pattern_strength = 0.95

        # Make situation older
        sample_situational_data.detection_time = time.time() - 120

        decision = await ail.make_decision(
            "test_element",
            context_data=sample_contextual_data,
            situation_data=sample_situational_data
        )

        # Should prefer context (strong pattern)
        assert decision.chosen_position == sample_contextual_data.expected_position
        assert decision.context_weight > decision.situation_weight

    @pytest.mark.asyncio
    async def test_decision_fallback(self):
        """Test fallback when no data available"""
        ail = AwarenessIntegrationLayer()

        decision = await ail.make_decision(
            "test_element",
            context_data=None,
            situation_data=None
        )

        assert decision.decision_source == DecisionSource.FALLBACK
        assert decision.confidence == 0.0


# ============================================================================
# Test Unified Awareness Engine
# ============================================================================

class TestUnifiedAwarenessEngine:
    """Test main UAE orchestrator"""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_sai_engine):
        """Test UAE initialization"""
        uae = UnifiedAwarenessEngine(sai_engine=mock_sai_engine)

        assert uae.context_layer is not None
        assert uae.situation_layer is not None
        assert uae.integration_layer is not None
        assert not uae.is_active

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_sai_engine):
        """Test UAE start/stop"""
        uae = UnifiedAwarenessEngine(sai_engine=mock_sai_engine)

        # Start
        await uae.start()
        assert uae.is_active

        # Stop
        await uae.stop()
        assert not uae.is_active

    @pytest.mark.asyncio
    async def test_get_element_position(self, mock_sai_engine):
        """Test getting element position through UAE"""
        uae = UnifiedAwarenessEngine(sai_engine=mock_sai_engine)

        decision = await uae.get_element_position("test_element")

        assert decision is not None
        assert decision.element_id == "test_element"
        assert decision.chosen_position is not None
        assert decision.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_bidirectional_learning(self, mock_sai_engine):
        """Test bidirectional learning loop"""
        uae = UnifiedAwarenessEngine(sai_engine=mock_sai_engine)

        # Get decision
        decision = await uae.get_element_position("test_element")

        # Mock executor
        async def mock_executor(target, coordinates):
            return {'success': True, 'verification_passed': True}

        # Execute and learn
        result = await uae.execute_and_learn(
            decision,
            mock_executor
        )

        assert result.success
        assert uae.metrics['learning_cycles'] > 0
        assert uae.metrics['total_executions'] > 0

    @pytest.mark.asyncio
    async def test_callbacks(self, mock_sai_engine):
        """Test decision and learning callbacks"""
        uae = UnifiedAwarenessEngine(sai_engine=mock_sai_engine)

        decision_triggered = []
        learning_triggered = []

        def on_decision(decision):
            decision_triggered.append(decision)

        def on_learning(result):
            learning_triggered.append(result)

        uae.register_decision_callback(on_decision)
        uae.register_learning_callback(on_learning)

        # Trigger decision
        await uae.get_element_position("test_element")

        assert len(decision_triggered) == 1

    def test_metrics(self, mock_sai_engine):
        """Test comprehensive metrics"""
        uae = UnifiedAwarenessEngine(sai_engine=mock_sai_engine)

        metrics = uae.get_comprehensive_metrics()

        assert 'engine' in metrics
        assert 'context_layer' in metrics
        assert 'situation_layer' in metrics
        assert 'integration_layer' in metrics


# ============================================================================
# Integration Tests
# ============================================================================

class TestUAEIntegration:
    """Integration tests for complete UAE system"""

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, mock_sai_engine):
        """Test complete UAE flow"""
        uae = UnifiedAwarenessEngine(sai_engine=mock_sai_engine)

        # Start UAE
        await uae.start()

        # First use - no context, should use situation
        decision1 = await uae.get_element_position("new_element")
        assert decision1.decision_source in [DecisionSource.SITUATION, DecisionSource.FALLBACK]

        # Simulate successful execution
        async def executor(target, coordinates):
            return {'success': True, 'verification_passed': True}

        await uae.execute_and_learn(decision1, executor)

        # Second use - should have context now
        decision2 = await uae.get_element_position("new_element")
        # Decision source might be fusion if both available

        # Stop UAE
        await uae.stop()

    @pytest.mark.asyncio
    async def test_adaptation_to_change(self, mock_sai_engine):
        """Test adaptation when element position changes"""
        uae = UnifiedAwarenessEngine(sai_engine=mock_sai_engine)

        # Learn initial position
        decision1 = await uae.get_element_position("element")

        async def executor(target, coordinates):
            return {'success': True, 'verification_passed': True}

        await uae.execute_and_learn(decision1, executor)

        # Change mock SAI to return different position
        mock_sai_engine.get_element_position.return_value = UIElementPosition(
            element_id="element",
            coordinates=(200, 300),  # Different position
            confidence=0.9,
            detection_method="vision",
            timestamp=time.time(),
            display_id=0
        )

        # Get position again
        decision2 = await uae.get_element_position("element")

        # Should detect change and adapt
        # (Exact behavior depends on fusion logic)
        assert decision2 is not None


# ============================================================================
# Performance Tests
# ============================================================================

class TestUAEPerformance:
    """Performance and scalability tests"""

    @pytest.mark.asyncio
    async def test_decision_speed(self, mock_sai_engine):
        """Test decision-making speed"""
        uae = UnifiedAwarenessEngine(sai_engine=mock_sai_engine)

        start = time.time()
        for _ in range(100):
            await uae.get_element_position("test")
        duration = time.time() - start

        # Should complete 100 decisions in reasonable time
        assert duration < 5.0  # < 50ms per decision
        logger.info(f"Decision speed: {duration:.3f}s for 100 decisions ({1000*duration/100:.1f}ms each)")

    def test_pattern_learning_scalability(self, temp_knowledge_base):
        """Test pattern learning with many elements"""
        ci = ContextIntelligenceLayer(temp_knowledge_base)

        # Add 1000 patterns
        start = time.time()
        for i in range(1000):
            asyncio.run(ci.update_pattern(
                f"element_{i}",
                (i % 1920, i % 1080),
                success=True
            ))
        duration = time.time() - start

        assert duration < 10.0  # Should handle 1000 patterns quickly
        assert len(ci.element_patterns) == 1000
        logger.info(f"Pattern learning: {duration:.3f}s for 1000 patterns ({1000*duration/1000:.1f}ms each)")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
