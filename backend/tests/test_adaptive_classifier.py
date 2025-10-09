"""
Test suite for Adaptive Intent Classification.
"""
import pytest
from backend.core.intent.adaptive_classifier import (
    IntentSignal,
    IntentResult,
    LexicalClassifier,
    ContextAwareClassifier,
    AdaptiveIntentEngine,
    WeightedVotingStrategy,
    ConfidenceThresholdStrategy,
)


class TestIntentSignal:
    """Test IntentSignal model."""

    def test_signal_creation(self):
        """Test signal creation with valid confidence."""
        signal = IntentSignal(
            label="test_intent",
            confidence=0.85,
            source="test_classifier",
        )

        assert signal.label == "test_intent"
        assert signal.confidence == 0.85

    def test_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError):
            IntentSignal(
                label="test",
                confidence=1.5,  # Invalid
                source="test",
            )


class TestLexicalClassifier:
    """Test LexicalClassifier."""

    def test_simple_pattern_matching(self):
        """Test basic pattern matching."""
        patterns = {
            "greeting": ["hello", "hi", "hey"],
            "farewell": ["bye", "goodbye", "see you"],
        }

        classifier = LexicalClassifier(
            name="test_lexical",
            patterns=patterns,
        )

        signals = classifier.classify("hello there", {})

        assert len(signals) > 0
        assert signals[0].label == "greeting"
        assert signals[0].confidence > 0.5

    def test_no_match(self):
        """Test when no patterns match."""
        patterns = {
            "greeting": ["hello", "hi"],
        }

        classifier = LexicalClassifier(
            name="test",
            patterns=patterns,
        )

        signals = classifier.classify("random text", {})

        assert len(signals) == 0

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        patterns = {
            "test": ["Hello"],
        }

        classifier = LexicalClassifier(
            name="test",
            patterns=patterns,
            case_sensitive=False,
        )

        signals = classifier.classify("HELLO world", {})

        assert len(signals) > 0
        assert signals[0].label == "test"


class TestContextAwareClassifier:
    """Test ContextAwareClassifier."""

    def test_context_boosting(self):
        """Test that context boosts intent confidence."""
        base_patterns = {
            "vision": ["terminal", "screen"],
        }

        base = LexicalClassifier(
            name="base",
            patterns=base_patterns,
        )

        # Boost vision intent when recent_context = vision
        boosters = {
            "recent_context": {
                "vision": 1.5,  # 50% boost
            }
        }

        aware = ContextAwareClassifier(
            name="context_aware",
            base_classifier=base,
            context_boosters=boosters,
        )

        # Without context
        signals1 = aware.classify("terminal window", {})
        base_confidence = signals1[0].confidence if signals1 else 0.0

        # With context
        signals2 = aware.classify("terminal window", {"recent_context": "vision"})
        boosted_confidence = signals2[0].confidence if signals2 else 0.0

        assert boosted_confidence > base_confidence

    @pytest.mark.asyncio
    async def test_async_classification(self):
        """Test async classification."""
        base_patterns = {
            "test": ["hello"],
        }

        base = LexicalClassifier(name="base", patterns=base_patterns)
        aware = ContextAwareClassifier(
            name="aware",
            base_classifier=base,
            context_boosters={},
        )

        signals = await aware.classify_async("hello", {})

        assert len(signals) > 0


class TestEnsembleStrategies:
    """Test ensemble aggregation strategies."""

    def test_weighted_voting(self):
        """Test weighted voting strategy."""
        strategy = WeightedVotingStrategy(
            source_weights={"classifier1": 2.0, "classifier2": 1.0},
            min_confidence=0.5,
        )

        signals = [
            IntentSignal("intent_a", 0.7, "classifier1"),
            IntentSignal("intent_a", 0.6, "classifier2"),
            IntentSignal("intent_b", 0.8, "classifier2"),
        ]

        result = strategy.aggregate(signals)

        # intent_a: (0.7*2 + 0.6*1) / 2 = 1.0
        # intent_b: (0.8*1) / 1 = 0.8
        assert result.primary_intent == "intent_a"
        assert result.confidence == 1.0

    def test_confidence_threshold(self):
        """Test confidence threshold strategy."""
        strategy = ConfidenceThresholdStrategy(min_confidence=0.8)

        signals = [
            IntentSignal("intent_a", 0.95, "classifier1"),
            IntentSignal("intent_b", 0.75, "classifier2"),
        ]

        result = strategy.aggregate(signals)

        assert result.primary_intent == "intent_a"
        assert result.confidence == 0.95

    def test_below_threshold(self):
        """Test when all signals below threshold."""
        strategy = ConfidenceThresholdStrategy(min_confidence=0.9)

        signals = [
            IntentSignal("intent_a", 0.7, "classifier1"),
            IntentSignal("intent_b", 0.6, "classifier2"),
        ]

        result = strategy.aggregate(signals)

        assert result.primary_intent == "unknown"


class TestAdaptiveIntentEngine:
    """Test AdaptiveIntentEngine."""

    @pytest.mark.asyncio
    async def test_single_classifier(self):
        """Test engine with single classifier."""
        patterns = {
            "greeting": ["hello", "hi"],
            "farewell": ["bye", "goodbye"],
        }

        classifier = LexicalClassifier(
            name="lexical",
            patterns=patterns,
        )

        engine = AdaptiveIntentEngine(
            classifiers=[classifier],
            strategy=ConfidenceThresholdStrategy(min_confidence=0.5),
        )

        result = await engine.classify("hello there")

        assert result.primary_intent == "greeting"
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_multiple_classifiers(self):
        """Test engine with multiple classifiers."""
        patterns1 = {"intent_a": ["hello"]}
        patterns2 = {"intent_a": ["hello", "hi"]}

        classifier1 = LexicalClassifier(name="c1", patterns=patterns1, priority=50)
        classifier2 = LexicalClassifier(name="c2", patterns=patterns2, priority=60)

        engine = AdaptiveIntentEngine(
            classifiers=[classifier1, classifier2],
            strategy=WeightedVotingStrategy(),
        )

        result = await engine.classify("hello")

        assert result.primary_intent == "intent_a"
        assert len(result.all_signals) >= 2

    @pytest.mark.asyncio
    async def test_add_remove_classifier(self):
        """Test dynamic classifier management."""
        engine = AdaptiveIntentEngine()

        assert engine.classifier_count == 0

        patterns = {"test": ["hello"]}
        classifier = LexicalClassifier(name="test", patterns=patterns)

        engine.add_classifier(classifier)
        assert engine.classifier_count == 1

        engine.remove_classifier("test")
        assert engine.classifier_count == 0

    @pytest.mark.asyncio
    async def test_no_match(self):
        """Test when no classifier matches."""
        patterns = {"greeting": ["hello"]}
        classifier = LexicalClassifier(name="test", patterns=patterns)

        engine = AdaptiveIntentEngine(
            classifiers=[classifier],
            strategy=ConfidenceThresholdStrategy(min_confidence=0.5),
        )

        result = await engine.classify("random text")

        assert result.primary_intent == "unknown"
        assert len(result.all_signals) == 0
