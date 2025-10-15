"""
Adaptive Intent Classification System
ML-ready, pluggable, and dynamically extensible.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, Sequence
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IntentSignal:
    """Single intent detection signal."""
    label: str
    confidence: float  # 0.0-1.0
    source: str  # classifier name
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


@dataclass(slots=True)
class IntentResult:
    """Aggregated intent classification result."""
    primary_intent: str
    confidence: float
    all_signals: list[IntentSignal] = field(default_factory=list)
    reasoning: str = ""
    context_hints: dict[str, Any] = field(default_factory=dict)

    def add_signal(self, signal: IntentSignal) -> None:
        """Add signal and recalculate."""
        self.all_signals.append(signal)

    def get_signals_by_label(self, label: str) -> list[IntentSignal]:
        """Filter signals by intent label."""
        return [s for s in self.all_signals if s.label == label]


class IntentClassifier(ABC):
    """Base classifier interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Classifier identifier."""
        ...

    @property
    def priority(self) -> int:
        """Execution priority (higher = earlier). Default: 50."""
        return 50

    @property
    def async_capable(self) -> bool:
        """Whether this classifier supports async execution."""
        return False

    @abstractmethod
    def classify(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        """
        Synchronous classification.
        Returns list of signals (can be multiple intents with different confidences).
        """
        ...

    async def classify_async(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        """
        Async classification (for ML models, API calls, etc.).
        Default: runs sync version in executor.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.classify, text, context)


class LexicalClassifier(IntentClassifier):
    """
    Pattern-based classifier using configurable rules.
    No hardcoding - loads from config.
    """

    def __init__(
        self,
        name: str,
        patterns: dict[str, list[str]],
        priority: int = 50,
        case_sensitive: bool = False,
        word_boundary: bool = True,
    ):
        self._name = name
        self._case_sensitive = case_sensitive
        self._word_boundary = word_boundary
        self._priority = priority

        # Compile patterns
        self._compiled: dict[str, list[re.Pattern]] = defaultdict(list)
        for intent, pattern_list in patterns.items():
            for pattern in pattern_list:
                flags = 0 if case_sensitive else re.IGNORECASE
                if word_boundary:
                    pattern = rf"\b{re.escape(pattern)}\b"
                self._compiled[intent].append(re.compile(pattern, flags))

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def classify(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        signals = []

        for intent, patterns in self._compiled.items():
            matches = sum(1 for p in patterns if p.search(text))
            if matches > 0:
                # Confidence = match ratio * pattern coverage
                match_ratio = matches / len(patterns)
                coverage = min(1.0, len(text.split()) / 10.0)  # longer = more context
                confidence = min(0.95, match_ratio * 0.8 + coverage * 0.2)

                signals.append(
                    IntentSignal(
                        label=intent,
                        confidence=confidence,
                        source=self.name,
                        features={"matches": matches, "patterns_total": len(patterns)},
                    )
                )

        return signals


class SemanticEmbeddingClassifier(IntentClassifier):
    """
    Embedding-based classifier (sentence transformers, OpenAI embeddings, etc.).
    Loads reference embeddings dynamically.
    """

    def __init__(
        self,
        name: str,
        embedding_fn: Callable[[str], Awaitable[list[float]]],
        intent_embeddings: dict[str, list[list[float]]],  # intent -> list of example embeddings
        threshold: float = 0.75,
        priority: int = 60,
    ):
        self._name = name
        self._embedding_fn = embedding_fn
        self._intent_embeddings = intent_embeddings
        self._threshold = threshold
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def async_capable(self) -> bool:
        return True

    def classify(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        # Sync version not supported
        raise NotImplementedError("Use classify_async for embedding-based classification")

    async def classify_async(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        text_embedding = await self._embedding_fn(text)
        signals = []

        for intent, ref_embeddings in self._intent_embeddings.items():
            # Compute cosine similarity with all reference embeddings
            similarities = [
                self._cosine_similarity(text_embedding, ref)
                for ref in ref_embeddings
            ]
            max_sim = max(similarities) if similarities else 0.0

            if max_sim >= self._threshold:
                signals.append(
                    IntentSignal(
                        label=intent,
                        confidence=max_sim,
                        source=self.name,
                        features={"max_similarity": max_sim, "ref_count": len(similarities)},
                    )
                )

        return signals

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class ContextAwareClassifier(IntentClassifier):
    """
    Uses conversation context to boost/suppress intents.
    Example: if recent context is "vision", boost vision-related intents.
    """

    def __init__(
        self,
        name: str,
        base_classifier: IntentClassifier,
        context_boosters: dict[str, dict[str, float]],  # context_key -> {intent: boost_factor}
        priority: int = 70,
    ):
        self._name = name
        self._base = base_classifier
        self._boosters = context_boosters
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def async_capable(self) -> bool:
        return self._base.async_capable

    def classify(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        base_signals = self._base.classify(text, context)
        return self._apply_boosts(base_signals, context)

    async def classify_async(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        base_signals = await self._base.classify_async(text, context)
        return self._apply_boosts(base_signals, context)

    def _apply_boosts(
        self, signals: list[IntentSignal], context: dict[str, Any]
    ) -> list[IntentSignal]:
        boosted = []
        for signal in signals:
            boost = 1.0
            for ctx_key, intent_boosts in self._boosters.items():
                if ctx_key in context and context[ctx_key]:
                    boost *= intent_boosts.get(signal.label, 1.0)

            new_conf = min(1.0, signal.confidence * boost)
            boosted.append(
                IntentSignal(
                    label=signal.label,
                    confidence=new_conf,
                    source=f"{self.name}+{signal.source}",
                    features={**signal.features, "boost_factor": boost},
                    metadata=signal.metadata,
                )
            )

        return boosted


class EnsembleStrategy(ABC):
    """Strategy for combining multiple signals."""

    @abstractmethod
    def aggregate(self, signals: list[IntentSignal]) -> IntentResult:
        ...


class WeightedVotingStrategy(EnsembleStrategy):
    """Combine signals using weighted voting."""

    def __init__(
        self,
        source_weights: dict[str, float] | None = None,
        min_confidence: float = 0.5,
    ):
        self._source_weights = source_weights or {}
        self._min_confidence = min_confidence

    def aggregate(self, signals: list[IntentSignal]) -> IntentResult:
        if not signals:
            return IntentResult(primary_intent="unknown", confidence=0.0)

        # Group by intent label
        by_intent: dict[str, list[IntentSignal]] = defaultdict(list)
        for sig in signals:
            by_intent[sig.label].append(sig)

        # Calculate weighted scores
        scores: dict[str, float] = {}
        for intent, sigs in by_intent.items():
            weighted_sum = sum(
                sig.confidence * self._source_weights.get(sig.source, 1.0)
                for sig in sigs
            )
            scores[intent] = weighted_sum / len(sigs)  # average

        # Pick highest
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        if best_score < self._min_confidence:
            return IntentResult(
                primary_intent="unknown",
                confidence=best_score,
                all_signals=signals,
                reasoning=f"Highest score {best_score:.2f} below threshold {self._min_confidence}",
            )

        return IntentResult(
            primary_intent=best_intent,
            confidence=best_score,
            all_signals=signals,
            reasoning=f"Weighted voting: {best_intent} scored {best_score:.2f}",
        )


class ConfidenceThresholdStrategy(EnsembleStrategy):
    """Pick highest confidence signal above threshold."""

    def __init__(self, min_confidence: float = 0.7):
        self._min_confidence = min_confidence

    def aggregate(self, signals: list[IntentSignal]) -> IntentResult:
        if not signals:
            return IntentResult(primary_intent="unknown", confidence=0.0)

        # Sort by confidence descending
        sorted_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)
        best = sorted_signals[0]

        if best.confidence < self._min_confidence:
            return IntentResult(
                primary_intent="unknown",
                confidence=best.confidence,
                all_signals=signals,
                reasoning=f"Best confidence {best.confidence:.2f} below threshold",
            )

        return IntentResult(
            primary_intent=best.label,
            confidence=best.confidence,
            all_signals=signals,
            reasoning=f"Highest confidence: {best.label} at {best.confidence:.2f}",
        )


class AdaptiveIntentEngine:
    """
    Orchestrates multiple classifiers and aggregates results.
    Fully dynamic - classifiers can be added/removed at runtime.
    """

    def __init__(
        self,
        classifiers: Sequence[IntentClassifier] | None = None,
        strategy: EnsembleStrategy | None = None,
    ):
        self._classifiers: list[IntentClassifier] = list(classifiers or [])
        self._strategy = strategy or WeightedVotingStrategy()

    def add_classifier(self, classifier: IntentClassifier) -> None:
        """Add classifier at runtime."""
        self._classifiers.append(classifier)
        self._classifiers.sort(key=lambda c: c.priority, reverse=True)

    def remove_classifier(self, name: str) -> None:
        """Remove classifier by name."""
        self._classifiers = [c for c in self._classifiers if c.name != name]

    async def classify(
        self, text: str, context: dict[str, Any] | None = None
    ) -> IntentResult:
        """
        Run all classifiers and aggregate results.
        Handles both sync and async classifiers efficiently.
        """
        context = context or {}
        all_signals: list[IntentSignal] = []

        # Separate sync and async classifiers
        sync_classifiers = [c for c in self._classifiers if not c.async_capable]
        async_classifiers = [c for c in self._classifiers if c.async_capable]

        # Run sync classifiers
        for classifier in sync_classifiers:
            try:
                signals = classifier.classify(text, context)
                all_signals.extend(signals)
                logger.debug(
                    f"Classifier {classifier.name} produced {len(signals)} signals"
                )
            except Exception as e:
                logger.error(f"Classifier {classifier.name} failed: {e}", exc_info=True)

        # Run async classifiers concurrently
        if async_classifiers:
            tasks = [c.classify_async(text, context) for c in async_classifiers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for classifier, result in zip(async_classifiers, results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Async classifier {classifier.name} failed: {result}",
                        exc_info=result,
                    )
                else:
                    all_signals.extend(result)
                    logger.debug(
                        f"Async classifier {classifier.name} produced {len(result)} signals"
                    )

        # Aggregate
        return self._strategy.aggregate(all_signals)

    def get_classifier(self, name: str) -> IntentClassifier | None:
        """Retrieve classifier by name."""
        return next((c for c in self._classifiers if c.name == name), None)

    @property
    def classifier_count(self) -> int:
        return len(self._classifiers)
