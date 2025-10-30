"""
Adaptive Intent Classification System

This module provides a comprehensive, ML-ready intent classification framework that is
pluggable and dynamically extensible. It supports multiple classification strategies
including lexical pattern matching, semantic embeddings, and context-aware classification.

The system uses an ensemble approach where multiple classifiers can contribute signals
that are then aggregated using configurable strategies to determine the final intent.

Example:
    >>> from backend.core.intent.adaptive_classifier import AdaptiveIntentEngine, LexicalClassifier
    >>> 
    >>> # Create a simple lexical classifier
    >>> patterns = {"greeting": ["hello", "hi", "hey"], "goodbye": ["bye", "farewell"]}
    >>> classifier = LexicalClassifier("lexical", patterns)
    >>> 
    >>> # Create engine and classify
    >>> engine = AdaptiveIntentEngine([classifier])
    >>> result = await engine.classify("hello there")
    >>> print(result.primary_intent)  # "greeting"
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
    """Single intent detection signal from a classifier.
    
    Represents a single classification result with confidence score and metadata.
    Multiple signals can be generated for the same input text by different classifiers.
    
    Attributes:
        label: The intent label/name (e.g., "greeting", "question", "command")
        confidence: Confidence score between 0.0 and 1.0
        source: Name of the classifier that generated this signal
        features: Additional features extracted during classification
        metadata: Extra metadata about the classification process
        
    Raises:
        ValueError: If confidence is not between 0.0 and 1.0
        
    Example:
        >>> signal = IntentSignal("greeting", 0.85, "lexical_classifier")
        >>> print(f"{signal.label}: {signal.confidence}")
        greeting: 0.85
    """
    label: str
    confidence: float  # 0.0-1.0
    source: str  # classifier name
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate confidence score is within valid range.
        
        Raises:
            ValueError: If confidence is not between 0.0 and 1.0
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


@dataclass(slots=True)
class IntentResult:
    """Aggregated intent classification result from multiple classifiers.
    
    Contains the final classification decision along with all contributing signals
    and reasoning for the decision.
    
    Attributes:
        primary_intent: The final determined intent label
        confidence: Overall confidence in the primary intent
        all_signals: List of all signals that contributed to this result
        reasoning: Human-readable explanation of how the decision was made
        context_hints: Additional context information for downstream processing
        
    Example:
        >>> result = IntentResult("greeting", 0.9)
        >>> result.add_signal(IntentSignal("greeting", 0.85, "lexical"))
        >>> print(len(result.all_signals))  # 1
    """
    primary_intent: str
    confidence: float
    all_signals: list[IntentSignal] = field(default_factory=list)
    reasoning: str = ""
    context_hints: dict[str, Any] = field(default_factory=dict)

    def add_signal(self, signal: IntentSignal) -> None:
        """Add a signal to the result.
        
        Args:
            signal: The IntentSignal to add to this result
        """
        self.all_signals.append(signal)

    def get_signals_by_label(self, label: str) -> list[IntentSignal]:
        """Filter signals by intent label.
        
        Args:
            label: The intent label to filter by
            
        Returns:
            List of signals matching the specified label
        """
        return [s for s in self.all_signals if s.label == label]


class IntentClassifier(ABC):
    """Abstract base class for intent classifiers.
    
    Defines the interface that all intent classifiers must implement.
    Supports both synchronous and asynchronous classification.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this classifier.
        
        Returns:
            String identifier for the classifier
        """
        ...

    @property
    def priority(self) -> int:
        """Execution priority for this classifier.
        
        Higher values execute earlier. Used for ordering classifiers
        when multiple are available.
        
        Returns:
            Priority value (default: 50)
        """
        return 50

    @property
    def async_capable(self) -> bool:
        """Whether this classifier supports asynchronous execution.
        
        Returns:
            True if classifier can run asynchronously, False otherwise
        """
        return False

    @abstractmethod
    def classify(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        """Perform synchronous intent classification.
        
        Args:
            text: Input text to classify
            context: Additional context information
            
        Returns:
            List of IntentSignal objects representing classification results
            
        Note:
            Can return multiple signals with different intents and confidences
        """
        ...

    async def classify_async(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        """Perform asynchronous intent classification.
        
        Default implementation runs the synchronous version in an executor.
        Override for true async implementations (e.g., API calls, ML models).
        
        Args:
            text: Input text to classify
            context: Additional context information
            
        Returns:
            List of IntentSignal objects representing classification results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.classify, text, context)


class LexicalClassifier(IntentClassifier):
    """Pattern-based classifier using configurable regex rules.
    
    Uses regular expressions to match text patterns against predefined intent patterns.
    Supports case sensitivity and word boundary matching options.
    
    Attributes:
        _name: Classifier identifier
        _case_sensitive: Whether pattern matching is case sensitive
        _word_boundary: Whether to enforce word boundaries in patterns
        _priority: Execution priority
        _compiled: Compiled regex patterns organized by intent
        
    Example:
        >>> patterns = {
        ...     "greeting": ["hello", "hi", "hey"],
        ...     "question": ["what", "how", "why"]
        ... }
        >>> classifier = LexicalClassifier("lexical", patterns)
        >>> signals = classifier.classify("hello world", {})
        >>> print(signals[0].label)  # "greeting"
    """

    def __init__(
        self,
        name: str,
        patterns: dict[str, list[str]],
        priority: int = 50,
        case_sensitive: bool = False,
        word_boundary: bool = True,
    ):
        """Initialize the lexical classifier.
        
        Args:
            name: Unique identifier for this classifier
            patterns: Dictionary mapping intent labels to lists of pattern strings
            priority: Execution priority (higher = earlier)
            case_sensitive: Whether pattern matching should be case sensitive
            word_boundary: Whether to enforce word boundaries in pattern matching
        """
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
        """Return the classifier name.
        
        Returns:
            The classifier's unique identifier
        """
        return self._name

    @property
    def priority(self) -> int:
        """Return the classifier priority.
        
        Returns:
            The execution priority value
        """
        return self._priority

    def classify(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        """Classify text using lexical patterns.
        
        Args:
            text: Input text to classify
            context: Additional context (unused in lexical classification)
            
        Returns:
            List of IntentSignal objects for matching patterns
            
        Note:
            Confidence is calculated based on match ratio and text coverage
        """
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
    """Embedding-based classifier using semantic similarity.
    
    Uses vector embeddings to classify text based on semantic similarity
    to reference examples. Supports various embedding providers (sentence
    transformers, OpenAI, etc.).
    
    Attributes:
        _name: Classifier identifier
        _embedding_fn: Async function to generate embeddings
        _intent_embeddings: Reference embeddings for each intent
        _threshold: Minimum similarity threshold for classification
        _priority: Execution priority
        
    Example:
        >>> async def embed_fn(text):
        ...     # Your embedding implementation
        ...     return [0.1, 0.2, 0.3]  # Mock embedding
        >>> 
        >>> intent_embeddings = {
        ...     "greeting": [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]
        ... }
        >>> classifier = SemanticEmbeddingClassifier("semantic", embed_fn, intent_embeddings)
    """

    def __init__(
        self,
        name: str,
        embedding_fn: Callable[[str], Awaitable[list[float]]],
        intent_embeddings: dict[str, list[list[float]]],  # intent -> list of example embeddings
        threshold: float = 0.75,
        priority: int = 60,
    ):
        """Initialize the semantic embedding classifier.
        
        Args:
            name: Unique identifier for this classifier
            embedding_fn: Async function that takes text and returns embedding vector
            intent_embeddings: Dictionary mapping intents to lists of reference embeddings
            threshold: Minimum cosine similarity threshold for classification
            priority: Execution priority (higher = earlier)
        """
        self._name = name
        self._embedding_fn = embedding_fn
        self._intent_embeddings = intent_embeddings
        self._threshold = threshold
        self._priority = priority

    @property
    def name(self) -> str:
        """Return the classifier name.
        
        Returns:
            The classifier's unique identifier
        """
        return self._name

    @property
    def priority(self) -> int:
        """Return the classifier priority.
        
        Returns:
            The execution priority value
        """
        return self._priority

    @property
    def async_capable(self) -> bool:
        """Return async capability status.
        
        Returns:
            True, as this classifier requires async embedding generation
        """
        return True

    def classify(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        """Synchronous classification not supported.
        
        Args:
            text: Input text to classify
            context: Additional context information
            
        Raises:
            NotImplementedError: Always, as this classifier requires async operation
        """
        # Sync version not supported
        raise NotImplementedError("Use classify_async for embedding-based classification")

    async def classify_async(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        """Classify text using semantic embeddings.
        
        Args:
            text: Input text to classify
            context: Additional context (unused in embedding classification)
            
        Returns:
            List of IntentSignal objects for intents above similarity threshold
            
        Note:
            Uses cosine similarity to compare input embedding with reference embeddings
        """
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
        """Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity value between -1 and 1
            
        Note:
            Returns 0.0 if vectors have different lengths or zero norms
        """
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class ContextAwareClassifier(IntentClassifier):
    """Context-aware classifier that modifies base classifier results.
    
    Wraps another classifier and applies context-based boosts or penalties
    to classification results based on conversation history or other contextual
    information.
    
    Attributes:
        _name: Classifier identifier
        _base: Base classifier to wrap
        _boosters: Context-based boost factors for different intents
        _priority: Execution priority
        
    Example:
        >>> base = LexicalClassifier("base", {"greeting": ["hello"]})
        >>> boosters = {"recent_greeting": {"greeting": 1.5, "goodbye": 0.5}}
        >>> classifier = ContextAwareClassifier("context", base, boosters)
    """

    def __init__(
        self,
        name: str,
        base_classifier: IntentClassifier,
        context_boosters: dict[str, dict[str, float]],  # context_key -> {intent: boost_factor}
        priority: int = 70,
    ):
        """Initialize the context-aware classifier.
        
        Args:
            name: Unique identifier for this classifier
            base_classifier: The base classifier to wrap
            context_boosters: Dictionary mapping context keys to intent boost factors
            priority: Execution priority (higher = earlier)
        """
        self._name = name
        self._base = base_classifier
        self._boosters = context_boosters
        self._priority = priority

    @property
    def name(self) -> str:
        """Return the classifier name.
        
        Returns:
            The classifier's unique identifier
        """
        return self._name

    @property
    def priority(self) -> int:
        """Return the classifier priority.
        
        Returns:
            The execution priority value
        """
        return self._priority

    @property
    def async_capable(self) -> bool:
        """Return async capability based on base classifier.
        
        Returns:
            True if base classifier supports async operation
        """
        return self._base.async_capable

    def classify(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        """Classify text with context-aware boosting.
        
        Args:
            text: Input text to classify
            context: Context information used for boosting
            
        Returns:
            List of IntentSignal objects with context-adjusted confidences
        """
        base_signals = self._base.classify(text, context)
        return self._apply_boosts(base_signals, context)

    async def classify_async(self, text: str, context: dict[str, Any]) -> list[IntentSignal]:
        """Classify text asynchronously with context-aware boosting.
        
        Args:
            text: Input text to classify
            context: Context information used for boosting
            
        Returns:
            List of IntentSignal objects with context-adjusted confidences
        """
        base_signals = await self._base.classify_async(text, context)
        return self._apply_boosts(base_signals, context)

    def _apply_boosts(
        self, signals: list[IntentSignal], context: dict[str, Any]
    ) -> list[IntentSignal]:
        """Apply context-based boosts to classification signals.
        
        Args:
            signals: Original signals from base classifier
            context: Context information for determining boosts
            
        Returns:
            List of signals with adjusted confidence scores
        """
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
    """Abstract base class for signal aggregation strategies.
    
    Defines the interface for combining multiple IntentSignal objects
    into a single IntentResult.
    """

    @abstractmethod
    def aggregate(self, signals: list[IntentSignal]) -> IntentResult:
        """Aggregate multiple signals into a single result.
        
        Args:
            signals: List of IntentSignal objects to aggregate
            
        Returns:
            IntentResult containing the aggregated classification decision
        """
        ...


class WeightedVotingStrategy(EnsembleStrategy):
    """Ensemble strategy using weighted voting to combine signals.
    
    Combines signals by weighting them based on their source classifier
    and computing weighted averages for each intent.
    
    Attributes:
        _source_weights: Weights for different classifier sources
        _min_confidence: Minimum confidence threshold for valid results
        
    Example:
        >>> weights = {"lexical": 1.0, "semantic": 1.5}
        >>> strategy = WeightedVotingStrategy(weights, min_confidence=0.6)
    """

    def __init__(
        self,
        source_weights: dict[str, float] | None = None,
        min_confidence: float = 0.5,
    ):
        """Initialize the weighted voting strategy.
        
        Args:
            source_weights: Dictionary mapping classifier names to weight values
            min_confidence: Minimum confidence threshold for accepting results
        """
        self._source_weights = source_weights or {}
        self._min_confidence = min_confidence

    def aggregate(self, signals: list[IntentSignal]) -> IntentResult:
        """Aggregate signals using weighted voting.
        
        Args:
            signals: List of IntentSignal objects to aggregate
            
        Returns:
            IntentResult with the highest-scoring intent or "unknown" if below threshold
            
        Note:
            Groups signals by intent label and computes weighted average scores
        """
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
    """Ensemble strategy that selects the highest confidence signal above threshold.
    
    Simple strategy that picks the single signal with the highest confidence,
    provided it meets the minimum threshold requirement.
    
    Attributes:
        _min_confidence: Minimum confidence threshold for accepting results
        
    Example:
        >>> strategy = ConfidenceThresholdStrategy(min_confidence=0.8)
    """

    def __init__(self, min_confidence: float = 0.7):
        """Initialize the confidence threshold strategy.
        
        Args:
            min_confidence: Minimum confidence threshold for accepting results
        """
        self._min_confidence = min_confidence

    def aggregate(self, signals: list[IntentSignal]) -> IntentResult:
        """Aggregate signals by selecting highest confidence above threshold.
        
        Args:
            signals: List of IntentSignal objects to aggregate
            
        Returns:
            IntentResult with the highest confidence intent or "unknown" if below threshold
        """
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
    """Main orchestrator for the adaptive intent classification system.
    
    Manages multiple classifiers and aggregates their results using configurable
    strategies. Supports dynamic addition/removal of classifiers at runtime.
    
    Attributes:
        _classifiers: List of registered intent classifiers
        _strategy: Strategy for aggregating classification results
        
    Example:
        >>> lexical = LexicalClassifier("lex", {"greeting": ["hello", "hi"]})
        >>> engine = AdaptiveIntentEngine([lexical])
        >>> result = await engine.classify("hello world")
        >>> print(result.primary_intent)  # "greeting"
    """

    def __init__(
        self,
        classifiers: Sequence[IntentClassifier] | None = None,
        strategy: EnsembleStrategy | None = None,
    ):
        """Initialize the adaptive intent engine.
        
        Args:
            classifiers: Initial list of classifiers to register
            strategy: Strategy for aggregating results (defaults to WeightedVotingStrategy)
        """
        self._classifiers: list[IntentClassifier] = list(classifiers or [])
        self._strategy = strategy or WeightedVotingStrategy()

    def add_classifier(self, classifier: IntentClassifier) -> None:
        """Add a classifier to the engine at runtime.
        
        Args:
            classifier: The IntentClassifier to add
            
        Note:
            Classifiers are automatically sorted by priority after addition
        """
        self._classifiers.append(classifier)
        self._classifiers.sort(key=lambda c: c.priority, reverse=True)

    def remove_classifier(self, name: str) -> None:
        """Remove a classifier from the engine by name.
        
        Args:
            name: Name of the classifier to remove
        """
        self._classifiers = [c for c in self._classifiers if c.name != name]

    async def classify(
        self, text: str, context: dict[str, Any] | None = None
    ) -> IntentResult:
        """Classify input text using all registered classifiers.
        
        Runs all classifiers (both sync and async) and aggregates their results
        using the configured strategy. Handles errors gracefully by logging
        and continuing with other classifiers.
        
        Args:
            text: Input text to classify
            context: Optional context information for classification
            
        Returns:
            IntentResult containing the aggregated classification decision
            
        Note:
            Synchronous and asynchronous classifiers are handled efficiently,
            with async classifiers run concurrently
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
        """Retrieve a classifier by name.
        
        Args:
            name: Name of the classifier to retrieve
            
        Returns:
            The IntentClassifier with the given name, or None if not found
        """
        return next((c for c in self._classifiers if c.name == name), None)

    @property
    def classifier_count(self) -> int:
        """Get the number of registered classifiers.
        
        Returns:
            Number of classifiers currently registered
        """
        return len(self._classifiers)