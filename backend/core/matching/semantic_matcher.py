"""
Semantic Context Matching Engine
Uses embeddings and semantic similarity to match user input with relevant contexts.
"""
from __future__ import annotations

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from backend.core.models.context_envelope import ContextEnvelope

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MatchScore:
    """Semantic match result."""
    context_id: str
    score: float  # 0.0-1.0
    method: str  # matching method used
    features: dict[str, Any]


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding vector for text."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        ...


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings (text-embedding-3-small/large)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=api_key)
            self._model = model
            self._dim = 1536 if "small" in model else 3072
        except ImportError:
            raise RuntimeError("OpenAI package required. Install: pip install openai")

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding."""
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return np.array(response.data[0].embedding)

    @property
    def dimension(self) -> int:
        return self._dim


class SentenceTransformerProvider(EmbeddingProvider):
    """Local sentence transformers (e.g., all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        except ImportError:
            raise RuntimeError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding (runs in executor to avoid blocking)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._model.encode, text)

    @property
    def dimension(self) -> int:
        return self._dim


class CachedEmbeddingProvider(EmbeddingProvider):
    """Caching wrapper for embedding providers."""

    def __init__(self, provider: EmbeddingProvider, max_cache_size: int = 10000):
        self._provider = provider
        self._cache: dict[str, np.ndarray] = {}
        self._max_cache_size = max_cache_size

    async def embed(self, text: str) -> np.ndarray:
        """Get embedding with caching."""
        cache_key = text.strip().lower()

        if cache_key in self._cache:
            logger.debug(f"Embedding cache hit for: {text[:50]}...")
            return self._cache[cache_key]

        # Generate
        embedding = await self._provider.embed(text)

        # Cache with LRU eviction
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest (first inserted)
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[cache_key] = embedding
        return embedding

    @property
    def dimension(self) -> int:
        return self._provider.dimension


class SemanticMatcher:
    """
    Matches user input to context envelopes using semantic similarity.
    Combines multiple signals: embeddings, keyword overlap, recency, etc.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        similarity_threshold: float = 0.65,
        recency_weight: float = 0.2,
        keyword_weight: float = 0.15,
        use_intelligent_selection: bool = True,
    ):
        self._embedder = embedding_provider
        self._similarity_threshold = similarity_threshold
        self._recency_weight = recency_weight
        self._keyword_weight = keyword_weight
        self.use_intelligent_selection = use_intelligent_selection

    async def _match_with_intelligent_selection(
        self,
        user_input: str,
        candidates: list[ContextEnvelope],
        top_k: int = 5,
    ) -> list[MatchScore]:
        """
        Match using intelligent model selection for embedding generation.
        Note: This primarily helps if we add alternative embedding models.
        Currently uses OpenAI embeddings, so benefit is limited.
        """
        try:
            from backend.core.hybrid_orchestrator import HybridOrchestrator

            orchestrator = HybridOrchestrator()
            if not orchestrator.is_running:
                await orchestrator.start()

            # Build rich context
            context = {
                "task_type": "semantic_matching",
                "embedding_provider": type(self._embedder).__name__ if self._embedder else None,
                "cache_enabled": isinstance(self._embedder, CachedEmbeddingProvider),
                "similarity_threshold": self._similarity_threshold,
                "candidate_count": len(candidates),
                "input_length": len(user_input),
            }

            logger.info(f"Semantic matching with intelligent selection (note: limited benefit with OpenAI embeddings)")

            # For now, we still use the standard embedding provider
            # In the future, intelligent selection could choose between:
            # - OpenAI text-embedding-3-small (fast, good quality)
            # - OpenAI text-embedding-3-large (slower, better quality)
            # - Local sentence-transformers (no API cost, faster for small batches)
            # - Cohere embeddings (alternative API)

            # Generate input embedding (standard approach for now)
            input_embedding = None
            if self._embedder:
                try:
                    input_embedding = await self._embedder.embed(user_input)
                except Exception as e:
                    logger.error(f"Embedding generation failed: {e}", exc_info=True)

            # Score each candidate
            scores: list[MatchScore] = []

            for ctx in candidates:
                # Skip invalid contexts
                if not ctx.is_valid():
                    continue

                # Extract searchable text from context
                search_text = self._extract_searchable_text(ctx)

                # Calculate component scores
                semantic_score = 0.0
                if input_embedding is not None:
                    semantic_score = await self._semantic_similarity(
                        input_embedding, search_text
                    )

                keyword_score = self._keyword_similarity(user_input, search_text)
                recency_score = self._recency_score(ctx)

                # Weighted combination
                if input_embedding is not None:
                    final_score = (
                        0.65 * semantic_score
                        + self._keyword_weight * keyword_score
                        + self._recency_weight * recency_score
                    )
                    method = "semantic+keyword+recency+intelligent"
                else:
                    # Fallback without embeddings
                    final_score = (
                        0.7 * keyword_score + 0.3 * recency_score
                    )
                    method = "keyword+recency+intelligent"

                # Only include if above threshold
                if final_score >= self._similarity_threshold:
                    scores.append(
                        MatchScore(
                            context_id=ctx.metadata.id,
                            score=final_score,
                            method=method,
                            features={
                                "semantic": semantic_score,
                                "keyword": keyword_score,
                                "recency": recency_score,
                                "category": ctx.metadata.category.name,
                            },
                        )
                    )

            # Sort by score descending
            scores.sort(key=lambda s: s.score, reverse=True)

            return scores[:top_k]

        except ImportError:
            logger.warning("Hybrid orchestrator not available, falling back to standard matching")
            raise
        except Exception as e:
            logger.error(f"Error in intelligent selection: {e}")
            raise

    async def match(
        self,
        user_input: str,
        candidates: list[ContextEnvelope],
        top_k: int = 5,
    ) -> list[MatchScore]:
        """
        Find best matching contexts for user input.

        Scoring:
        - Semantic similarity (if embedder available): 65%
        - Keyword overlap: 15%
        - Recency: 20%
        """
        if not candidates:
            return []

        # Try intelligent selection first (limited benefit for embeddings)
        if self.use_intelligent_selection:
            try:
                return await self._match_with_intelligent_selection(user_input, candidates, top_k)
            except Exception as e:
                logger.warning(f"Intelligent selection failed, falling back to standard matching: {e}")

        # Standard matching approach
        # Generate input embedding
        input_embedding = None
        if self._embedder:
            try:
                input_embedding = await self._embedder.embed(user_input)
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}", exc_info=True)

        # Score each candidate
        scores: list[MatchScore] = []

        for ctx in candidates:
            # Skip invalid contexts
            if not ctx.is_valid():
                continue

            # Extract searchable text from context
            search_text = self._extract_searchable_text(ctx)

            # Calculate component scores
            semantic_score = 0.0
            if input_embedding is not None:
                semantic_score = await self._semantic_similarity(
                    input_embedding, search_text
                )

            keyword_score = self._keyword_similarity(user_input, search_text)
            recency_score = self._recency_score(ctx)

            # Weighted combination
            if input_embedding is not None:
                final_score = (
                    0.65 * semantic_score
                    + self._keyword_weight * keyword_score
                    + self._recency_weight * recency_score
                )
                method = "semantic+keyword+recency"
            else:
                # Fallback without embeddings
                final_score = (
                    0.7 * keyword_score + 0.3 * recency_score
                )
                method = "keyword+recency"

            # Only include if above threshold
            if final_score >= self._similarity_threshold:
                scores.append(
                    MatchScore(
                        context_id=ctx.metadata.id,
                        score=final_score,
                        method=method,
                        features={
                            "semantic": semantic_score,
                            "keyword": keyword_score,
                            "recency": recency_score,
                            "category": ctx.metadata.category.name,
                        },
                    )
                )

        # Sort by score descending
        scores.sort(key=lambda s: s.score, reverse=True)

        return scores[:top_k]

    async def find_best_match(
        self, user_input: str, candidates: list[ContextEnvelope]
    ) -> tuple[ContextEnvelope, MatchScore] | None:
        """Find single best matching context."""
        matches = await self.match(user_input, candidates, top_k=1)
        if not matches:
            return None

        best_score = matches[0]
        best_ctx = next(
            (c for c in candidates if c.metadata.id == best_score.context_id), None
        )

        if best_ctx:
            return best_ctx, best_score
        return None

    def _extract_searchable_text(self, ctx: ContextEnvelope) -> str:
        """Extract text content from context for matching."""
        parts = []

        # Add tags
        parts.extend(ctx.metadata.tags)

        # Add category
        parts.append(ctx.metadata.category.name)

        # Extract from payload (duck typing)
        payload = ctx.payload

        if hasattr(payload, "summary"):
            parts.append(payload.summary)

        if hasattr(payload, "question_text"):
            parts.append(payload.question_text)

        if hasattr(payload, "window_type"):
            parts.append(payload.window_type)

        if hasattr(payload, "ocr_text") and payload.ocr_text:
            parts.append(payload.ocr_text[:500])  # Limit length

        return " ".join(str(p) for p in parts)

    async def _semantic_similarity(
        self, input_embedding: np.ndarray, context_text: str
    ) -> float:
        """Cosine similarity between input and context embeddings."""
        if not self._embedder:
            return 0.0

        try:
            context_embedding = await self._embedder.embed(context_text)
            return self._cosine_similarity(input_embedding, context_embedding)
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _keyword_similarity(input_text: str, context_text: str) -> float:
        """Simple keyword overlap score."""
        input_words = set(input_text.lower().split())
        context_words = set(context_text.lower().split())

        if not input_words or not context_words:
            return 0.0

        intersection = input_words.intersection(context_words)
        union = input_words.union(context_words)

        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _recency_score(ctx: ContextEnvelope) -> float:
        """Score based on how recent the context is."""
        from datetime import datetime

        now = datetime.utcnow()
        age_seconds = (now - ctx.metadata.created_at).total_seconds()

        # Exponential decay (half-life of 60 seconds)
        return np.exp(-age_seconds / 60.0)


class HybridMatcher:
    """
    Combines multiple matching strategies.
    Useful when you want to try different approaches and ensemble results.
    """

    def __init__(
        self,
        matchers: list[SemanticMatcher],
        aggregation: str = "max",  # "max", "mean", "weighted"
    ):
        self._matchers = matchers
        self._aggregation = aggregation

    async def match(
        self, user_input: str, candidates: list[ContextEnvelope], top_k: int = 5
    ) -> list[MatchScore]:
        """Run all matchers and aggregate results."""
        import asyncio

        # Run matchers concurrently
        tasks = [m.match(user_input, candidates, top_k * 2) for m in self._matchers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate scores by context ID
        aggregated: dict[str, list[float]] = {}

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Matcher failed: {result}", exc_info=result)
                continue

            for match_score in result:
                if match_score.context_id not in aggregated:
                    aggregated[match_score.context_id] = []
                aggregated[match_score.context_id].append(match_score.score)

        # Combine scores
        final_scores = []
        for ctx_id, scores in aggregated.items():
            if self._aggregation == "max":
                final = max(scores)
            elif self._aggregation == "mean":
                final = sum(scores) / len(scores)
            elif self._aggregation == "weighted":
                # Weight by position in list
                weights = [1.0 / (i + 1) for i in range(len(scores))]
                final = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            else:
                final = max(scores)

            final_scores.append(
                MatchScore(
                    context_id=ctx_id,
                    score=final,
                    method=f"hybrid_{self._aggregation}",
                    features={"component_scores": scores},
                )
            )

        # Sort and limit
        final_scores.sort(key=lambda s: s.score, reverse=True)
        return final_scores[:top_k]
