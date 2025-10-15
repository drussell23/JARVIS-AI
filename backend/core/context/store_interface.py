"""
Abstract Context Store Interface
Supports multiple backends: memory, Redis, database, etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, Iterable, TypeVar

from backend.core.models.context_envelope import ContextEnvelope, ContextPayload

T = TypeVar("T", bound=ContextPayload)


class ContextQuery:
    """Fluent query builder for context retrieval."""

    def __init__(self):
        self._filters: list[callable] = []
        self._limit: int | None = None
        self._sort_key: callable | None = None
        self._reverse: bool = True

    def with_category(self, category: str) -> ContextQuery:
        """Filter by category."""
        self._filters.append(lambda env: env.metadata.category.name == category.upper())
        return self

    def with_tag(self, tag: str) -> ContextQuery:
        """Filter by tag presence."""
        self._filters.append(lambda env: tag in env.metadata.tags)
        return self

    def with_min_relevance(self, score: float) -> ContextQuery:
        """Filter by minimum relevance score."""
        self._filters.append(lambda env: env.relevance_score() >= score)
        return self

    def with_state(self, *states: str) -> ContextQuery:
        """Filter by state."""
        state_set = {s.upper() for s in states}
        self._filters.append(lambda env: env.state.name in state_set)
        return self

    def created_after(self, dt: datetime) -> ContextQuery:
        """Filter by creation time."""
        self._filters.append(lambda env: env.metadata.created_at > dt)
        return self

    def sort_by_relevance(self, descending: bool = True) -> ContextQuery:
        """Sort by relevance score."""
        self._sort_key = lambda env: env.relevance_score()
        self._reverse = descending
        return self

    def sort_by_created_at(self, descending: bool = True) -> ContextQuery:
        """Sort by creation time."""
        self._sort_key = lambda env: env.metadata.created_at
        self._reverse = descending
        return self

    def limit(self, n: int) -> ContextQuery:
        """Limit results."""
        self._limit = n
        return self

    def apply(self, contexts: Iterable[ContextEnvelope]) -> list[ContextEnvelope]:
        """Execute query on collection."""
        # Filter
        filtered = contexts
        for f in self._filters:
            filtered = [env for env in filtered if f(env)]

        # Sort
        if self._sort_key:
            filtered = sorted(filtered, key=self._sort_key, reverse=self._reverse)

        # Limit
        if self._limit is not None:
            filtered = filtered[: self._limit]

        return filtered


class ContextStore(ABC, Generic[T]):
    """
    Abstract context store interface.
    Implementations: InMemoryStore, RedisStore, PostgresStore, etc.
    """

    @abstractmethod
    async def add(self, envelope: ContextEnvelope[T]) -> str:
        """
        Add context envelope to store.
        Returns: envelope ID.
        """
        ...

    @abstractmethod
    async def get(self, envelope_id: str) -> ContextEnvelope[T] | None:
        """Retrieve envelope by ID."""
        ...

    @abstractmethod
    async def update(self, envelope: ContextEnvelope[T]) -> bool:
        """Update existing envelope. Returns success status."""
        ...

    @abstractmethod
    async def delete(self, envelope_id: str) -> bool:
        """Remove envelope by ID."""
        ...

    @abstractmethod
    async def query(self, query: ContextQuery) -> list[ContextEnvelope[T]]:
        """Execute fluent query."""
        ...

    @abstractmethod
    async def get_all(self) -> list[ContextEnvelope[T]]:
        """Retrieve all envelopes."""
        ...

    @abstractmethod
    async def clear_expired(self) -> int:
        """Remove expired envelopes. Returns count removed."""
        ...

    @abstractmethod
    async def clear_all(self) -> None:
        """Remove all envelopes."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Total envelope count."""
        ...

    # Advanced methods

    async def get_most_relevant(
        self, query: ContextQuery | None = None, limit: int = 1
    ) -> list[ContextEnvelope[T]]:
        """Get most relevant contexts, optionally filtered by query."""
        if query is None:
            query = ContextQuery()
        return await self.query(query.sort_by_relevance().limit(limit))

    async def get_recent(
        self, seconds: int, limit: int | None = None
    ) -> list[ContextEnvelope[T]]:
        """Get contexts created in last N seconds."""
        cutoff = datetime.utcnow()
        from datetime import timedelta

        cutoff = cutoff - timedelta(seconds=seconds)

        query = ContextQuery().created_after(cutoff).sort_by_created_at()
        if limit:
            query = query.limit(limit)

        return await self.query(query)

    async def mark_consumed(self, envelope_id: str) -> bool:
        """Mark envelope as consumed."""
        env = await self.get(envelope_id)
        if env:
            env.consume()
            return await self.update(env)
        return False

    async def invalidate(self, envelope_id: str, reason: str = "") -> bool:
        """Invalidate envelope."""
        env = await self.get(envelope_id)
        if env:
            env.invalidate(reason)
            return await self.update(env)
        return False


class ContextStoreFactory:
    """Factory for creating context stores based on configuration."""

    _registry: dict[str, type[ContextStore]] = {}

    @classmethod
    def register(cls, name: str, store_class: type[ContextStore]) -> None:
        """Register store implementation."""
        cls._registry[name] = store_class

    @classmethod
    def create(cls, backend: str, **kwargs) -> ContextStore:
        """Create store instance by backend name."""
        store_class = cls._registry.get(backend)
        if not store_class:
            raise ValueError(
                f"Unknown context store backend: {backend}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return store_class(**kwargs)
