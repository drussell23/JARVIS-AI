"""
Abstract Context Store Interface

This module provides the abstract interface and query builder for context storage systems.
Supports multiple backends including memory, Redis, database, and other storage solutions.

The module defines:
- ContextQuery: Fluent query builder for filtering and sorting contexts
- ContextStore: Abstract base class for context storage implementations
- ContextStoreFactory: Factory pattern for creating store instances

Example:
    >>> store = ContextStoreFactory.create('memory')
    >>> query = ContextQuery().with_category('CONVERSATION').limit(10)
    >>> results = await store.query(query)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, Iterable, TypeVar

from backend.core.models.context_envelope import ContextEnvelope, ContextPayload

T = TypeVar("T", bound=ContextPayload)


class ContextQuery:
    """Fluent query builder for context retrieval.
    
    Provides a chainable interface for building complex queries to filter,
    sort, and limit context envelopes. Supports filtering by category, tags,
    relevance scores, state, and creation time.
    
    Attributes:
        _filters: List of filter functions to apply to contexts
        _limit: Maximum number of results to return
        _sort_key: Function to extract sort key from context envelope
        _reverse: Whether to sort in descending order
    
    Example:
        >>> query = (ContextQuery()
        ...     .with_category('CONVERSATION')
        ...     .with_min_relevance(0.8)
        ...     .sort_by_relevance()
        ...     .limit(5))
        >>> results = query.apply(contexts)
    """

    def __init__(self) -> None:
        """Initialize empty query builder."""
        self._filters: list[callable] = []
        self._limit: int | None = None
        self._sort_key: callable | None = None
        self._reverse: bool = True

    def with_category(self, category: str) -> ContextQuery:
        """Filter contexts by category.
        
        Args:
            category: Category name to filter by (case-insensitive)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> query = ContextQuery().with_category('conversation')
        """
        self._filters.append(lambda env: env.metadata.category.name == category.upper())
        return self

    def with_tag(self, tag: str) -> ContextQuery:
        """Filter contexts by tag presence.
        
        Args:
            tag: Tag that must be present in context metadata
            
        Returns:
            Self for method chaining
            
        Example:
            >>> query = ContextQuery().with_tag('important')
        """
        self._filters.append(lambda env: tag in env.metadata.tags)
        return self

    def with_min_relevance(self, score: float) -> ContextQuery:
        """Filter contexts by minimum relevance score.
        
        Args:
            score: Minimum relevance score (0.0 to 1.0)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> query = ContextQuery().with_min_relevance(0.7)
        """
        self._filters.append(lambda env: env.relevance_score() >= score)
        return self

    def with_state(self, *states: str) -> ContextQuery:
        """Filter contexts by state.
        
        Args:
            *states: One or more state names to match (case-insensitive)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> query = ContextQuery().with_state('active', 'pending')
        """
        state_set = {s.upper() for s in states}
        self._filters.append(lambda env: env.state.name in state_set)
        return self

    def created_after(self, dt: datetime) -> ContextQuery:
        """Filter contexts created after specified time.
        
        Args:
            dt: Minimum creation datetime
            
        Returns:
            Self for method chaining
            
        Example:
            >>> from datetime import datetime, timedelta
            >>> cutoff = datetime.utcnow() - timedelta(hours=1)
            >>> query = ContextQuery().created_after(cutoff)
        """
        self._filters.append(lambda env: env.metadata.created_at > dt)
        return self

    def sort_by_relevance(self, descending: bool = True) -> ContextQuery:
        """Sort contexts by relevance score.
        
        Args:
            descending: Whether to sort in descending order (highest first)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> query = ContextQuery().sort_by_relevance(descending=False)
        """
        self._sort_key = lambda env: env.relevance_score()
        self._reverse = descending
        return self

    def sort_by_created_at(self, descending: bool = True) -> ContextQuery:
        """Sort contexts by creation time.
        
        Args:
            descending: Whether to sort in descending order (newest first)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> query = ContextQuery().sort_by_created_at(descending=False)
        """
        self._sort_key = lambda env: env.metadata.created_at
        self._reverse = descending
        return self

    def limit(self, n: int) -> ContextQuery:
        """Limit number of results returned.
        
        Args:
            n: Maximum number of results to return
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If n is negative
            
        Example:
            >>> query = ContextQuery().limit(10)
        """
        if n < 0:
            raise ValueError("Limit must be non-negative")
        self._limit = n
        return self

    def apply(self, contexts: Iterable[ContextEnvelope]) -> list[ContextEnvelope]:
        """Execute query on collection of contexts.
        
        Applies all filters, sorting, and limiting in sequence.
        
        Args:
            contexts: Iterable of context envelopes to query
            
        Returns:
            List of filtered, sorted, and limited context envelopes
            
        Example:
            >>> contexts = [env1, env2, env3]
            >>> query = ContextQuery().with_category('test').limit(2)
            >>> results = query.apply(contexts)
        """
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
    """Abstract context store interface.
    
    Defines the contract for context storage implementations. Supports
    CRUD operations, querying, and advanced context management features.
    
    Type Parameters:
        T: Type of context payload, must be bound to ContextPayload
    
    Implementations include:
        - InMemoryStore: Fast in-memory storage
        - RedisStore: Distributed Redis-based storage  
        - PostgresStore: Persistent database storage
    
    Example:
        >>> store = ContextStoreFactory.create('memory')
        >>> envelope_id = await store.add(envelope)
        >>> retrieved = await store.get(envelope_id)
    """

    @abstractmethod
    async def add(self, envelope: ContextEnvelope[T]) -> str:
        """Add context envelope to store.
        
        Args:
            envelope: Context envelope to store
            
        Returns:
            Unique identifier for the stored envelope
            
        Raises:
            StorageError: If storage operation fails
            
        Example:
            >>> envelope_id = await store.add(my_envelope)
        """
        ...

    @abstractmethod
    async def get(self, envelope_id: str) -> ContextEnvelope[T] | None:
        """Retrieve envelope by ID.
        
        Args:
            envelope_id: Unique identifier of envelope to retrieve
            
        Returns:
            Context envelope if found, None otherwise
            
        Example:
            >>> envelope = await store.get('envelope_123')
            >>> if envelope:
            ...     print(f"Found: {envelope.metadata.category}")
        """
        ...

    @abstractmethod
    async def update(self, envelope: ContextEnvelope[T]) -> bool:
        """Update existing envelope.
        
        Args:
            envelope: Modified envelope to update
            
        Returns:
            True if update succeeded, False if envelope not found
            
        Raises:
            StorageError: If storage operation fails
            
        Example:
            >>> envelope.metadata.tags.add('updated')
            >>> success = await store.update(envelope)
        """
        ...

    @abstractmethod
    async def delete(self, envelope_id: str) -> bool:
        """Remove envelope by ID.
        
        Args:
            envelope_id: Unique identifier of envelope to delete
            
        Returns:
            True if deletion succeeded, False if envelope not found
            
        Example:
            >>> deleted = await store.delete('envelope_123')
        """
        ...

    @abstractmethod
    async def query(self, query: ContextQuery) -> list[ContextEnvelope[T]]:
        """Execute fluent query.
        
        Args:
            query: ContextQuery instance with filters and sorting
            
        Returns:
            List of matching context envelopes
            
        Example:
            >>> query = ContextQuery().with_category('CONVERSATION').limit(5)
            >>> results = await store.query(query)
        """
        ...

    @abstractmethod
    async def get_all(self) -> list[ContextEnvelope[T]]:
        """Retrieve all envelopes.
        
        Returns:
            List of all context envelopes in store
            
        Note:
            Use with caution on large datasets. Consider using query() with
            pagination for better performance.
            
        Example:
            >>> all_contexts = await store.get_all()
        """
        ...

    @abstractmethod
    async def clear_expired(self) -> int:
        """Remove expired envelopes.
        
        Removes all envelopes that have exceeded their TTL or are marked
        as expired based on their state and metadata.
        
        Returns:
            Number of envelopes removed
            
        Example:
            >>> removed_count = await store.clear_expired()
            >>> print(f"Cleaned up {removed_count} expired contexts")
        """
        ...

    @abstractmethod
    async def clear_all(self) -> None:
        """Remove all envelopes.
        
        Completely empties the context store. Use with extreme caution.
        
        Example:
            >>> await store.clear_all()  # Removes everything!
        """
        ...

    @abstractmethod
    async def count(self) -> int:
        """Get total envelope count.
        
        Returns:
            Total number of envelopes in store
            
        Example:
            >>> total = await store.count()
            >>> print(f"Store contains {total} contexts")
        """
        ...

    # Advanced methods

    async def get_most_relevant(
        self, query: ContextQuery | None = None, limit: int = 1
    ) -> list[ContextEnvelope[T]]:
        """Get most relevant contexts, optionally filtered by query.
        
        Args:
            query: Optional query to filter contexts before ranking
            limit: Maximum number of results to return
            
        Returns:
            List of most relevant context envelopes
            
        Example:
            >>> # Get top 3 relevant conversation contexts
            >>> query = ContextQuery().with_category('CONVERSATION')
            >>> top_contexts = await store.get_most_relevant(query, limit=3)
        """
        if query is None:
            query = ContextQuery()
        return await self.query(query.sort_by_relevance().limit(limit))

    async def get_recent(
        self, seconds: int, limit: int | None = None
    ) -> list[ContextEnvelope[T]]:
        """Get contexts created in last N seconds.
        
        Args:
            seconds: Number of seconds to look back
            limit: Optional maximum number of results
            
        Returns:
            List of recently created context envelopes
            
        Example:
            >>> # Get contexts from last 5 minutes
            >>> recent = await store.get_recent(300, limit=10)
        """
        cutoff = datetime.utcnow()
        from datetime import timedelta

        cutoff = cutoff - timedelta(seconds=seconds)

        query = ContextQuery().created_after(cutoff).sort_by_created_at()
        if limit:
            query = query.limit(limit)

        return await self.query(query)

    async def mark_consumed(self, envelope_id: str) -> bool:
        """Mark envelope as consumed.
        
        Updates the envelope state to indicate it has been processed
        or consumed by the system.
        
        Args:
            envelope_id: ID of envelope to mark as consumed
            
        Returns:
            True if successfully marked, False if envelope not found
            
        Example:
            >>> success = await store.mark_consumed('envelope_123')
        """
        env = await self.get(envelope_id)
        if env:
            env.consume()
            return await self.update(env)
        return False

    async def invalidate(self, envelope_id: str, reason: str = "") -> bool:
        """Invalidate envelope.
        
        Marks the envelope as invalid, preventing it from being used
        in future operations.
        
        Args:
            envelope_id: ID of envelope to invalidate
            reason: Optional reason for invalidation
            
        Returns:
            True if successfully invalidated, False if envelope not found
            
        Example:
            >>> success = await store.invalidate('envelope_123', 'Outdated data')
        """
        env = await self.get(envelope_id)
        if env:
            env.invalidate(reason)
            return await self.update(env)
        return False


class ContextStoreFactory:
    """Factory for creating context stores based on configuration.
    
    Implements the factory pattern to create appropriate context store
    instances based on backend type. Supports registration of custom
    store implementations.
    
    Attributes:
        _registry: Dictionary mapping backend names to store classes
        
    Example:
        >>> # Register custom store
        >>> ContextStoreFactory.register('custom', MyCustomStore)
        >>> 
        >>> # Create store instance
        >>> store = ContextStoreFactory.create('redis', host='localhost')
    """

    _registry: dict[str, type[ContextStore]] = {}

    @classmethod
    def register(cls, name: str, store_class: type[ContextStore]) -> None:
        """Register store implementation.
        
        Args:
            name: Backend name identifier
            store_class: Context store class to register
            
        Raises:
            TypeError: If store_class is not a ContextStore subclass
            
        Example:
            >>> class MyStore(ContextStore):
            ...     pass
            >>> ContextStoreFactory.register('mystore', MyStore)
        """
        if not issubclass(store_class, ContextStore):
            raise TypeError(f"Store class must inherit from ContextStore")
        cls._registry[name] = store_class

    @classmethod
    def create(cls, backend: str, **kwargs: Any) -> ContextStore:
        """Create store instance by backend name.
        
        Args:
            backend: Name of registered backend
            **kwargs: Configuration parameters passed to store constructor
            
        Returns:
            Configured context store instance
            
        Raises:
            ValueError: If backend is not registered
            
        Example:
            >>> # Create memory store
            >>> store = ContextStoreFactory.create('memory')
            >>> 
            >>> # Create Redis store with config
            >>> store = ContextStoreFactory.create('redis', 
            ...     host='localhost', port=6379, db=0)
        """
        store_class = cls._registry.get(backend)
        if not store_class:
            raise ValueError(
                f"Unknown context store backend: {backend}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return store_class(**kwargs)