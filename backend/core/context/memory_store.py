"""
In-Memory Context Store Implementation
High-performance, no external dependencies.
"""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any

from backend.core.models.context_envelope import ContextEnvelope, ContextPayload, ContextState
from .store_interface import ContextStore, ContextQuery

logger = logging.getLogger(__name__)


class InMemoryContextStore(ContextStore):
    """
    Thread-safe in-memory context store with LRU eviction.
    Suitable for single-instance deployments.
    """

    def __init__(self, max_size: int = 1000, auto_cleanup_interval: int = 60):
        self._store: OrderedDict[str, ContextEnvelope] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._cleanup_interval = auto_cleanup_interval
        self._cleanup_task: asyncio.Task | None = None

    async def start_auto_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._auto_cleanup_loop())
            logger.info("Started auto-cleanup task")

    async def stop_auto_cleanup(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped auto-cleanup task")

    async def _auto_cleanup_loop(self) -> None:
        """Background loop for periodic cleanup."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                removed = await self.clear_expired()
                if removed > 0:
                    logger.debug(f"Auto-cleanup removed {removed} expired contexts")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-cleanup error: {e}", exc_info=True)

    async def add(self, envelope: ContextEnvelope) -> str:
        """Add envelope with LRU eviction if needed."""
        async with self._lock:
            envelope_id = envelope.metadata.id

            # Evict oldest if at capacity
            if len(self._store) >= self._max_size and envelope_id not in self._store:
                evicted_id, _ = self._store.popitem(last=False)
                logger.debug(f"Evicted context {evicted_id} (LRU)")

            self._store[envelope_id] = envelope
            self._store.move_to_end(envelope_id)  # Mark as most recent

            logger.debug(
                f"Added context {envelope_id} "
                f"(category={envelope.metadata.category.name}, "
                f"priority={envelope.metadata.priority.name})"
            )

            return envelope_id

    async def get(self, envelope_id: str) -> ContextEnvelope | None:
        """Retrieve and refresh LRU position."""
        async with self._lock:
            envelope = self._store.get(envelope_id)
            if envelope:
                self._store.move_to_end(envelope_id)  # LRU refresh
                envelope.access()  # Update access metadata
            return envelope

    async def update(self, envelope: ContextEnvelope) -> bool:
        """Update existing envelope."""
        async with self._lock:
            envelope_id = envelope.metadata.id
            if envelope_id in self._store:
                self._store[envelope_id] = envelope
                self._store.move_to_end(envelope_id)
                return True
            return False

    async def delete(self, envelope_id: str) -> bool:
        """Remove envelope."""
        async with self._lock:
            if envelope_id in self._store:
                del self._store[envelope_id]
                logger.debug(f"Deleted context {envelope_id}")
                return True
            return False

    async def query(self, query: ContextQuery) -> list[ContextEnvelope]:
        """Execute query on all contexts."""
        async with self._lock:
            contexts = list(self._store.values())

        # Apply query filters/sorting outside lock
        return query.apply(contexts)

    async def get_all(self) -> list[ContextEnvelope]:
        """Retrieve all envelopes (most recent first)."""
        async with self._lock:
            return list(reversed(self._store.values()))

    async def clear_expired(self) -> int:
        """Remove expired and invalidated contexts."""
        now = datetime.utcnow()
        to_remove = []

        async with self._lock:
            for envelope_id, envelope in self._store.items():
                if envelope.is_expired(now) or envelope.state == ContextState.INVALIDATED:
                    to_remove.append(envelope_id)

            for envelope_id in to_remove:
                del self._store[envelope_id]

        if to_remove:
            logger.debug(f"Cleared {len(to_remove)} expired/invalidated contexts")

            # Telemetry: Track expired contexts
            try:
                from backend.core.telemetry.events import get_telemetry
                telemetry = get_telemetry()
                await telemetry.track_event(
                    "follow_up.contexts_expired",
                    {"count": len(to_remove)}
                )
            except:
                pass  # Telemetry is optional

        return len(to_remove)

    async def clear_all(self) -> None:
        """Clear entire store."""
        async with self._lock:
            self._store.clear()
        logger.info("Cleared all contexts from store")

    async def count(self) -> int:
        """Get current count."""
        async with self._lock:
            return len(self._store)

    # Additional optimized methods

    async def get_by_tags(self, *tags: str) -> list[ContextEnvelope]:
        """Fast retrieval by tags."""
        tag_set = set(tags)
        async with self._lock:
            return [
                env
                for env in self._store.values()
                if tag_set.intersection(env.metadata.tags) and env.is_valid()
            ]

    async def get_by_category(self, category: str) -> list[ContextEnvelope]:
        """Fast retrieval by category."""
        async with self._lock:
            return [
                env
                for env in self._store.values()
                if env.metadata.category.name == category.upper() and env.is_valid()
            ]

    def __len__(self) -> int:
        """Synchronous length (for debugging)."""
        return len(self._store)

    async def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        async with self._lock:
            total = len(self._store)
            by_state = {}
            by_category = {}
            avg_relevance = 0.0

            for env in self._store.values():
                # Count by state
                state_name = env.state.name
                by_state[state_name] = by_state.get(state_name, 0) + 1

                # Count by category
                cat_name = env.metadata.category.name
                by_category[cat_name] = by_category.get(cat_name, 0) + 1

                # Sum relevance
                avg_relevance += env.relevance_score()

            if total > 0:
                avg_relevance /= total

            return {
                "total": total,
                "max_size": self._max_size,
                "utilization": total / self._max_size if self._max_size > 0 else 0,
                "by_state": by_state,
                "by_category": by_category,
                "avg_relevance": avg_relevance,
            }


# Register with factory
from .store_interface import ContextStoreFactory

ContextStoreFactory.register("memory", InMemoryContextStore)
