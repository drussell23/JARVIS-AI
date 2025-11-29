"""
JARVIS macOS Helper - Unified Event Bus

Central event bus for macOS helper layer.
Provides async event routing with filtering, priority handling, and integration
with the AGI OS event stream.

Features:
- Async-first design with zero blocking
- Priority-based event processing
- Event filtering and routing
- Automatic correlation tracking
- Deduplication with configurable windows
- Integration with AGI OS ProactiveEventStream
- Metrics and observability
- Circuit breaker for handler failures

Architecture:
    Events → Queue → Priority Sort → Filter → Dispatch → Handlers
                                               ↓
                                         AGI OS Bridge
                                               ↓
                                      ProactiveEventStream
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union
import heapq

from .event_types import (
    MacOSEvent,
    MacOSEventType,
    MacOSEventPriority,
    MacOSEventHandler,
    EventCategory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Subscription and Handler Types
# =============================================================================

@dataclass
class MacOSEventSubscription:
    """
    Represents a subscription to macOS events.

    Subscriptions can filter by event types, categories, priority,
    and custom filter functions.
    """
    handler: MacOSEventHandler
    subscription_id: str = ""
    event_types: Set[MacOSEventType] = field(default_factory=set)
    categories: Set[EventCategory] = field(default_factory=set)
    min_priority: MacOSEventPriority = MacOSEventPriority.DEBUG
    filter_func: Optional[Callable[[MacOSEvent], bool]] = None
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Circuit breaker for handler failures
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    max_failures: int = 5
    circuit_open: bool = False
    circuit_reset_after: timedelta = field(default_factory=lambda: timedelta(minutes=1))

    def __post_init__(self):
        if not self.subscription_id:
            self.subscription_id = hashlib.md5(
                f"{id(self.handler)}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:8]

    def should_handle(self, event: MacOSEvent) -> bool:
        """Check if this subscription should handle the event."""
        # Check circuit breaker
        if self.circuit_open:
            if (datetime.now() - self.last_failure) > self.circuit_reset_after:
                self.circuit_open = False
                self.failure_count = 0
                logger.info(f"Circuit breaker reset for subscription {self.subscription_id}")
            else:
                return False

        # Check priority
        if event.priority.value < self.min_priority.value:
            return False

        # Check event types (empty means all)
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check categories (empty means all)
        if self.categories and event.category not in self.categories:
            return False

        # Check custom filter
        if self.filter_func and not self.filter_func(event):
            return False

        return True

    def record_failure(self):
        """Record a handler failure."""
        self.failure_count += 1
        self.last_failure = datetime.now()

        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            logger.warning(
                f"Circuit breaker opened for subscription {self.subscription_id} "
                f"after {self.failure_count} failures"
            )

    def record_success(self):
        """Record a successful handler execution."""
        self.failure_count = 0


@dataclass
class PrioritizedEvent:
    """Event wrapper for priority queue ordering."""
    priority: int  # Negated for max-heap behavior
    timestamp: float
    event: MacOSEvent

    def __lt__(self, other: "PrioritizedEvent") -> bool:
        # Higher priority (lower number) comes first
        # For same priority, earlier timestamp comes first
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


# =============================================================================
# Event Bus Statistics
# =============================================================================

@dataclass
class EventBusStats:
    """Statistics for the event bus."""
    events_emitted: int = 0
    events_processed: int = 0
    events_deduplicated: int = 0
    events_bridged_to_agi: int = 0
    handlers_called: int = 0
    handler_errors: int = 0
    handlers_circuit_opened: int = 0
    avg_processing_time_ms: float = 0.0
    queue_high_water_mark: int = 0
    subscriptions_active: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events_emitted": self.events_emitted,
            "events_processed": self.events_processed,
            "events_deduplicated": self.events_deduplicated,
            "events_bridged_to_agi": self.events_bridged_to_agi,
            "handlers_called": self.handlers_called,
            "handler_errors": self.handler_errors,
            "handlers_circuit_opened": self.handlers_circuit_opened,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "queue_high_water_mark": self.queue_high_water_mark,
            "subscriptions_active": self.subscriptions_active,
        }


# =============================================================================
# Main Event Bus
# =============================================================================

class MacOSEventBus:
    """
    Central event bus for macOS helper layer.

    Provides async event routing with:
    - Priority-based processing
    - Event filtering and routing
    - AGI OS integration
    - Deduplication
    - Metrics and observability
    """

    def __init__(
        self,
        enable_agi_bridge: bool = True,
        max_queue_size: int = 10000,
        dedup_window_seconds: float = 2.0,
        max_dedup_entries: int = 500,
        max_history_entries: int = 1000,
    ):
        """
        Initialize the event bus.

        Args:
            enable_agi_bridge: Bridge events to AGI OS ProactiveEventStream
            max_queue_size: Maximum event queue size
            dedup_window_seconds: Time window for deduplication
            max_dedup_entries: Maximum deduplication cache entries
            max_history_entries: Maximum event history entries
        """
        self._enable_agi_bridge = enable_agi_bridge
        self._max_queue_size = max_queue_size
        self._dedup_window = timedelta(seconds=dedup_window_seconds)
        self._max_dedup_entries = max_dedup_entries
        self._max_history_entries = max_history_entries

        # Event queue (priority queue)
        self._event_queue: List[PrioritizedEvent] = []
        self._queue_lock = asyncio.Lock()

        # Subscriptions
        self._subscriptions: Dict[str, MacOSEventSubscription] = {}
        self._type_subscriptions: Dict[MacOSEventType, Set[str]] = {
            event_type: set() for event_type in MacOSEventType
        }
        self._category_subscriptions: Dict[EventCategory, Set[str]] = {
            category: set() for category in EventCategory
        }
        self._global_subscriptions: Set[str] = set()

        # Deduplication
        self._recent_events: OrderedDict[str, datetime] = OrderedDict()

        # Event history for correlation
        self._event_history: OrderedDict[str, MacOSEvent] = OrderedDict()
        self._correlation_map: Dict[str, Set[str]] = {}  # correlation_id -> event_ids

        # AGI OS bridge (lazy loaded)
        self._agi_event_stream = None

        # Processing state
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._process_event = asyncio.Event()

        # Statistics
        self._stats = EventBusStats()
        self._processing_times: List[float] = []

        logger.info(
            f"MacOSEventBus initialized (agi_bridge={enable_agi_bridge}, "
            f"queue_size={max_queue_size})"
        )

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """Start the event bus processor."""
        if self._running:
            logger.warning("Event bus already running")
            return

        self._running = True
        self._processor_task = asyncio.create_task(
            self._event_processor(),
            name="macos_event_bus_processor"
        )

        # Initialize AGI bridge if enabled
        if self._enable_agi_bridge:
            await self._init_agi_bridge()

        logger.info("MacOSEventBus started")

    async def stop(self) -> None:
        """Stop the event bus processor."""
        if not self._running:
            return

        self._running = False

        # Wake up processor to exit
        self._process_event.set()

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.info("MacOSEventBus stopped")

    async def _init_agi_bridge(self) -> None:
        """Initialize bridge to AGI OS ProactiveEventStream."""
        try:
            from agi_os.proactive_event_stream import get_event_stream
            self._agi_event_stream = await get_event_stream()
            logger.info("AGI OS event stream bridge initialized")
        except ImportError:
            logger.warning("AGI OS event stream not available - bridge disabled")
            self._enable_agi_bridge = False
        except Exception as e:
            logger.error(f"Failed to initialize AGI bridge: {e}")
            self._enable_agi_bridge = False

    # =========================================================================
    # Event Emission
    # =========================================================================

    async def emit(
        self,
        event: MacOSEvent,
        deduplicate: bool = True,
        bridge_to_agi: bool = True,
    ) -> bool:
        """
        Emit an event to the bus.

        Args:
            event: The event to emit
            deduplicate: Whether to skip duplicate events
            bridge_to_agi: Whether to bridge to AGI OS

        Returns:
            True if event was emitted, False if deduplicated or queue full
        """
        # Check for duplicates
        if deduplicate:
            dedup_key = self._get_dedup_key(event)
            now = datetime.now()

            if dedup_key in self._recent_events:
                last_time = self._recent_events[dedup_key]
                if (now - last_time) < self._dedup_window:
                    self._stats.events_deduplicated += 1
                    logger.debug(f"Deduplicated event: {event.event_type.value}")
                    return False

            # Record for deduplication
            self._recent_events[dedup_key] = now
            self._trim_dedup_cache()

        # Check queue capacity
        async with self._queue_lock:
            if len(self._event_queue) >= self._max_queue_size:
                logger.warning("Event queue full, dropping event")
                return False

            # Add to priority queue (negate priority for max-heap behavior)
            prioritized = PrioritizedEvent(
                priority=-event.priority.value,
                timestamp=event.timestamp.timestamp(),
                event=event,
            )
            heapq.heappush(self._event_queue, prioritized)

            # Update high water mark
            queue_size = len(self._event_queue)
            if queue_size > self._stats.queue_high_water_mark:
                self._stats.queue_high_water_mark = queue_size

        # Add to history
        self._event_history[event.event_id] = event
        self._trim_history()

        # Track correlation
        if event.correlation_id:
            if event.correlation_id not in self._correlation_map:
                self._correlation_map[event.correlation_id] = set()
            self._correlation_map[event.correlation_id].add(event.event_id)

        self._stats.events_emitted += 1

        # Signal processor
        self._process_event.set()

        # Bridge to AGI if needed
        if bridge_to_agi and self._enable_agi_bridge and event.requires_agi_processing:
            await self._bridge_to_agi(event)

        logger.debug(
            f"Event emitted: {event.event_type.value} from {event.source} "
            f"(priority={event.priority.name})"
        )

        return True

    async def _bridge_to_agi(self, event: MacOSEvent) -> None:
        """Bridge a macOS event to AGI OS ProactiveEventStream."""
        if not self._agi_event_stream:
            return

        try:
            from agi_os.proactive_event_stream import AGIEvent, EventType, EventPriority

            # Map macOS event to AGI event
            agi_event_type = self._map_to_agi_event_type(event)
            if not agi_event_type:
                return

            agi_priority = EventPriority(min(event.priority.value, 5))

            agi_event = AGIEvent(
                event_type=agi_event_type,
                source=f"macos_helper.{event.source}",
                data=event.data,
                priority=agi_priority,
                correlation_id=event.correlation_id,
                requires_narration=event.requires_voice_narration,
                metadata={
                    "macos_event_id": event.event_id,
                    "macos_event_type": event.event_type.value,
                    "category": event.category.value,
                }
            )

            await self._agi_event_stream.emit(agi_event, deduplicate=False)
            self._stats.events_bridged_to_agi += 1

        except Exception as e:
            logger.error(f"Failed to bridge event to AGI OS: {e}")

    def _map_to_agi_event_type(self, event: MacOSEvent):
        """Map macOS event type to AGI OS event type."""
        try:
            from agi_os.proactive_event_stream import EventType

            # Map common event types
            mapping = {
                MacOSEventType.NOTIFICATION_RECEIVED: EventType.NOTIFICATION_DETECTED,
                MacOSEventType.APP_LAUNCHED: EventType.APP_CHANGED,
                MacOSEventType.APP_ACTIVATED: EventType.APP_CHANGED,
                MacOSEventType.SPACE_CHANGED: EventType.CONTENT_CHANGED,
                MacOSEventType.CALENDAR_EVENT_UPCOMING: EventType.MEETING_DETECTED,
                MacOSEventType.SCREEN_LOCKED: EventType.SYSTEM_STOPPED,
                MacOSEventType.SCREEN_UNLOCKED: EventType.SYSTEM_STARTED,
                MacOSEventType.HELPER_ERROR: EventType.WARNING_DETECTED,
            }

            return mapping.get(event.event_type)
        except ImportError:
            return None

    # =========================================================================
    # Subscription Methods
    # =========================================================================

    def subscribe(
        self,
        handler: MacOSEventHandler,
        event_types: Optional[Union[MacOSEventType, List[MacOSEventType]]] = None,
        categories: Optional[Union[EventCategory, List[EventCategory]]] = None,
        min_priority: MacOSEventPriority = MacOSEventPriority.DEBUG,
        filter_func: Optional[Callable[[MacOSEvent], bool]] = None,
        description: str = "",
    ) -> str:
        """
        Subscribe to events.

        Args:
            handler: Async handler function
            event_types: Event types to subscribe to (None for all)
            categories: Categories to subscribe to (None for all)
            min_priority: Minimum priority to receive
            filter_func: Optional custom filter function
            description: Description for debugging

        Returns:
            Subscription ID for unsubscribing
        """
        # Normalize event types
        if event_types is None:
            types_set = set()
        elif isinstance(event_types, MacOSEventType):
            types_set = {event_types}
        else:
            types_set = set(event_types)

        # Normalize categories
        if categories is None:
            cats_set = set()
        elif isinstance(categories, EventCategory):
            cats_set = {categories}
        else:
            cats_set = set(categories)

        subscription = MacOSEventSubscription(
            handler=handler,
            event_types=types_set,
            categories=cats_set,
            min_priority=min_priority,
            filter_func=filter_func,
            description=description,
        )

        self._subscriptions[subscription.subscription_id] = subscription

        # Register in type-specific maps
        if types_set:
            for event_type in types_set:
                self._type_subscriptions[event_type].add(subscription.subscription_id)
        elif cats_set:
            for category in cats_set:
                self._category_subscriptions[category].add(subscription.subscription_id)
        else:
            self._global_subscriptions.add(subscription.subscription_id)

        self._stats.subscriptions_active = len(self._subscriptions)

        logger.debug(
            f"Subscription added: {subscription.subscription_id} "
            f"({description or 'no description'})"
        )

        return subscription.subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if unsubscribed, False if not found
        """
        if subscription_id not in self._subscriptions:
            return False

        subscription = self._subscriptions.pop(subscription_id)

        # Remove from type maps
        for event_type in subscription.event_types:
            self._type_subscriptions[event_type].discard(subscription_id)
        for category in subscription.categories:
            self._category_subscriptions[category].discard(subscription_id)
        self._global_subscriptions.discard(subscription_id)

        self._stats.subscriptions_active = len(self._subscriptions)

        logger.debug(f"Subscription removed: {subscription_id}")
        return True

    # =========================================================================
    # Event Processing
    # =========================================================================

    async def _event_processor(self) -> None:
        """Background task that processes events."""
        while self._running:
            try:
                # Wait for events with timeout
                try:
                    await asyncio.wait_for(
                        self._process_event.wait(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                self._process_event.clear()

                # Process all queued events
                while self._running:
                    async with self._queue_lock:
                        if not self._event_queue:
                            break
                        prioritized = heapq.heappop(self._event_queue)

                    event = prioritized.event
                    start_time = asyncio.get_event_loop().time()

                    await self._process_single_event(event)

                    # Track processing time
                    elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                    self._processing_times.append(elapsed_ms)
                    if len(self._processing_times) > 100:
                        self._processing_times = self._processing_times[-100:]
                    self._stats.avg_processing_time_ms = sum(self._processing_times) / len(self._processing_times)

                    self._stats.events_processed += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Event processor error: {e}")

    async def _process_single_event(self, event: MacOSEvent) -> None:
        """Process a single event by calling relevant handlers."""
        # Find relevant subscriptions
        relevant_subs = self._get_relevant_subscriptions(event)

        # Call handlers concurrently
        tasks = []
        for sub_id in relevant_subs:
            subscription = self._subscriptions.get(sub_id)
            if subscription and subscription.should_handle(event):
                tasks.append(self._call_handler(subscription, event))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _call_handler(
        self,
        subscription: MacOSEventSubscription,
        event: MacOSEvent
    ) -> None:
        """Call a single handler with error handling."""
        try:
            await subscription.handler(event)
            subscription.record_success()
            self._stats.handlers_called += 1
        except Exception as e:
            subscription.record_failure()
            self._stats.handler_errors += 1
            if subscription.circuit_open:
                self._stats.handlers_circuit_opened += 1
            logger.error(
                f"Handler error for {event.event_type.value} "
                f"(subscription={subscription.subscription_id}): {e}"
            )

    def _get_relevant_subscriptions(self, event: MacOSEvent) -> Set[str]:
        """Get subscriptions relevant to an event."""
        relevant = set()

        # Type-specific subscriptions
        relevant.update(self._type_subscriptions.get(event.event_type, set()))

        # Category-specific subscriptions
        relevant.update(self._category_subscriptions.get(event.category, set()))

        # Global subscriptions
        relevant.update(self._global_subscriptions)

        return relevant

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _get_dedup_key(self, event: MacOSEvent) -> str:
        """Generate deduplication key for an event."""
        # Key based on type, source, and significant data
        significant_data = {
            k: v for k, v in sorted(event.data.items())
            if k not in ["timestamp", "event_id"]
        }
        return hashlib.md5(
            f"{event.event_type.value}{event.source}{significant_data}".encode()
        ).hexdigest()[:16]

    def _trim_dedup_cache(self) -> None:
        """Trim deduplication cache."""
        now = datetime.now()

        # Remove expired entries
        expired = [
            key for key, time in self._recent_events.items()
            if now - time > self._dedup_window
        ]
        for key in expired:
            del self._recent_events[key]

        # Trim to max size
        while len(self._recent_events) > self._max_dedup_entries:
            self._recent_events.popitem(last=False)

    def _trim_history(self) -> None:
        """Trim event history."""
        while len(self._event_history) > self._max_history_entries:
            old_id, _ = self._event_history.popitem(last=False)
            # Clean up correlation map
            for corr_id, event_ids in list(self._correlation_map.items()):
                event_ids.discard(old_id)
                if not event_ids:
                    del self._correlation_map[corr_id]

    def get_correlated_events(self, correlation_id: str) -> List[MacOSEvent]:
        """Get events with the same correlation ID."""
        event_ids = self._correlation_map.get(correlation_id, set())
        return [
            self._event_history[eid]
            for eid in event_ids
            if eid in self._event_history
        ]

    def get_recent_events(
        self,
        event_type: Optional[MacOSEventType] = None,
        category: Optional[EventCategory] = None,
        source: Optional[str] = None,
        minutes: int = 5,
    ) -> List[MacOSEvent]:
        """Get recent events with optional filtering."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        events = []

        for event in reversed(self._event_history.values()):
            if event.timestamp < cutoff:
                break

            if event_type and event.event_type != event_type:
                continue
            if category and event.category != category:
                continue
            if source and event.source != source:
                continue

            events.append(event)

        return events

    def create_correlation_id(self) -> str:
        """Create a new correlation ID for linking events."""
        return hashlib.md5(
            f"{datetime.now().isoformat()}{id(self)}".encode()
        ).hexdigest()[:12]

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self._stats.to_dict(),
            "queue_size": len(self._event_queue),
            "history_size": len(self._event_history),
            "dedup_cache_size": len(self._recent_events),
            "running": self._running,
            "agi_bridge_enabled": self._enable_agi_bridge,
        }


# =============================================================================
# Singleton Pattern
# =============================================================================

_event_bus: Optional[MacOSEventBus] = None


async def get_macos_event_bus(
    enable_agi_bridge: bool = True,
    auto_start: bool = True,
) -> MacOSEventBus:
    """
    Get the global macOS event bus instance.

    Args:
        enable_agi_bridge: Bridge events to AGI OS
        auto_start: Automatically start the bus if not running

    Returns:
        The MacOSEventBus singleton
    """
    global _event_bus

    if _event_bus is None:
        _event_bus = MacOSEventBus(enable_agi_bridge=enable_agi_bridge)

    if auto_start and not _event_bus._running:
        await _event_bus.start()

    return _event_bus


async def stop_macos_event_bus() -> None:
    """Stop the global macOS event bus."""
    global _event_bus

    if _event_bus is not None:
        await _event_bus.stop()
        _event_bus = None
