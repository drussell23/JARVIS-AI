"""
Comprehensive Telemetry and Observability Framework
Tracks all follow-up and context operations with structured events.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event categories."""
    # Context lifecycle
    CONTEXT_CREATED = "context.created"
    CONTEXT_ACCESSED = "context.accessed"
    CONTEXT_CONSUMED = "context.consumed"
    CONTEXT_EXPIRED = "context.expired"
    CONTEXT_INVALIDATED = "context.invalidated"

    # Intent detection
    INTENT_DETECTED = "intent.detected"
    INTENT_CLASSIFICATION_FAILED = "intent.classification_failed"

    # Routing
    ROUTE_MATCHED = "route.matched"
    ROUTE_FAILED = "route.failed"
    ROUTE_NO_MATCH = "route.no_match"

    # Follow-up handling
    FOLLOWUP_INITIATED = "followup.initiated"
    FOLLOWUP_RESOLVED = "followup.resolved"
    FOLLOWUP_CONTEXT_MISSING = "followup.context_missing"

    # Vision operations
    VISION_SNAPSHOT_TAKEN = "vision.snapshot_taken"
    VISION_OCR_EXECUTED = "vision.ocr_executed"
    VISION_ANALYSIS_COMPLETED = "vision.analysis_completed"

    # Matching
    SEMANTIC_MATCH_EXECUTED = "semantic.match_executed"
    SEMANTIC_MATCH_FOUND = "semantic.match_found"
    SEMANTIC_MATCH_NONE = "semantic.match_none"

    # Performance
    LATENCY_RECORDED = "latency.recorded"
    ERROR_OCCURRED = "error.occurred"


@dataclass(slots=True)
class TelemetryEvent:
    """Base telemetry event."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    user_id: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "properties": self.properties,
            "metrics": self.metrics,
            "tags": self.tags,
        }


class EventSink(Protocol):
    """Protocol for event sinks (loggers, metrics backends, etc.)."""

    async def emit(self, event: TelemetryEvent) -> None:
        """Emit event to sink."""
        ...


class LoggingEventSink:
    """Emit events to Python logging."""

    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger or logging.getLogger("telemetry")

    async def emit(self, event: TelemetryEvent) -> None:
        """Log event as structured JSON."""
        import json

        self._logger.info(
            f"[{event.event_type.value}] {json.dumps(event.to_dict(), default=str)}"
        )


class PrometheusEventSink:
    """Emit metrics to Prometheus (requires prometheus_client)."""

    def __init__(self):
        try:
            from prometheus_client import Counter, Histogram, Gauge

            # Define metrics
            self._event_counter = Counter(
                "jarvis_events_total",
                "Total events emitted",
                ["event_type"],
            )

            self._latency_histogram = Histogram(
                "jarvis_operation_latency_seconds",
                "Operation latency",
                ["operation"],
            )

            self._context_gauge = Gauge(
                "jarvis_active_contexts",
                "Number of active contexts",
            )

        except ImportError:
            raise RuntimeError(
                "Prometheus support requires prometheus_client. "
                "Install: pip install prometheus-client"
            )

    async def emit(self, event: TelemetryEvent) -> None:
        """Update Prometheus metrics."""
        # Increment event counter
        self._event_counter.labels(event_type=event.event_type.value).inc()

        # Record latency if present
        if "latency_seconds" in event.metrics:
            operation = event.properties.get("operation", "unknown")
            self._latency_histogram.labels(operation=operation).observe(
                event.metrics["latency_seconds"]
            )

        # Update context gauge
        if event.event_type == EventType.CONTEXT_CREATED:
            self._context_gauge.inc()
        elif event.event_type in (
            EventType.CONTEXT_CONSUMED,
            EventType.CONTEXT_EXPIRED,
            EventType.CONTEXT_INVALIDATED,
        ):
            self._context_gauge.dec()


class OpenTelemetryEventSink:
    """Emit events to OpenTelemetry."""

    def __init__(self):
        try:
            from opentelemetry import trace, metrics

            self._tracer = trace.get_tracer(__name__)
            self._meter = metrics.get_meter(__name__)

            # Create instruments
            self._event_counter = self._meter.create_counter(
                "jarvis.events.total",
                description="Total events emitted",
            )

        except ImportError:
            raise RuntimeError(
                "OpenTelemetry support requires opentelemetry-api. "
                "Install: pip install opentelemetry-api opentelemetry-sdk"
            )

    async def emit(self, event: TelemetryEvent) -> None:
        """Emit to OpenTelemetry."""
        # Record event
        self._event_counter.add(
            1,
            attributes={
                "event_type": event.event_type.value,
                **event.properties,
            },
        )


class InMemoryEventSink:
    """Store events in memory for testing/debugging."""

    def __init__(self, max_events: int = 10000):
        self._events: list[TelemetryEvent] = []
        self._max_events = max_events

    async def emit(self, event: TelemetryEvent) -> None:
        """Store event."""
        self._events.append(event)

        # LRU eviction
        if len(self._events) > self._max_events:
            self._events.pop(0)

    def get_events(
        self, event_type: EventType | None = None, limit: int | None = None
    ) -> list[TelemetryEvent]:
        """Retrieve events."""
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if limit:
            events = events[-limit:]

        return events

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()

    @property
    def count(self) -> int:
        return len(self._events)


class TelemetryManager:
    """
    Central telemetry manager.
    Coordinates multiple sinks and provides high-level API.
    """

    def __init__(self):
        self._sinks: list[EventSink] = []
        self._default_session_id: str | None = None
        self._default_user_id: str | None = None
        self._enabled = True

    def add_sink(self, sink: EventSink) -> None:
        """Register event sink."""
        self._sinks.append(sink)
        logger.info(f"Added telemetry sink: {type(sink).__name__}")

    def set_session(self, session_id: str, user_id: str | None = None) -> None:
        """Set default session/user for subsequent events."""
        self._default_session_id = session_id
        self._default_user_id = user_id

    def enable(self) -> None:
        """Enable telemetry."""
        self._enabled = True

    def disable(self) -> None:
        """Disable telemetry."""
        self._enabled = False

    async def emit(
        self,
        event_type: EventType,
        properties: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Emit event to all sinks."""
        if not self._enabled:
            return

        event = TelemetryEvent(
            event_type=event_type,
            session_id=session_id or self._default_session_id,
            user_id=user_id or self._default_user_id,
            properties=properties or {},
            metrics=metrics or {},
            tags=tags or [],
        )

        # Emit to all sinks concurrently
        import asyncio

        tasks = [sink.emit(event) for sink in self._sinks]
        await asyncio.gather(*tasks, return_exceptions=True)

    # Convenience methods for common events

    async def track_context_created(
        self,
        context_id: str,
        category: str,
        priority: str,
        ttl_seconds: int,
        **extra,
    ) -> None:
        """Track context creation."""
        await self.emit(
            EventType.CONTEXT_CREATED,
            properties={
                "context_id": context_id,
                "category": category,
                "priority": priority,
                "ttl_seconds": ttl_seconds,
                **extra,
            },
        )

    async def track_context_accessed(self, context_id: str, access_count: int) -> None:
        """Track context access."""
        await self.emit(
            EventType.CONTEXT_ACCESSED,
            properties={"context_id": context_id},
            metrics={"access_count": float(access_count)},
        )

    async def track_intent_detected(
        self, intent_label: str, confidence: float, classifiers: list[str]
    ) -> None:
        """Track intent detection."""
        await self.emit(
            EventType.INTENT_DETECTED,
            properties={
                "intent_label": intent_label,
                "classifiers": classifiers,
            },
            metrics={"confidence": confidence},
        )

    async def track_route_matched(
        self, intent_label: str, handler_name: str, latency_ms: float
    ) -> None:
        """Track successful routing."""
        await self.emit(
            EventType.ROUTE_MATCHED,
            properties={
                "intent_label": intent_label,
                "handler_name": handler_name,
            },
            metrics={"latency_ms": latency_ms},
        )

    async def track_followup_resolved(
        self,
        context_id: str,
        window_type: str,
        response_type: str,
        latency_ms: float,
    ) -> None:
        """Track follow-up resolution."""
        await self.emit(
            EventType.FOLLOWUP_RESOLVED,
            properties={
                "context_id": context_id,
                "window_type": window_type,
                "response_type": response_type,
            },
            metrics={"latency_ms": latency_ms},
        )

    async def track_semantic_match(
        self,
        input_text: str,
        match_count: int,
        top_score: float | None,
        latency_ms: float,
    ) -> None:
        """Track semantic matching."""
        event_type = (
            EventType.SEMANTIC_MATCH_FOUND
            if match_count > 0
            else EventType.SEMANTIC_MATCH_NONE
        )

        await self.emit(
            event_type,
            properties={
                "input_length": len(input_text),
                "match_count": match_count,
            },
            metrics={
                "top_score": top_score or 0.0,
                "latency_ms": latency_ms,
            },
        )

    async def track_error(
        self, operation: str, error_type: str, error_message: str
    ) -> None:
        """Track error occurrence."""
        await self.emit(
            EventType.ERROR_OCCURRED,
            properties={
                "operation": operation,
                "error_type": error_type,
                "error_message": error_message,
            },
        )


# Global telemetry instance
_global_telemetry: TelemetryManager | None = None


def get_telemetry() -> TelemetryManager:
    """Get global telemetry manager."""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = TelemetryManager()
        # Add default logging sink
        _global_telemetry.add_sink(LoggingEventSink())
    return _global_telemetry


def init_telemetry(
    sinks: list[EventSink] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> TelemetryManager:
    """Initialize global telemetry with custom config."""
    global _global_telemetry
    _global_telemetry = TelemetryManager()

    if sinks:
        for sink in sinks:
            _global_telemetry.add_sink(sink)

    if session_id:
        _global_telemetry.set_session(session_id, user_id)

    return _global_telemetry


# Context manager for latency tracking
class LatencyTracker:
    """Context manager for tracking operation latency."""

    def __init__(self, operation: str, telemetry: TelemetryManager | None = None):
        self._operation = operation
        self._telemetry = telemetry or get_telemetry()
        self._start_time: float | None = None

    async def __aenter__(self):
        self._start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._start_time is not None:
            latency_seconds = time.perf_counter() - self._start_time
            await self._telemetry.emit(
                EventType.LATENCY_RECORDED,
                properties={"operation": self._operation},
                metrics={"latency_seconds": latency_seconds},
            )

        if exc_type is not None:
            await self._telemetry.track_error(
                operation=self._operation,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
            )
