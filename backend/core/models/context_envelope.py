"""
Advanced Context Envelope Models
Provides dynamic, type-safe context handling without hardcoding.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Generic, Protocol, TypeVar, TypedDict, runtime_checkable
from uuid import uuid4


class ContextCategory(Enum):
    """Dynamic context categories - extensible without code changes."""
    VISION = auto()
    COMMAND = auto()
    MEMORY = auto()
    INTERACTION = auto()
    SYSTEM = auto()

    @classmethod
    def from_string(cls, value: str) -> ContextCategory:
        """Case-insensitive lookup."""
        try:
            return cls[value.upper()]
        except KeyError:
            return cls.SYSTEM  # fallback


class ContextPriority(Enum):
    """Priority levels for context resolution."""
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 10


class ContextState(Enum):
    """Lifecycle states."""
    PENDING = "pending"
    ACTIVE = "active"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


@runtime_checkable
class ContextPayload(Protocol):
    """Protocol for type-safe context payloads."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextPayload:
        """Deserialize from dict."""
        ...

    def validate(self) -> bool:
        """Validate payload integrity."""
        ...


T = TypeVar('T', bound=ContextPayload)


@dataclass(slots=True, frozen=True)
class ContextMetadata:
    """Immutable metadata for context tracking."""
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    category: ContextCategory = ContextCategory.SYSTEM
    priority: ContextPriority = ContextPriority.NORMAL
    source: str = "unknown"
    tags: tuple[str, ...] = field(default_factory=tuple)
    parent_id: str | None = None

    def with_tags(self, *new_tags: str) -> ContextMetadata:
        """Immutable tag addition."""
        return ContextMetadata(
            id=self.id,
            created_at=self.created_at,
            category=self.category,
            priority=self.priority,
            source=self.source,
            tags=self.tags + new_tags,
            parent_id=self.parent_id
        )


@dataclass(slots=True)
class ContextEnvelope(Generic[T]):
    """
    Dynamic context container with lifecycle management.
    Generic over payload type for type safety.
    """
    metadata: ContextMetadata
    payload: T
    state: ContextState = ContextState.PENDING
    ttl_seconds: int = 120
    decay_rate: float = 0.0  # 0-1, relevance decay per second
    access_count: int = 0
    last_accessed: datetime | None = None
    constraints: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check absolute expiry."""
        now = now or datetime.utcnow()
        return now > self.metadata.created_at + timedelta(seconds=self.ttl_seconds)

    def is_valid(self, now: datetime | None = None) -> bool:
        """Check validity (state + expiry + payload)."""
        if self.state in (ContextState.EXPIRED, ContextState.INVALIDATED):
            return False
        if self.is_expired(now):
            return False
        if isinstance(self.payload, ContextPayload):
            return self.payload.validate()
        return True

    def relevance_score(self, now: datetime | None = None) -> float:
        """
        Calculate time-decayed relevance [0.0-1.0].
        Factors: priority, age, decay rate, access pattern.
        """
        now = now or datetime.utcnow()

        if not self.is_valid(now):
            return 0.0

        # Base score from priority
        base = self.metadata.priority.value / 100.0

        # Time decay
        age_seconds = (now - self.metadata.created_at).total_seconds()
        time_factor = max(0.0, 1.0 - (age_seconds * self.decay_rate))

        # Access recency boost (if accessed recently, higher score)
        access_boost = 1.0
        if self.last_accessed:
            recency_seconds = (now - self.last_accessed).total_seconds()
            access_boost = 1.0 + (0.2 * max(0, 1.0 - recency_seconds / 30.0))

        return min(1.0, base * time_factor * access_boost)

    def access(self) -> None:
        """Mark context as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        if self.state == ContextState.PENDING:
            self.state = ContextState.ACTIVE

    def consume(self) -> None:
        """Mark context as consumed (one-shot)."""
        self.state = ContextState.CONSUMED
        self.access()

    def invalidate(self, reason: str = "") -> None:
        """Manually invalidate."""
        self.state = ContextState.INVALIDATED
        if reason:
            self.constraints["invalidation_reason"] = reason

    def matches_constraint(self, key: str, value: Any) -> bool:
        """Check if constraint matches."""
        return self.constraints.get(key) == value

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire envelope."""
        payload_dict = (
            self.payload.to_dict()
            if isinstance(self.payload, ContextPayload)
            else self.payload
        )

        return {
            "metadata": {
                "id": self.metadata.id,
                "created_at": self.metadata.created_at.isoformat(),
                "category": self.metadata.category.name,
                "priority": self.metadata.priority.name,
                "source": self.metadata.source,
                "tags": list(self.metadata.tags),
                "parent_id": self.metadata.parent_id,
            },
            "payload": payload_dict,
            "state": self.state.value,
            "ttl_seconds": self.ttl_seconds,
            "decay_rate": self.decay_rate,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "constraints": self.constraints,
        }


@dataclass(slots=True)
class VisionContextPayload:
    """Concrete vision context payload."""
    window_type: str
    window_id: str
    space_id: str
    snapshot_id: str
    summary: str
    ocr_text: str | None = None
    detected_elements: list[dict[str, Any]] = field(default_factory=list)
    analysis_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_type": self.window_type,
            "window_id": self.window_id,
            "space_id": self.space_id,
            "snapshot_id": self.snapshot_id,
            "summary": self.summary,
            "ocr_text": self.ocr_text,
            "detected_elements": self.detected_elements,
            "analysis_metadata": self.analysis_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionContextPayload:
        return cls(
            window_type=data["window_type"],
            window_id=data["window_id"],
            space_id=data["space_id"],
            snapshot_id=data["snapshot_id"],
            summary=data["summary"],
            ocr_text=data.get("ocr_text"),
            detected_elements=data.get("detected_elements", []),
            analysis_metadata=data.get("analysis_metadata", {}),
        )

    def validate(self) -> bool:
        return bool(self.window_id and self.snapshot_id and self.summary)


@dataclass(slots=True)
class InteractionContextPayload:
    """Follow-up / interaction context."""
    question_text: str
    expected_response_types: tuple[str, ...]
    linked_context_id: str | None = None
    clarification_needed: bool = False
    partial_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_text": self.question_text,
            "expected_response_types": list(self.expected_response_types),
            "linked_context_id": self.linked_context_id,
            "clarification_needed": self.clarification_needed,
            "partial_info": self.partial_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionContextPayload:
        return cls(
            question_text=data["question_text"],
            expected_response_types=tuple(data["expected_response_types"]),
            linked_context_id=data.get("linked_context_id"),
            clarification_needed=data.get("clarification_needed", False),
            partial_info=data.get("partial_info", {}),
        )

    def validate(self) -> bool:
        return bool(self.question_text and self.expected_response_types)


# Type aliases for common envelope types
VisionEnvelope = ContextEnvelope[VisionContextPayload]
InteractionEnvelope = ContextEnvelope[InteractionContextPayload]
