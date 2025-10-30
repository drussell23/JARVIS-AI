"""
Advanced Context Envelope Models

This module provides dynamic, type-safe context handling without hardcoding.
It implements a flexible context management system with lifecycle tracking,
priority-based resolution, and extensible payload types.

The core components include:
- ContextEnvelope: Generic container for context data with metadata
- Various payload types for different context categories
- Enums for categorization, priority, and state management
- Protocol-based type safety for extensibility

Example:
    >>> from datetime import datetime
    >>> metadata = ContextMetadata(category=ContextCategory.VISION, priority=ContextPriority.HIGH)
    >>> payload = VisionContextPayload("browser", "win123", "space1", "snap456", "Login form")
    >>> envelope = ContextEnvelope(metadata, payload)
    >>> envelope.access()
    >>> print(envelope.relevance_score())
    0.75
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Generic, Protocol, TypeVar, TypedDict, runtime_checkable
from uuid import uuid4


class ContextCategory(Enum):
    """Dynamic context categories - extensible without code changes.
    
    Provides predefined categories for organizing different types of context
    data while allowing fallback behavior for unknown categories.
    
    Attributes:
        VISION: Visual/UI context from screen analysis
        COMMAND: Command execution context
        MEMORY: Persistent memory context
        INTERACTION: User interaction context
        SYSTEM: System-level context (default fallback)
    """
    VISION = auto()
    COMMAND = auto()
    MEMORY = auto()
    INTERACTION = auto()
    SYSTEM = auto()

    @classmethod
    def from_string(cls, value: str) -> ContextCategory:
        """Create ContextCategory from string with case-insensitive lookup.
        
        Args:
            value: String representation of the category
            
        Returns:
            ContextCategory enum value, defaults to SYSTEM if not found
            
        Example:
            >>> ContextCategory.from_string("vision")
            ContextCategory.VISION
            >>> ContextCategory.from_string("unknown")
            ContextCategory.SYSTEM
        """
        try:
            return cls[value.upper()]
        except KeyError:
            return cls.SYSTEM  # fallback


class ContextPriority(Enum):
    """Priority levels for context resolution.
    
    Numeric values are used for scoring and comparison operations.
    Higher values indicate higher priority.
    
    Attributes:
        CRITICAL: Highest priority (100) - immediate attention required
        HIGH: High priority (75) - important but not critical
        NORMAL: Normal priority (50) - standard processing
        LOW: Low priority (25) - can be deferred
        BACKGROUND: Lowest priority (10) - background processing only
    """
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 10


class ContextState(Enum):
    """Lifecycle states for context envelopes.
    
    Tracks the current state of a context envelope throughout its lifecycle.
    
    Attributes:
        PENDING: Created but not yet accessed
        ACTIVE: Currently being used
        CONSUMED: Used once and marked as consumed
        EXPIRED: Past its time-to-live
        INVALIDATED: Manually invalidated
    """
    PENDING = "pending"
    ACTIVE = "active"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


@runtime_checkable
class ContextPayload(Protocol):
    """Protocol for type-safe context payloads.
    
    Defines the interface that all context payload types must implement
    to ensure consistent serialization, deserialization, and validation.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize payload to dictionary format.
        
        Returns:
            Dictionary representation of the payload
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextPayload:
        """Deserialize payload from dictionary format.
        
        Args:
            data: Dictionary containing payload data
            
        Returns:
            New instance of the payload class
        """
        ...

    def validate(self) -> bool:
        """Validate payload integrity and completeness.
        
        Returns:
            True if payload is valid, False otherwise
        """
        ...


T = TypeVar('T', bound=ContextPayload)


@dataclass(slots=True, frozen=True)
class ContextMetadata:
    """Immutable metadata for context tracking.
    
    Contains all metadata associated with a context envelope, including
    identification, categorization, timing, and relationship information.
    
    Attributes:
        id: Unique identifier for the context
        created_at: Timestamp when context was created
        category: Category classification of the context
        priority: Priority level for processing
        source: Source system or component that created the context
        tags: Tuple of string tags for additional classification
        parent_id: Optional ID of parent context for hierarchical relationships
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    category: ContextCategory = ContextCategory.SYSTEM
    priority: ContextPriority = ContextPriority.NORMAL
    source: str = "unknown"
    tags: tuple[str, ...] = field(default_factory=tuple)
    parent_id: str | None = None

    def with_tags(self, *new_tags: str) -> ContextMetadata:
        """Create new metadata instance with additional tags.
        
        Since metadata is immutable, this creates a new instance with
        the existing tags plus the new ones.
        
        Args:
            *new_tags: Variable number of string tags to add
            
        Returns:
            New ContextMetadata instance with combined tags
            
        Example:
            >>> metadata = ContextMetadata(tags=("ui", "form"))
            >>> new_metadata = metadata.with_tags("login", "secure")
            >>> new_metadata.tags
            ('ui', 'form', 'login', 'secure')
        """
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
    """Dynamic context container with lifecycle management.
    
    Generic container that wraps context payloads with metadata, state tracking,
    and lifecycle management. Provides relevance scoring, expiration handling,
    and access tracking.
    
    Type Parameters:
        T: Type of the payload, must implement ContextPayload protocol
        
    Attributes:
        metadata: Immutable metadata for the context
        payload: The actual context data
        state: Current lifecycle state
        ttl_seconds: Time-to-live in seconds (default 120)
        decay_rate: Relevance decay rate per second (0.0-1.0)
        access_count: Number of times context has been accessed
        last_accessed: Timestamp of last access
        constraints: Additional constraints or metadata
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
        """Check if context has exceeded its time-to-live.
        
        Args:
            now: Current timestamp, defaults to datetime.utcnow()
            
        Returns:
            True if context has expired, False otherwise
            
        Example:
            >>> envelope = ContextEnvelope(metadata, payload, ttl_seconds=60)
            >>> # After 61 seconds
            >>> envelope.is_expired()
            True
        """
        now = now or datetime.utcnow()
        return now > self.metadata.created_at + timedelta(seconds=self.ttl_seconds)

    def is_valid(self, now: datetime | None = None) -> bool:
        """Check overall validity of the context envelope.
        
        Validates state, expiration, and payload integrity.
        
        Args:
            now: Current timestamp, defaults to datetime.utcnow()
            
        Returns:
            True if context is valid and usable, False otherwise
        """
        if self.state in (ContextState.EXPIRED, ContextState.INVALIDATED):
            return False
        if self.is_expired(now):
            return False
        if isinstance(self.payload, ContextPayload):
            return self.payload.validate()
        return True

    def relevance_score(self, now: datetime | None = None) -> float:
        """Calculate time-decayed relevance score.
        
        Computes a relevance score between 0.0 and 1.0 based on multiple factors:
        - Base priority level
        - Time-based decay
        - Recent access patterns
        
        Args:
            now: Current timestamp, defaults to datetime.utcnow()
            
        Returns:
            Relevance score between 0.0 (irrelevant) and 1.0 (highly relevant)
            
        Example:
            >>> envelope = ContextEnvelope(metadata, payload)
            >>> envelope.metadata.priority = ContextPriority.HIGH
            >>> envelope.access()
            >>> score = envelope.relevance_score()
            >>> 0.7 <= score <= 1.0
            True
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
        """Mark context as accessed and update tracking information.
        
        Increments access count, updates last accessed timestamp, and
        transitions from PENDING to ACTIVE state if needed.
        """
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        if self.state == ContextState.PENDING:
            self.state = ContextState.ACTIVE

    def consume(self) -> None:
        """Mark context as consumed for one-shot usage patterns.
        
        Sets state to CONSUMED and calls access() to update tracking.
        Used for contexts that should only be used once.
        """
        self.state = ContextState.CONSUMED
        self.access()

    def invalidate(self, reason: str = "") -> None:
        """Manually invalidate the context envelope.
        
        Sets state to INVALIDATED and optionally records the reason.
        
        Args:
            reason: Optional reason for invalidation
        """
        self.state = ContextState.INVALIDATED
        if reason:
            self.constraints["invalidation_reason"] = reason

    def matches_constraint(self, key: str, value: Any) -> bool:
        """Check if a constraint key matches the given value.
        
        Args:
            key: Constraint key to check
            value: Expected value for the constraint
            
        Returns:
            True if constraint matches, False otherwise
        """
        return self.constraints.get(key) == value

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire envelope to dictionary format.
        
        Converts the envelope and all its components to a dictionary
        suitable for JSON serialization or storage.
        
        Returns:
            Dictionary representation of the complete envelope
            
        Example:
            >>> envelope = ContextEnvelope(metadata, payload)
            >>> data = envelope.to_dict()
            >>> "metadata" in data and "payload" in data
            True
        """
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
    """Concrete vision context payload for UI/screen analysis data.
    
    Contains information about visual elements, OCR text, and analysis
    results from screen captures or UI inspections.
    
    Attributes:
        window_type: Type of window (e.g., "browser", "application")
        window_id: Unique identifier for the window
        space_id: Identifier for the workspace or desktop space
        snapshot_id: Unique identifier for the screen snapshot
        summary: Human-readable summary of the visual content
        ocr_text: Extracted text from OCR analysis (optional)
        detected_elements: List of detected UI elements with metadata
        analysis_metadata: Additional analysis data and metrics
    """
    window_type: str
    window_id: str
    space_id: str
    snapshot_id: str
    summary: str
    ocr_text: str | None = None
    detected_elements: list[dict[str, Any]] = field(default_factory=list)
    analysis_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize vision payload to dictionary format.
        
        Returns:
            Dictionary representation of the vision context data
        """
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
        """Create VisionContextPayload from dictionary data.
        
        Args:
            data: Dictionary containing vision context data
            
        Returns:
            New VisionContextPayload instance
            
        Raises:
            KeyError: If required fields are missing from data
        """
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
        """Validate vision payload has required fields.
        
        Returns:
            True if all required fields are present and non-empty
        """
        return bool(self.window_id and self.snapshot_id and self.summary)


@dataclass(slots=True)
class InteractionContextPayload:
    """Context payload for follow-up interactions and clarifications.
    
    Used when the system needs additional information from the user
    or when managing multi-turn conversations.
    
    Attributes:
        question_text: The question or prompt for the user
        expected_response_types: Types of responses expected (e.g., "text", "choice")
        linked_context_id: ID of related context that prompted this interaction
        clarification_needed: Whether this is a clarification request
        partial_info: Any partial information already collected
    """
    question_text: str
    expected_response_types: tuple[str, ...]
    linked_context_id: str | None = None
    clarification_needed: bool = False
    partial_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize interaction payload to dictionary format.
        
        Returns:
            Dictionary representation of the interaction context data
        """
        return {
            "question_text": self.question_text,
            "expected_response_types": list(self.expected_response_types),
            "linked_context_id": self.linked_context_id,
            "clarification_needed": self.clarification_needed,
            "partial_info": self.partial_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionContextPayload:
        """Create InteractionContextPayload from dictionary data.
        
        Args:
            data: Dictionary containing interaction context data
            
        Returns:
            New InteractionContextPayload instance
            
        Raises:
            KeyError: If required fields are missing from data
        """
        return cls(
            question_text=data["question_text"],
            expected_response_types=tuple(data["expected_response_types"]),
            linked_context_id=data.get("linked_context_id"),
            clarification_needed=data.get("clarification_needed", False),
            partial_info=data.get("partial_info", {}),
        )

    def validate(self) -> bool:
        """Validate interaction payload has required fields.
        
        Returns:
            True if question text and expected response types are present
        """
        return bool(self.question_text and self.expected_response_types)


# Type aliases for common envelope types
VisionEnvelope = ContextEnvelope[VisionContextPayload]
"""Type alias for vision context envelopes."""

InteractionEnvelope = ContextEnvelope[InteractionContextPayload]
"""Type alias for interaction context envelopes."""