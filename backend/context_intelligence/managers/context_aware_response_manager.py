"""
Context-Aware Response Manager
===============================

Enriches responses with conversation context to avoid re-asking for information.

Good Response (Context-Aware):
✅ Query: "What's the error?"
   Context: User just asked about Space 3
   Response: "The error in Space 3 is a TypeError on line 421."

Bad Response (Context-Unaware):
❌ Query: "What's the error?"
   Response: "Which space do you mean?"
   (User has to re-specify)

Strategy:
- Track recent conversation context (spaces, files, apps, entities)
- Inject missing context into responses
- Use ImplicitReferenceResolver to understand recent entities
- Maintain conversation history and entity references
- Enrich vague queries with contextual information
"""

import asyncio
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context to track"""
    SPACE = "space"
    WINDOW = "window"
    APPLICATION = "application"
    FILE = "file"
    ERROR = "error"
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    URL = "url"
    QUERY = "query"


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""
    timestamp: datetime
    query: str
    response: str
    entities: Dict[str, Any]  # Entities mentioned in this turn
    context_used: Dict[str, Any]  # Context that was used
    turn_id: int


@dataclass
class EntityReference:
    """Reference to an entity in conversation"""
    entity_type: ContextType
    entity_value: str
    last_mentioned: datetime
    mention_count: int
    context: Optional[Dict[str, Any]] = None
    confidence: float = 1.0


@dataclass
class ConversationContext:
    """Current conversation context"""
    recent_spaces: List[int]
    recent_files: List[str]
    recent_apps: List[str]
    recent_errors: List[str]
    last_entity_by_type: Dict[ContextType, EntityReference]
    turn_count: int
    session_start: datetime


@dataclass
class ContextEnrichment:
    """Result of context enrichment"""
    original_response: str
    enriched_response: str
    context_added: Dict[str, Any]
    confidence: float
    enrichment_time: float


class ConversationTracker:
    """
    Tracks conversation history and entity references.

    Maintains:
    - Recent conversation turns (last N)
    - Entity references with recency
    - Context about what was discussed
    """

    def __init__(self, max_history: int = 10, context_ttl: float = 300.0):
        """
        Initialize conversation tracker.

        Args:
            max_history: Maximum number of turns to keep
            context_ttl: Time-to-live for context in seconds (5 minutes default)
        """
        self.max_history = max_history
        self.context_ttl = context_ttl

        # Conversation history
        self.turns: deque[ConversationTurn] = deque(maxlen=max_history)
        self.turn_counter = 0

        # Entity tracking
        self.entity_references: Dict[str, EntityReference] = {}

        # Current context
        self.session_start = datetime.now()

    def add_turn(
        self,
        query: str,
        response: str,
        entities: Optional[Dict[str, Any]] = None,
        context_used: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """
        Add a conversation turn to history.

        Args:
            query: User query
            response: System response
            entities: Entities mentioned
            context_used: Context used in this turn

        Returns:
            ConversationTurn object
        """
        self.turn_counter += 1

        turn = ConversationTurn(
            timestamp=datetime.now(),
            query=query,
            response=response,
            entities=entities or {},
            context_used=context_used or {},
            turn_id=self.turn_counter
        )

        self.turns.append(turn)

        # Update entity references
        if entities:
            self._update_entity_references(entities)

        return turn

    def _update_entity_references(self, entities: Dict[str, Any]):
        """Update entity reference tracking"""
        now = datetime.now()

        for entity_type_str, entity_values in entities.items():
            try:
                entity_type = ContextType(entity_type_str)
            except ValueError:
                # Unknown entity type, skip
                continue

            # Handle single value or list
            values = entity_values if isinstance(entity_values, list) else [entity_values]

            for value in values:
                key = f"{entity_type.value}:{value}"

                if key in self.entity_references:
                    # Update existing reference
                    ref = self.entity_references[key]
                    ref.last_mentioned = now
                    ref.mention_count += 1
                else:
                    # Create new reference
                    self.entity_references[key] = EntityReference(
                        entity_type=entity_type,
                        entity_value=str(value),
                        last_mentioned=now,
                        mention_count=1,
                        confidence=1.0
                    )

    def get_recent_context(self, max_age: Optional[float] = None) -> ConversationContext:
        """
        Get recent conversation context.

        Args:
            max_age: Maximum age of context in seconds (uses TTL if not specified)

        Returns:
            ConversationContext with recent entities
        """
        max_age = max_age or self.context_ttl
        cutoff_time = datetime.now() - timedelta(seconds=max_age)

        # Filter recent entity references
        recent_entities = {
            key: ref for key, ref in self.entity_references.items()
            if ref.last_mentioned >= cutoff_time
        }

        # Group by type
        recent_spaces = []
        recent_files = []
        recent_apps = []
        recent_errors = []
        last_by_type = {}

        for ref in recent_entities.values():
            if ref.entity_type == ContextType.SPACE:
                try:
                    recent_spaces.append(int(ref.entity_value))
                except ValueError:
                    pass
            elif ref.entity_type == ContextType.FILE:
                recent_files.append(ref.entity_value)
            elif ref.entity_type == ContextType.APPLICATION:
                recent_apps.append(ref.entity_value)
            elif ref.entity_type == ContextType.ERROR:
                recent_errors.append(ref.entity_value)

            # Track most recent by type
            if ref.entity_type not in last_by_type:
                last_by_type[ref.entity_type] = ref
            elif ref.last_mentioned > last_by_type[ref.entity_type].last_mentioned:
                last_by_type[ref.entity_type] = ref

        return ConversationContext(
            recent_spaces=sorted(set(recent_spaces), key=lambda x: -x),  # Most recent first
            recent_files=recent_files,
            recent_apps=recent_apps,
            recent_errors=recent_errors,
            last_entity_by_type=last_by_type,
            turn_count=len(self.turns),
            session_start=self.session_start
        )

    def get_last_mentioned_entity(self, entity_type: ContextType) -> Optional[EntityReference]:
        """Get the most recently mentioned entity of a type"""
        context = self.get_recent_context()
        return context.last_entity_by_type.get(entity_type)

    def get_recent_turns(self, count: int = 3) -> List[ConversationTurn]:
        """Get the N most recent conversation turns"""
        return list(self.turns)[-count:]

    def clear_old_context(self):
        """Clear context older than TTL"""
        cutoff_time = datetime.now() - timedelta(seconds=self.context_ttl)

        # Remove old entity references
        old_keys = [
            key for key, ref in self.entity_references.items()
            if ref.last_mentioned < cutoff_time
        ]

        for key in old_keys:
            del self.entity_references[key]


class ContextInjector:
    """
    Injects missing context into responses.

    Transforms:
    - "The error is a TypeError" → "The error in Space 3 is a TypeError"
    - "The function is broken" → "The function 'process_command' in test.py is broken"
    """

    def __init__(self, conversation_tracker: ConversationTracker):
        """Initialize context injector"""
        self.tracker = conversation_tracker

    def inject_context(
        self,
        response: str,
        query: Optional[str] = None,
        context: Optional[ConversationContext] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Inject missing context into response.

        Args:
            response: Original response
            query: Original query (for context clues)
            context: Current conversation context

        Returns:
            Tuple of (enriched_response, context_added)
        """
        context = context or self.tracker.get_recent_context()
        enriched = response
        context_added = {}

        # Check if response is missing space context
        if self._needs_space_context(response, query):
            enriched, space_added = self._inject_space_context(enriched, context)
            if space_added:
                context_added.update(space_added)

        # Check if response is missing file context
        if self._needs_file_context(response, query):
            enriched, file_added = self._inject_file_context(enriched, context)
            if file_added:
                context_added.update(file_added)

        # Check if response is missing app context
        if self._needs_app_context(response, query):
            enriched, app_added = self._inject_app_context(enriched, context)
            if app_added:
                context_added.update(app_added)

        return enriched, context_added

    def _needs_space_context(self, response: str, query: Optional[str]) -> bool:
        """Check if response needs space context"""
        # If response doesn't mention a space but query asks about location
        response_lower = response.lower()
        query_lower = query.lower() if query else ""

        # Response doesn't have space info
        has_space = "space" in response_lower or any(f"space {i}" in response_lower for i in range(1, 11))

        # Query is asking about location/state
        asking_about_location = any(word in query_lower for word in [
            "what", "where", "which", "show", "error", "issue", "problem"
        ])

        return not has_space and asking_about_location

    def _needs_file_context(self, response: str, query: Optional[str]) -> bool:
        """Check if response needs file context"""
        response_lower = response.lower()
        query_lower = query.lower() if query else ""

        # Response doesn't mention a file
        has_file = any(ext in response_lower for ext in [
            ".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs"
        ])

        # Query asks about code/errors
        asking_about_code = any(word in query_lower for word in [
            "error", "function", "class", "code", "file", "line"
        ])

        return not has_file and asking_about_code

    def _needs_app_context(self, response: str, query: Optional[str]) -> bool:
        """Check if response needs app context"""
        response_lower = response.lower()
        query_lower = query.lower() if query else ""

        # Query asks about apps
        asking_about_app = any(word in query_lower for word in [
            "app", "application", "running", "open"
        ])

        return asking_about_app

    def _inject_space_context(
        self, response: str, context: ConversationContext
    ) -> Tuple[str, Dict[str, Any]]:
        """Inject space context into response"""
        if not context.recent_spaces:
            return response, {}

        # Get most recent space
        space_id = context.recent_spaces[0]

        # Find insertion point (after "The" or at beginning)
        if response.startswith("The "):
            enriched = f"The {response[4:]}"
            # Insert space context after article
            enriched = f"The error in Space {space_id} {response[10:]}" if "error" in response.lower() else f"The content in Space {space_id} {response[4:]}"
        else:
            enriched = f"In Space {space_id}: {response}"

        return enriched, {"space_id": space_id}

    def _inject_file_context(
        self, response: str, context: ConversationContext
    ) -> Tuple[str, Dict[str, Any]]:
        """Inject file context into response"""
        if not context.recent_files:
            return response, {}

        file_name = context.recent_files[-1]  # Most recent

        # Look for places to inject file context
        if "error" in response.lower() and "in" not in response.lower():
            # "The error is a TypeError" → "The error in test.py is a TypeError"
            enriched = response.replace("error is", f"error in {file_name} is")
            return enriched, {"file": file_name}

        return response, {}

    def _inject_app_context(
        self, response: str, context: ConversationContext
    ) -> Tuple[str, Dict[str, Any]]:
        """Inject app context into response"""
        if not context.recent_apps:
            return response, {}

        app_name = context.recent_apps[-1]

        enriched = f"{response} (in {app_name})"
        return enriched, {"app": app_name}


class ImplicitContextResolver:
    """
    Resolves implicit context using ImplicitReferenceResolver.

    Bridges conversation context with entity resolution.
    """

    def __init__(
        self,
        implicit_resolver: Optional[Any] = None,
        conversation_tracker: Optional[ConversationTracker] = None
    ):
        """
        Initialize implicit context resolver.

        Args:
            implicit_resolver: ImplicitReferenceResolver instance
            conversation_tracker: ConversationTracker instance
        """
        self.implicit_resolver = implicit_resolver
        self.tracker = conversation_tracker

    async def resolve_context_for_response(
        self,
        query: str,
        response: str
    ) -> Dict[str, Any]:
        """
        Resolve implicit context for enriching response.

        Args:
            query: Original user query
            response: System response

        Returns:
            Dictionary with resolved context
        """
        resolved_context = {}

        # Use implicit resolver if available
        if self.implicit_resolver:
            try:
                # Get entity resolution from query
                resolved_entities = await asyncio.to_thread(
                    self.implicit_resolver.get_context
                )

                if resolved_entities:
                    # Extract relevant context
                    if "current_space" in resolved_entities:
                        resolved_context["space_id"] = resolved_entities["current_space"]

                    if "recent_entities" in resolved_entities:
                        recent = resolved_entities["recent_entities"]
                        if "files" in recent:
                            resolved_context["files"] = recent["files"]
                        if "apps" in recent:
                            resolved_context["apps"] = recent["apps"]

            except Exception as e:
                logger.debug(f"Implicit context resolution failed: {e}")

        # Augment with conversation tracker context
        if self.tracker:
            conv_context = self.tracker.get_recent_context()

            if conv_context.recent_spaces and "space_id" not in resolved_context:
                resolved_context["space_id"] = conv_context.recent_spaces[0]

            if conv_context.recent_files and "files" not in resolved_context:
                resolved_context["files"] = conv_context.recent_files

            if conv_context.recent_apps and "apps" not in resolved_context:
                resolved_context["apps"] = conv_context.recent_apps

        return resolved_context


class ContextAwareResponseManager:
    """
    Main manager for context-aware response generation.

    Enriches responses with conversation context to avoid re-asking for information.
    """

    def __init__(
        self,
        implicit_resolver: Optional[Any] = None,
        max_history: int = 10,
        context_ttl: float = 300.0
    ):
        """
        Initialize Context-Aware Response Manager.

        Args:
            implicit_resolver: Optional ImplicitReferenceResolver
            max_history: Maximum conversation turns to track
            context_ttl: Context time-to-live in seconds
        """
        self.conversation_tracker = ConversationTracker(max_history, context_ttl)
        self.context_injector = ContextInjector(self.conversation_tracker)
        self.implicit_resolver = ImplicitContextResolver(
            implicit_resolver, self.conversation_tracker
        )

    async def enrich_response(
        self,
        query: str,
        response: str,
        extracted_entities: Optional[Dict[str, Any]] = None
    ) -> ContextEnrichment:
        """
        Enrich response with conversation context.

        Args:
            query: User query
            response: System response
            extracted_entities: Entities extracted from query/response

        Returns:
            ContextEnrichment with enriched response
        """
        start_time = time.time()

        # Step 1: Get current conversation context
        context = self.conversation_tracker.get_recent_context()

        # Step 2: Resolve implicit context from resolver
        resolved_context = await self.implicit_resolver.resolve_context_for_response(
            query, response
        )

        # Step 3: Inject missing context into response
        enriched_response, context_added = self.context_injector.inject_context(
            response, query, context
        )

        # Merge resolved context with added context
        all_context = {**resolved_context, **context_added}

        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(context_added, context)

        # Step 5: Record this turn
        self.conversation_tracker.add_turn(
            query=query,
            response=enriched_response,
            entities=extracted_entities or {},
            context_used=all_context
        )

        execution_time = time.time() - start_time

        return ContextEnrichment(
            original_response=response,
            enriched_response=enriched_response,
            context_added=all_context,
            confidence=confidence,
            enrichment_time=execution_time
        )

    def _calculate_confidence(
        self,
        context_added: Dict[str, Any],
        conversation_context: ConversationContext
    ) -> float:
        """Calculate confidence in context enrichment"""
        if not context_added:
            return 1.0  # No enrichment needed

        confidence = 1.0

        # Reduce confidence if context is old
        if conversation_context.last_entity_by_type:
            most_recent = max(
                ref.last_mentioned
                for ref in conversation_context.last_entity_by_type.values()
            )
            age = (datetime.now() - most_recent).total_seconds()

            # Decay confidence based on age (5 minutes = full confidence, 30 minutes = 50%)
            confidence *= max(0.5, 1.0 - (age / 1800.0))

        # Increase confidence if entity was mentioned multiple times
        if "space_id" in context_added:
            space_ref = self.conversation_tracker.get_last_mentioned_entity(ContextType.SPACE)
            if space_ref and space_ref.mention_count > 1:
                confidence = min(1.0, confidence + 0.1 * (space_ref.mention_count - 1))

        return confidence

    def add_entity_mention(
        self,
        entity_type: str,
        entity_value: Any,
        context: Optional[Dict[str, Any]] = None
    ):
        """Manually add an entity mention to tracking"""
        entities = {entity_type: entity_value}
        self.conversation_tracker._update_entity_references(entities)

    def get_recent_context(self) -> ConversationContext:
        """Get current conversation context"""
        return self.conversation_tracker.get_recent_context()

    def clear_context(self):
        """Clear conversation context"""
        self.conversation_tracker.entity_references.clear()
        self.conversation_tracker.turns.clear()


# Global instance
_context_aware_manager: Optional[ContextAwareResponseManager] = None


def get_context_aware_response_manager() -> Optional[ContextAwareResponseManager]:
    """Get the global ContextAwareResponseManager instance"""
    return _context_aware_manager


def initialize_context_aware_response_manager(
    implicit_resolver: Optional[Any] = None,
    max_history: int = 10,
    context_ttl: float = 300.0
) -> ContextAwareResponseManager:
    """
    Initialize the global ContextAwareResponseManager instance.

    Args:
        implicit_resolver: Optional ImplicitReferenceResolver
        max_history: Maximum conversation turns to track
        context_ttl: Context time-to-live in seconds (5 minutes default)

    Returns:
        ContextAwareResponseManager instance
    """
    global _context_aware_manager

    _context_aware_manager = ContextAwareResponseManager(
        implicit_resolver=implicit_resolver,
        max_history=max_history,
        context_ttl=context_ttl
    )

    return _context_aware_manager
