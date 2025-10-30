"""
Implicit Reference Resolver - Advanced Natural Language Understanding
======================================================================

This module provides sophisticated natural language understanding capabilities for JARVIS,
enabling it to resolve implicit references in user queries like "it", "that", "this",
and contextual questions.

The system combines conversational context, visual attention tracking, and workspace
awareness to understand what users are referring to when they use pronouns or
implicit references.

Key Features:
    - Conversational Context Tracking: Remembers the last 10 exchanges
    - Visual Attention Mechanism: Knows what you were looking at and when
    - Temporal Relevance: Recent things are more likely referents
    - Pronoun Resolution: it, that, this, these, those, them
    - Implicit Query Understanding: "what's wrong?" → find the error
    - Multi-Modal Context: Combines conversation + visual + workspace context

Architecture:
    User Query → Query Analyzer → Reference Resolver → Context Graph
         ↓              ↓                  ↓                ↓
    Parse Intent   Extract Refs    Find Referents   Retrieve Context
         ↓              ↓                  ↓                ↓
    Intent Type    Pronouns/Refs   Candidate List    Full Context
         ↓              ↓                  ↓                ↓
         └──────────────┴──────────────────┴────────────→ Response

Example:
    >>> resolver = ImplicitReferenceResolver(context_graph)
    >>> result = await resolver.resolve_query("what does it say?")
    >>> print(result['response'])
    "The error in Terminal (Space 2) is: ModuleNotFoundError: No module named 'requests'"
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import deque
import re

logger = logging.getLogger(__name__)


# ============================================================================
# QUERY INTENT CLASSIFICATION
# ============================================================================

class QueryIntent(Enum):
    """Types of intents a user query can have.
    
    This enum categorizes different types of user intentions when asking questions,
    enabling the system to provide appropriate responses based on what the user
    wants to accomplish.
    """
    # Information seeking
    EXPLAIN = "explain"              # "explain that", "what is this?"
    DESCRIBE = "describe"            # "what does it say?", "what's that?"
    LOCATE = "locate"                # "where is X?", "find the error"
    STATUS = "status"                # "what's happening?", "what's going on?"

    # Problem solving
    DIAGNOSE = "diagnose"            # "what's wrong?", "why did it fail?"
    FIX = "fix"                      # "how do I fix it?", "how to solve this?"
    PREVENT = "prevent"              # "how to avoid this?", "prevent that?"

    # Navigation/History
    RECALL = "recall"                # "what was that?", "show me the error again"
    COMPARE = "compare"              # "what changed?", "what's different?"
    SUMMARIZE = "summarize"          # "summarize this", "what happened?"

    # Meta/Control
    CLARIFY = "clarify"              # "which one?", "be more specific"
    UNKNOWN = "unknown"              # Can't determine intent


class ReferenceType(Enum):
    """Types of references found in user queries.
    
    Categorizes different ways users can refer to objects or concepts
    without explicitly naming them.
    """
    PRONOUN = "pronoun"              # it, that, this, these, those
    DEMONSTRATIVE = "demonstrative"  # this error, that terminal
    POSSESSIVE = "possessive"        # my code, your suggestion
    IMPLICIT = "implicit"            # "the error" (which error?)
    EXPLICIT = "explicit"            # "the error in terminal" (specific)


@dataclass
class ParsedReference:
    """A reference found in the user's query.
    
    Represents a single implicit or explicit reference that needs to be resolved
    to understand what the user is talking about.
    
    Attributes:
        reference_type: The type of reference (pronoun, demonstrative, etc.)
        text: Original text from the query ("it", "that error")
        span: Character positions in the original query
        modifier: Optional adjective/descriptor ("red", "last")
        entity_type: What type of thing is being referenced (error, terminal, file)
    """
    reference_type: ReferenceType
    text: str                        # Original text ("it", "that error")
    span: Tuple[int, int]            # Character positions
    modifier: Optional[str] = None   # Adjective/descriptor ("red", "last")
    entity_type: Optional[str] = None  # What type of thing (error, terminal, file)


@dataclass
class QueryParsed:
    """Parsed user query with intent and references.
    
    Contains the complete analysis of a user query, including what they want
    to do (intent) and what they're referring to (references).
    
    Attributes:
        intent: The classified intent of the query
        confidence: How confident we are in the analysis (0.0-1.0)
        references: List of references found in the query
        keywords: Important words extracted from the query
        temporal_marker: Time-related phrases ("just now", "earlier")
        original_query: The original user query text
    """
    intent: QueryIntent
    confidence: float
    references: List[ParsedReference]
    keywords: List[str]              # Important words
    temporal_marker: Optional[str] = None  # "just now", "earlier", "2 minutes ago"
    original_query: str = ""


# ============================================================================
# CONVERSATIONAL CONTEXT
# ============================================================================

@dataclass
class ConversationTurn:
    """A single turn in the conversation (user query + JARVIS response).
    
    Represents one complete exchange between the user and JARVIS, including
    the context that was used to generate the response.
    
    Attributes:
        turn_id: Unique identifier for this turn
        timestamp: When this turn occurred
        user_query: What the user asked
        jarvis_response: How JARVIS responded
        context_used: What context was used to answer
        entities_mentioned: Entities that were discussed in this turn
    """
    turn_id: str
    timestamp: datetime
    user_query: str
    jarvis_response: str
    context_used: Dict[str, Any]     # What context was used to answer
    entities_mentioned: List[str]     # Entities that were discussed

    def is_recent(self, within_seconds: int = 300) -> bool:
        """Check if this turn was recent.
        
        Args:
            within_seconds: How many seconds ago counts as "recent"
            
        Returns:
            True if this turn occurred within the specified time window
        """
        return (datetime.now() - self.timestamp).total_seconds() <= within_seconds


class ConversationalContext:
    """Tracks the conversation history to resolve references.
    
    This class maintains a rolling history of conversations to help resolve
    references like:
    - "it" → refers to subject of last exchange
    - "that" → refers to something mentioned in last 2-3 turns
    - "explain more" → continues previous topic
    
    Attributes:
        turns: Deque of recent conversation turns
        current_topic: What we're currently discussing
        turn_counter: Counter for generating unique turn IDs
    """

    def __init__(self, max_turns: int = 10):
        """Initialize conversational context.
        
        Args:
            max_turns: Maximum number of turns to keep in memory
        """
        self.turns: deque[ConversationTurn] = deque(maxlen=max_turns)
        self.current_topic: Optional[str] = None  # What we're currently discussing
        self.turn_counter = 0

    def add_turn(self, user_query: str, jarvis_response: str, context_used: Dict[str, Any]) -> None:
        """Add a new conversation turn.
        
        Args:
            user_query: What the user asked
            jarvis_response: How JARVIS responded
            context_used: Context that was used to generate the response
        """
        self.turn_counter += 1

        # Extract entities mentioned
        entities = self._extract_entities(user_query, jarvis_response, context_used)

        turn = ConversationTurn(
            turn_id=f"turn_{self.turn_counter}",
            timestamp=datetime.now(),
            user_query=user_query,
            jarvis_response=jarvis_response,
            context_used=context_used,
            entities_mentioned=entities
        )

        self.turns.append(turn)

        # Update current topic
        if entities:
            self.current_topic = entities[0]  # Most recent entity becomes topic

        logger.debug(f"[CONV-CONTEXT] Added turn {turn.turn_id}, entities: {entities}")

    def get_recent_turns(self, count: int = 3) -> List[ConversationTurn]:
        """Get the last N turns.
        
        Args:
            count: Number of recent turns to retrieve
            
        Returns:
            List of the most recent conversation turns
        """
        return list(self.turns)[-count:]

    def get_last_mentioned_entity(self, entity_type: Optional[str] = None) -> Optional[Tuple[str, datetime]]:
        """Get the most recently mentioned entity.
        
        Args:
            entity_type: Optional filter by entity type (e.g., "error", "file")
            
        Returns:
            Tuple of (entity_text, timestamp) if found, None otherwise
        """
        for turn in reversed(self.turns):
            for entity in turn.entities_mentioned:
                if entity_type is None or self._get_entity_type(entity) == entity_type:
                    return (entity, turn.timestamp)
        return None

    def find_entities_in_context(self, keywords: List[str]) -> List[Tuple[str, datetime, Dict[str, Any]]]:
        """Find entities in recent conversation matching keywords.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of tuples containing (entity, timestamp, context_used)
        """
        results = []
        for turn in reversed(self.turns):
            for entity in turn.entities_mentioned:
                # Check if any keyword appears in entity or its context
                if any(kw.lower() in entity.lower() for kw in keywords):
                    results.append((entity, turn.timestamp, turn.context_used))

        return results

    def _extract_entities(self, user_query: str, jarvis_response: str, context: Dict[str, Any]) -> List[str]:
        """Extract entities (errors, files, commands, etc.) from the conversation.
        
        Args:
            user_query: The user's query
            jarvis_response: JARVIS's response
            context: Context used for the response
            
        Returns:
            List of extracted entity strings
        """
        entities = []

        # From context
        if context.get("type") == "error":
            entities.append(f"error:{context.get('details', {}).get('error', 'unknown')[:50]}")
        elif context.get("type") == "terminal":
            if context.get("last_command"):
                entities.append(f"command:{context['last_command']}")

        # Entity keywords in query/response
        entity_patterns = [
            r'\b(?:error|exception|failure)\b.*',
            r'\b(?:file|folder|directory)\s+[\w/\.]+',
            r'\b(?:command|terminal|shell)\b',
            r'\b(?:function|class|method)\s+\w+',
        ]

        for pattern in entity_patterns:
            for match in re.finditer(pattern, user_query + " " + jarvis_response, re.IGNORECASE):
                entities.append(match.group(0))

        return list(set(entities))  # Deduplicate

    def _get_entity_type(self, entity: str) -> str:
        """Determine the type of an entity.
        
        Args:
            entity: Entity string to classify
            
        Returns:
            Entity type string ("error", "command", "file", "unknown")
        """
        if entity.startswith("error:"):
            return "error"
        elif entity.startswith("command:"):
            return "command"
        elif entity.startswith("file:"):
            return "file"
        else:
            return "unknown"


# ============================================================================
# VISUAL ATTENTION TRACKING
# ============================================================================

@dataclass
class VisualAttentionEvent:
    """Records what the user was looking at and when.
    
    Captures information about what was visible on the user's screen at a
    specific point in time, helping to resolve references to visual content.
    
    Attributes:
        timestamp: When this was observed
        space_id: Which desktop space this occurred in
        app_name: Name of the application
        window_title: Title of the window (if available)
        content_summary: Brief summary of what was visible
        content_type: Type of content ("error", "code", "documentation", etc.)
        significance: How important this content is
        ocr_text_hash: Hash of OCR text for deduplication
    """
    timestamp: datetime
    space_id: int
    app_name: str
    window_title: Optional[str]
    content_summary: str             # Brief summary of what was visible
    content_type: str                # "error", "code", "documentation", "terminal_output"
    significance: str                # "critical", "high", "normal", "low"
    ocr_text_hash: Optional[str] = None  # Hash of OCR text for deduplication

    def is_recent(self, within_seconds: int = 300) -> bool:
        """Check if this attention event was recent.
        
        Args:
            within_seconds: How many seconds ago counts as "recent"
            
        Returns:
            True if this event occurred within the specified time window
        """
        return (datetime.now() - self.timestamp).total_seconds() <= within_seconds


class VisualAttentionTracker:
    """Tracks what the user has been looking at.

    This class maintains a history of what was visible on the user's screen,
    enabling JARVIS to answer questions like "What did I just see?" or
    "What was that on screen?"

    When you switch spaces or scroll, JARVIS remembers what was visible.
    
    Attributes:
        attention_events: Deque of recent visual attention events
        last_critical_event: Most recent critical event for quick access
    """

    def __init__(self, max_events: int = 50):
        """Initialize visual attention tracker.
        
        Args:
            max_events: Maximum number of events to keep in memory
        """
        self.attention_events: deque[VisualAttentionEvent] = deque(maxlen=max_events)
        self.last_critical_event: Optional[VisualAttentionEvent] = None

    def record_attention(self,
                        space_id: int,
                        app_name: str,
                        content_summary: str,
                        content_type: str = "unknown",
                        significance: str = "normal",
                        window_title: Optional[str] = None,
                        ocr_text_hash: Optional[str] = None) -> None:
        """Record that the user was looking at something.
        
        Args:
            space_id: Desktop space ID where this occurred
            app_name: Name of the application
            content_summary: Brief description of what was visible
            content_type: Type of content being viewed
            significance: How important this content is
            window_title: Title of the window (optional)
            ocr_text_hash: Hash of OCR text for deduplication (optional)
        """
        event = VisualAttentionEvent(
            timestamp=datetime.now(),
            space_id=space_id,
            app_name=app_name,
            window_title=window_title,
            content_summary=content_summary,
            content_type=content_type,
            significance=significance,
            ocr_text_hash=ocr_text_hash
        )

        self.attention_events.append(event)

        # Track last critical event separately
        if significance == "critical":
            self.last_critical_event = event

        logger.debug(f"[ATTENTION] Recorded: {content_type} in {app_name} (Space {space_id}), significance={significance}")

    def get_most_recent_by_type(self, content_type: str, within_seconds: int = 300) -> Optional[VisualAttentionEvent]:
        """Get the most recent attention event of a specific type.
        
        Args:
            content_type: Type of content to search for
            within_seconds: Time window to search within
            
        Returns:
            Most recent matching event, or None if not found
        """
        cutoff = datetime.now() - timedelta(seconds=within_seconds)

        for event in reversed(self.attention_events):
            if event.timestamp < cutoff:
                break
            if event.content_type == content_type:
                return event

        return None

    def get_recent_critical(self, within_seconds: int = 300) -> Optional[VisualAttentionEvent]:
        """Get the most recent critical thing the user saw.
        
        Args:
            within_seconds: Time window to search within
            
        Returns:
            Most recent critical event, or None if not found
        """
        if self.last_critical_event and self.last_critical_event.is_recent(within_seconds):
            return self.last_critical_event
        return None

    def get_attention_in_space(self, space_id: int, within_seconds: int = 300) -> List[VisualAttentionEvent]:
        """Get all attention events in a specific space.
        
        Args:
            space_id: Desktop space ID to filter by
            within_seconds: Time window to search within
            
        Returns:
            List of attention events in the specified space
        """
        cutoff = datetime.now() - timedelta(seconds=within_seconds)

        return [
            event for event in self.attention_events
            if event.space_id == space_id and event.timestamp > cutoff
        ]

    def find_by_content(self, keywords: List[str], within_seconds: int = 300) -> List[VisualAttentionEvent]:
        """Find attention events matching keywords.
        
        Args:
            keywords: List of keywords to search for in content summaries
            within_seconds: Time window to search within
            
        Returns:
            List of matching attention events
        """
        cutoff = datetime.now() - timedelta(seconds=within_seconds)
        results = []

        for event in reversed(self.attention_events):
            if event.timestamp < cutoff:
                break

            # Check if any keyword matches
            content_lower = event.content_summary.lower()
            if any(kw.lower() in content_lower for kw in keywords):
                results.append(event)

        return results


# ============================================================================
# QUERY ANALYZER
# ============================================================================

class QueryAnalyzer:
    """Analyzes user queries to extract intent and references.

    This is the first stage of query processing: understanding WHAT the user
    is asking and WHAT they might be referring to.
    
    The analyzer uses pattern matching to identify:
    - Intent: What does the user want to do?
    - References: What are they talking about?
    - Keywords: Important terms in the query
    - Temporal markers: Time-related context
    
    Attributes:
        intent_patterns: Regex patterns for classifying user intents
        pronoun_patterns: Patterns for finding pronoun references
        temporal_patterns: Patterns for finding time-related phrases
    """

    def __init__(self):
        """Initialize the query analyzer with pattern dictionaries."""
        # Intent patterns (expanded, no hardcoding of specific errors)
        self.intent_patterns = {
            QueryIntent.DESCRIBE: [
                r'\bwhat\s+does\s+it\s+say\b',
                r'\b(say|show|display|read|see)\b',
            ],
            QueryIntent.EXPLAIN: [
                r'\b(explain|what is|tell me about|describe)\b',
                r'\bwhat\'?s\s+(this|that|it)\b',
            ],
            QueryIntent.LOCATE: [
                r'\b(where|find|locate|show me)\b',
            ],
            QueryIntent.STATUS: [
                r'\b(what\'?s\s+happening|going on|status|state)\b',
                r'\bwhat\s+am\s+i\b',
            ],
            QueryIntent.DIAGNOSE: [
                r'\b(what\'?s\s+wrong|problem|issue|broken|failed|error)\b',
                r'\bwhy\s+(did|does|is)\b',
            ],
            QueryIntent.FIX: [
                r'\b(how\s+to\s+fix|solve|resolve|repair)\b',
                r'\bhow\s+do\s+i\s+fix\b',
            ],
            QueryIntent.RECALL: [
                r'\b(again|repeat|what\s+was|remind|earlier)\b',
            ],
            QueryIntent.SUMMARIZE: [
                r'\b(summarize|summary|overview|recap)\b',
            ],
        }

        # Reference patterns
        self.pronoun_patterns = {
            'singular': r'\b(it|that|this)\b',
            'plural': r'\b(these|those|them)\b',
            'demonstrative': r'\b(that|this)\s+(\w+)',  # "that error", "this file"
        }

        # Temporal markers
        self.temporal_patterns = {
            'immediate': r'\b(just\s+now|right\s+now|now)\b',
            'recent': r'\b(earlier|before|recently|a\s+(?:few|couple)\s+\w+\s+ago)\b',
            'specific': r'\b(\d+)\s+(second|minute|hour)s?\s+ago\b',
        }

    def analyze(self, query: str) -> QueryParsed:
        """Analyze a query and extract intent + references.
        
        Args:
            query: User's natural language query
            
        Returns:
            ParsedQuery object containing analysis results
            
        Example:
            >>> analyzer = QueryAnalyzer()
            >>> result = analyzer.analyze("what does it say?")
            >>> print(result.intent)
            QueryIntent.DESCRIBE
        """
        query_lower = query.lower()

        # Determine intent
        intent = self._classify_intent(query_lower)

        # Extract references (pronouns, demonstratives)
        references = self._extract_references(query)

        # Extract keywords
        keywords = self._extract_keywords(query)

        # Extract temporal markers
        temporal = self._extract_temporal_marker(query_lower)

        # Calculate confidence
        confidence = self._calculate_confidence(intent, references, keywords)

        return QueryParsed(
            intent=intent,
            confidence=confidence,
            references=references,
            keywords=keywords,
            temporal_marker=temporal,
            original_query=query
        )

    def _classify_intent(self, query_lower: str) -> QueryIntent:
        """Classify the intent of the query.
        
        Args:
            query_lower: Lowercase version of the query
            
        Returns:
            Classified QueryIntent
        """
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        return QueryIntent.UNKNOWN

    def _extract_references(self, query: str) -> List[ParsedReference]:
        """Extract pronoun and demonstrative references.
        
        Args:
            query: Original query text
            
        Returns:
            List of ParsedReference objects found in the query
        """
        references = []

        # Pronouns
        for ref_type, pattern in self.pronoun_patterns.items():
            for match in re.finditer(pattern, query, re.IGNORECASE):
                ref_text = match.group(0)
                span = match.span()

                # Check for demonstrative (adjective + noun)
                if ref_type == 'demonstrative' and match.lastindex > 1:
                    entity_type = match.group(2)
                    references.append(ParsedReference(
                        reference_type=ReferenceType.DEMONSTRATIVE,
                        text=ref_text,
                        span=span,
                        entity_type=entity_type
                    ))
                else:
                    references.append(ParsedReference(
                        reference_type=ReferenceType.PRONOUN,
                        text=ref_text,
                        span=span
                    ))

        # Implicit references ("the error" without specifying which)
        implicit_pattern = r'\bthe\s+(\w+)'
        for match in re.finditer(implicit_pattern, query):
            entity = match.group(1)
            # Only if it's a noun that typically needs specification
            if entity in ['error', 'problem', 'issue', 'file', 'command', 'output', 'message']:
                references.append(ParsedReference(
                    reference_type=ReferenceType.IMPLICIT,
                    text=match.group(0),
                    span=match.span(),
                    entity_type=entity
                ))

        return references

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query.
        
        Args:
            query: Query text to analyze
            
        Returns:
            List of important keywords (stop words removed)
        """
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could',
                     'what', 'where', 'when', 'why', 'how', 'which', 'who'}

        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _extract_temporal_marker(self, query_lower: str) -> Optional[str]:
        """Extract temporal markers like 'just now', '2 minutes ago'.
        
        Args:
            query_lower: Lowercase query text
            
        Returns:
            Temporal marker string if found, None otherwise
        """
        for time_type, pattern in self.temporal_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                return match.group(0)
        return None

    def _calculate_confidence(self, intent: QueryIntent, references: List[ParsedReference], keywords: List[str]) -> float:
        """Calculate confidence in the analysis.
        
        Args:
            intent: Classified intent
            references: Found references
            keywords: Extracted keywords
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence

        if intent != QueryIntent.UNKNOWN:
            confidence += 0.2
        if references:
            confidence += 0.2
        if keywords:
            confidence += 0.1

        return min(1.0, confidence)


# ============================================================================
# IMPLICIT REFERENCE RESOLVER - Main System
# ============================================================================

class ImplicitReferenceResolver:
    """The main system that resolves implicit references like "it", "that", "the error".

    This combines:
    - Conversational context (what we just talked about)
    - Visual attention (what you just saw on screen)
    - Workspace context (what's happening in your spaces)
    - Temporal relevance (recent things are more likely)

    To answer queries like:
    - "what does it say?" → Finds the error you just saw
    - "explain that" → Explains the thing we just discussed
    - "how do I fix it?" → Fixes the problem from the last exchange
    
    Attributes:
        context_graph: MultiSpaceContextGraph instance for workspace context
        conversational_context: Tracks conversation history
        attention_tracker: Tracks visual attention events
        query_analyzer: Analyzes user queries for intent and references
    """

    def __init__(self, context_graph, conversational_context=None, attention_tracker=None):
        """Initialize the resolver.

        Args:
            context_graph: MultiSpaceContextGraph instance for workspace awareness
            conversational_context: Optional ConversationalContext instance
            attention_tracker: Optional VisualAttentionTracker instance
        """
        self.context_graph = context_graph
        self.conversational_context = conversational_context or ConversationalContext()
        self.attention_tracker = attention_tracker or VisualAttentionTracker()
        self.query_analyzer = QueryAnalyzer()

        logger.info("[IMPLICIT-RESOLVER] Initialized")

    async def resolve_query(self, query: str) -> Dict[str, Any]:
        """Resolve a query with implicit references.

        This is the main entry point. Returns a rich context dictionary.

        Args:
            query: User's natural language query

        Returns:
            Dict with resolved context including:
            - intent: What the user wants (string)
            - referent: What they're referring to (dict)
            - context: Full context about the referent (dict)
            - confidence: How confident we are (float)
            - response: Natural language response (string)
            - original_query: Original query text (string)
            
        Example:
            >>> resolver = ImplicitReferenceResolver(context_graph)
            >>> result = await resolver.resolve_query("what does it say?")
            >>> print(result['response'])
            "The error in Terminal says: ModuleNotFoundError: No module named 'requests'"
        """
        # Analyze the query
        parsed = self.query_analyzer.analyze(query)

        logger.debug(f"[IMPLICIT-RESOLVER] Query: '{query}'")
        logger.debug(f"[IMPLICIT-RESOLVER] Intent: {parsed.intent.value}, Confidence: {parsed.confidence:.2f}")
        logger.debug(f"[IMPLICIT-RESOLVER] References: {[r.text for r in parsed.references]}")

        # Resolve references to actual entities
        resolved_referent = await self._resolve_references(parsed)

        # Get full context about the referent
        full_context = await self._get_full_context(resolved_referent, parsed)

        # Generate response based on intent
        response = await self._generate_response(parsed, resolved_referent, full_context)

        # Record this turn in conversation context
        self.conversational_context.add_turn(
            user_query=query,
            jarvis_response=response,
            context_used=full_context
        )

        return {
            "intent": parsed.intent.value,
            "referent": resolved_referent,
            "context": full_context,
            "confidence": parsed.confidence,
            "response": response,
            "original_query": query
        }

    async def _resolve_references(self, parsed: QueryParsed) -> Dict[str, Any]:
        """Resolve references (it, that, the error) to actual entities.

        Strategy:
        1. Check conversation history (most recent mention)
        2. Check visual attention (what user just saw)
        3. Check workspace context (most significant recent event)
        4. Rank by temporal relevance + significance
        
        Args:
            parsed: Parsed query with references to resolve
            
        Returns:
            Dict containing the best candidate referent with metadata
        """
        candidates = []

        # Strategy 1: Conversational context
        if parsed.references:
            for ref in parsed.references: