"""
Query Context Manager for JARVIS Vision System
Tracks user patterns, context, and conversation flow to improve classification.

Provides rich context for intelligent query classification.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, Counter
from enum import Enum

logger = logging.getLogger(__name__)


class UserPattern(Enum):
    """Detected user usage patterns"""
    MORNING_OVERVIEW = "morning_overview"  # Morning workspace overviews
    DEEP_WORK_SESSION = "deep_work_session"  # Focused coding/work
    MULTI_TASK = "multi_task"  # Switching between many spaces
    VISUAL_FOCUSED = "visual_focused"  # Frequently asks visual queries
    METADATA_FOCUSED = "metadata_focused"  # Prefers quick metadata
    UNKNOWN = "unknown"


@dataclass
class QueryContext:
    """Context information for a query"""
    query: str
    intent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    active_space: Optional[int] = None
    total_spaces: int = 0
    active_apps: List[str] = field(default_factory=list)
    response_satisfied: bool = False
    latency_ms: float = 0


class QueryContextManager:
    """
    Manages query context and tracks user patterns.
    Provides context-aware information to improve classification.
    """

    def __init__(self, max_history: int = 100):
        """
        Initialize context manager

        Args:
            max_history: Maximum number of queries to keep in history
        """
        self.max_history = max_history

        # Query history
        self._query_history: deque = deque(maxlen=max_history)

        # Session tracking
        self._session_start = datetime.now()
        self._current_session_id = int(time.time())

        # User pattern detection
        self._detected_pattern: UserPattern = UserPattern.UNKNOWN
        self._pattern_confidence: float = 0.0
        self._pattern_last_updated: datetime = datetime.now()

        # Context state
        self._current_space: Optional[int] = None
        self._total_spaces: int = 0
        self._active_apps: List[str] = []
        self._last_query_time: Optional[datetime] = None

        # Intent tracking
        self._recent_intents: deque = deque(maxlen=20)
        self._intent_distribution: Counter = Counter()

        # Time-based patterns
        self._time_patterns: Dict[str, List[str]] = {
            'morning': [],  # 6am-10am
            'midday': [],  # 10am-2pm
            'afternoon': [],  # 2pm-6pm
            'evening': [],  # 6pm-10pm
            'night': []  # 10pm-6am
        }

        logger.info("[CONTEXT] Query context manager initialized")

    def record_query(
        self,
        query: str,
        intent: Optional[str] = None,
        active_space: Optional[int] = None,
        total_spaces: int = 0,
        active_apps: Optional[List[str]] = None,
        response_latency_ms: float = 0
    ) -> QueryContext:
        """
        Record a query and its context

        Args:
            query: User's query
            intent: Classified intent
            active_space: Current active desktop space
            total_spaces: Total number of desktop spaces
            active_apps: List of active applications
            response_latency_ms: Response time

        Returns:
            QueryContext object
        """
        context = QueryContext(
            query=query,
            intent=intent,
            timestamp=datetime.now(),
            active_space=active_space,
            total_spaces=total_spaces,
            active_apps=active_apps or [],
            latency_ms=response_latency_ms
        )

        # Update history
        self._query_history.append(context)

        # Update state
        self._current_space = active_space
        self._total_spaces = total_spaces
        self._active_apps = active_apps or []
        self._last_query_time = context.timestamp

        # Track intent
        if intent:
            self._recent_intents.append(intent)
            self._intent_distribution[intent] += 1

        # Track time-based patterns
        self._record_time_pattern(query)

        # Update pattern detection every 10 queries
        if len(self._query_history) % 10 == 0:
            self._update_pattern_detection()

        logger.debug(f"[CONTEXT] Recorded query: {query[:50]}... (intent: {intent})")

        return context

    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """
        Get contextual information for classifying a query

        Args:
            query: User's query

        Returns:
            Dictionary with context information
        """
        context = {
            # Current state
            'active_space': self._current_space,
            'total_spaces': self._total_spaces,
            'active_apps': self._active_apps,

            # Recent activity
            'recent_intent': self._recent_intents[-1] if self._recent_intents else None,
            'recent_intents': list(self._recent_intents)[-5:],

            # Time-based
            'time_since_last_query': (
                (datetime.now() - self._last_query_time).total_seconds()
                if self._last_query_time
                else 0
            ),
            'time_of_day': self._get_time_period(),
            'session_duration_minutes': (datetime.now() - self._session_start).total_seconds() / 60,

            # Pattern detection
            'user_pattern': self._detected_pattern.value,
            'pattern_confidence': self._pattern_confidence,

            # Intent distribution
            'intent_distribution': dict(self._intent_distribution),
            'dominant_intent': self._get_dominant_intent(),

            # Query characteristics
            'query_length': len(query.split()),
            'is_follow_up': self._is_follow_up_query(query),
            'similar_recent_queries': self._find_similar_recent_queries(query),
        }

        return context

    def mark_satisfied(self, query: str, satisfied: bool = True):
        """
        Mark whether user was satisfied with query response

        Args:
            query: The query
            satisfied: Whether user was satisfied
        """
        # Find most recent matching query in history
        for ctx in reversed(self._query_history):
            if ctx.query == query:
                ctx.response_satisfied = satisfied
                logger.debug(f"[CONTEXT] Marked query satisfaction: {satisfied}")
                break

    def get_user_preferences(self) -> Dict[str, Any]:
        """
        Analyze user preferences from history

        Returns:
            Dictionary with user preferences
        """
        if not self._query_history:
            return {
                'preferred_intent': None,
                'avg_query_frequency': 0,
                'typical_time_of_day': None,
                'multi_space_user': False
            }

        # Preferred intent
        preferred_intent = self._get_dominant_intent()

        # Query frequency (queries per hour)
        session_hours = (datetime.now() - self._session_start).total_seconds() / 3600
        avg_frequency = len(self._query_history) / max(session_hours, 0.1)

        # Typical time of day
        time_counts = Counter()
        for ctx in self._query_history:
            time_period = self._get_time_period(ctx.timestamp)
            time_counts[time_period] += 1
        typical_time = time_counts.most_common(1)[0][0] if time_counts else None

        # Multi-space usage
        space_changes = sum(
            1 for i in range(1, len(self._query_history))
            if (self._query_history[i].active_space != self._query_history[i-1].active_space)
        )
        multi_space_user = space_changes > len(self._query_history) * 0.3

        return {
            'preferred_intent': preferred_intent,
            'avg_query_frequency': avg_frequency,
            'typical_time_of_day': typical_time,
            'multi_space_user': multi_space_user,
            'detected_pattern': self._detected_pattern.value,
            'pattern_confidence': self._pattern_confidence
        }

    def _update_pattern_detection(self):
        """Detect user usage patterns from history"""
        if len(self._query_history) < 5:
            return

        # Morning overview pattern
        morning_queries = [
            ctx for ctx in self._query_history
            if self._get_time_period(ctx.timestamp) == 'morning'
        ]
        morning_overview_score = sum(
            1 for ctx in morning_queries
            if ctx.intent in ['metadata_only', 'deep_analysis']
        ) / max(len(morning_queries), 1)

        # Deep work session (consistent space, low query frequency)
        recent_20 = list(self._query_history)[-20:]
        space_changes = sum(
            1 for i in range(1, len(recent_20))
            if recent_20[i].active_space != recent_20[i-1].active_space
        )
        deep_work_score = 1.0 - (space_changes / max(len(recent_20), 1))

        # Multi-task pattern (frequent space changes)
        multi_task_score = space_changes / max(len(recent_20), 1)

        # Visual-focused pattern
        visual_queries = sum(
            1 for ctx in self._query_history
            if ctx.intent in ['visual_analysis', 'deep_analysis']
        )
        visual_score = visual_queries / len(self._query_history)

        # Metadata-focused pattern
        metadata_queries = sum(
            1 for ctx in self._query_history
            if ctx.intent == 'metadata_only'
        )
        metadata_score = metadata_queries / len(self._query_history)

        # Determine pattern
        scores = [
            (UserPattern.MORNING_OVERVIEW, morning_overview_score * 0.3),  # Less weight, time-specific
            (UserPattern.DEEP_WORK_SESSION, deep_work_score * 0.7),
            (UserPattern.MULTI_TASK, multi_task_score * 0.8),
            (UserPattern.VISUAL_FOCUSED, visual_score * 0.6),
            (UserPattern.METADATA_FOCUSED, metadata_score * 0.6),
        ]

        pattern, confidence = max(scores, key=lambda x: x[1])

        # Only update if confidence is reasonable
        if confidence > 0.5:
            self._detected_pattern = pattern
            self._pattern_confidence = min(confidence, 0.95)
            self._pattern_last_updated = datetime.now()

            logger.info(
                f"[CONTEXT] Detected user pattern: {pattern.value} "
                f"(confidence: {confidence:.2f})"
            )

    def _get_dominant_intent(self) -> Optional[str]:
        """Get the most common intent from history"""
        if not self._intent_distribution:
            return None
        return self._intent_distribution.most_common(1)[0][0]

    def _is_follow_up_query(self, query: str) -> bool:
        """Detect if query is a follow-up to previous query"""
        if not self._query_history:
            return False

        # Check time since last query
        if self._last_query_time:
            time_diff = (datetime.now() - self._last_query_time).total_seconds()
            if time_diff > 60:  # More than 1 minute - not a follow-up
                return False

        # Check for follow-up indicators
        follow_up_words = ['also', 'and', 'what about', 'how about', 'plus', 'additionally']
        query_lower = query.lower()

        return any(word in query_lower for word in follow_up_words)

    def _find_similar_recent_queries(self, query: str, limit: int = 5) -> List[str]:
        """Find similar recent queries"""
        if not self._query_history:
            return []

        query_words = set(query.lower().split())
        similar = []

        for ctx in reversed(list(self._query_history)[-20:]):
            ctx_words = set(ctx.query.lower().split())
            overlap = len(query_words & ctx_words)

            if overlap >= 2:  # At least 2 common words
                similar.append(ctx.query)

            if len(similar) >= limit:
                break

        return similar

    def _record_time_pattern(self, query: str):
        """Record query in time-based pattern tracking"""
        time_period = self._get_time_period()
        if time_period in self._time_patterns:
            self._time_patterns[time_period].append(query)

            # Keep only recent patterns (last 50 per period)
            if len(self._time_patterns[time_period]) > 50:
                self._time_patterns[time_period] = self._time_patterns[time_period][-50:]

    def _get_time_period(self, timestamp: Optional[datetime] = None) -> str:
        """Get time period for a timestamp"""
        if timestamp is None:
            timestamp = datetime.now()

        hour = timestamp.hour

        if 6 <= hour < 10:
            return 'morning'
        elif 10 <= hour < 14:
            return 'midday'
        elif 14 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            'session_id': self._current_session_id,
            'session_duration_minutes': (datetime.now() - self._session_start).total_seconds() / 60,
            'total_queries': len(self._query_history),
            'unique_intents': len(set(self._recent_intents)),
            'detected_pattern': self._detected_pattern.value,
            'pattern_confidence': self._pattern_confidence,
            'user_preferences': self.get_user_preferences(),
            'intent_distribution': dict(self._intent_distribution)
        }

    def reset_session(self):
        """Reset session tracking"""
        self._session_start = datetime.now()
        self._current_session_id = int(time.time())
        logger.info("[CONTEXT] Session reset")


# Singleton instance
_context_manager: Optional[QueryContextManager] = None


def get_context_manager() -> QueryContextManager:
    """Get or create the singleton context manager"""
    global _context_manager

    if _context_manager is None:
        _context_manager = QueryContextManager()

    return _context_manager
