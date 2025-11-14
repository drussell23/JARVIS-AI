"""
Temporal Query Handler v3.0 - INTELLIGENT TIME-BASED ANALYSIS
==============================================================

This module provides intelligent temporal query handling with ML-powered analysis capabilities.
It processes natural language queries about changes, patterns, and predictions across monitored spaces.

The handler integrates with multiple components:
- HybridProactiveMonitoringManager for ML-powered alerts
- ChangeDetectionManager for visual change detection
- ImplicitReferenceResolver for natural language understanding
- ConversationTracker for temporal context

Supported query types:
- ❓ "What changed in space 3?"
- ❓ "Has the error been fixed?"
- ❓ "What's new in the last 5 minutes?"
- ❓ "When did this error first appear?"
- ❓ "What patterns have you noticed?" (NEW)
- ❓ "Show me predicted events" (NEW)
- ❓ "Are there any anomalies?" (NEW)

**UPGRADED v3.0 Features**:
✅ Integration with HybridProactiveMonitoringManager (ML-powered!)
✅ Pattern-based timeline building (learns correlations)
✅ Predictive event analysis (anticipates future events)
✅ Anomaly-aware change detection (detects unusual behavior)
✅ Multi-space correlation tracking (cascading failures)
✅ Adaptive query complexity (fast for simple, deep for complex)
✅ Learned pattern integration (uses saved patterns)
✅ Alert quality scoring (prioritizes important changes)
✅ ImplicitReferenceResolver for natural language understanding
✅ Auto-build intelligent timeline from ML alerts

Architecture:

    User Query → ImplicitReferenceResolver → TemporalQueryHandler
         ↓               ↓                            ↓
    "what changed?"  Resolve refs          Classify query type
         ↓               ↓                            ↓
    "the error"     → "error #1234"        CHANGE_DETECTION
         ↓               ↓                            ↓
    space 3         → space_id: 3          Get monitoring data
         ↓               ↓                            ↓
    Resolved Query  HybridMonitoring       ML-powered analysis
         ↓               ↓                            ↓
    "What changed   ML alerts + patterns   Advanced diffing
     in space 3?"   + predictions               ↓
         ↓               ↓                   Anomaly detection
         └───────────────┴────────────────────────→ Intelligent Response
                                                     + Predictions
                                                     + Patterns
                                                     + Anomalies

Example:
    >>> handler = TemporalQueryHandler()
    >>> result = await handler.handle_query("What changed in space 3?")
    >>> print(result.summary)
    "Detected 2 change(s) in space 3: Content updated, Error resolved"
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
import hashlib
import json
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# TEMPORAL QUERY TYPES
# ============================================================================

class TemporalQueryType(Enum):
    """Types of temporal queries supported by the handler.
    
    Each type corresponds to a different analysis approach and response format.
    """
    CHANGE_DETECTION = auto()      # "What changed?"
    ERROR_TRACKING = auto()        # "Has the error been fixed?"
    TIMELINE = auto()              # "What's new in last 5 minutes?"
    FIRST_APPEARANCE = auto()      # "When did this first appear?"
    LAST_OCCURRENCE = auto()       # "When did I last see X?"
    COMPARISON = auto()            # "How is this different from before?"
    TREND_ANALYSIS = auto()        # "Is CPU usage increasing?"
    STATE_HISTORY = auto()         # "Show me history of space 3"
    MONITORING_REPORT = auto()     # "What has monitoring detected?"
    PATTERN_ANALYSIS = auto()      # "What patterns have you noticed?" (NEW v3.0)
    PREDICTIVE_ANALYSIS = auto()   # "Show me predicted events" (NEW v3.0)
    ANOMALY_ANALYSIS = auto()      # "Are there any anomalies?" (NEW v3.0)
    CORRELATION_ANALYSIS = auto()  # "How are spaces related?" (NEW v3.0)


class ChangeType(Enum):
    """Types of changes that can be detected in monitored spaces.
    
    Used to categorize and prioritize different kinds of state changes.
    """
    CONTENT_CHANGE = "content_change"        # Text/UI content changed
    LAYOUT_CHANGE = "layout_change"          # UI layout changed
    ERROR_APPEARED = "error_appeared"        # New error
    ERROR_RESOLVED = "error_resolved"        # Error fixed
    WINDOW_ADDED = "window_added"            # New window
    WINDOW_REMOVED = "window_removed"        # Window closed
    VALUE_CHANGED = "value_changed"          # Numeric value changed
    STATUS_CHANGED = "status_changed"        # Status indicator changed
    BUILD_COMPLETED = "build_completed"      # Build finished
    BUILD_FAILED = "build_failed"            # Build failed
    PROCESS_STARTED = "process_started"      # Process started
    PROCESS_STOPPED = "process_stopped"      # Process stopped
    ANOMALY_DETECTED = "anomaly_detected"    # Anomaly detected (NEW v3.0)
    PATTERN_RECOGNIZED = "pattern_recognized" # Pattern recognized (NEW v3.0)
    PREDICTIVE_EVENT = "predictive_event"    # Predicted event (NEW v3.0)
    CASCADING_FAILURE = "cascading_failure"  # Multi-space failure (NEW v3.0)
    NO_CHANGE = "no_change"                  # Nothing changed


@dataclass
class TimeRange:
    """Represents a time range for temporal queries.
    
    Attributes:
        start: Start datetime of the range
        end: End datetime of the range
    """
    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        """Get the duration of the time range.
        
        Returns:
            timedelta representing the duration
        """
        return self.end - self.start

    @property
    def duration_seconds(self) -> float:
        """Get the duration in seconds.
        
        Returns:
            Duration as a float number of seconds
        """
        return self.duration.total_seconds()

    @classmethod
    def from_natural_language(cls, text: str, reference_time: Optional[datetime] = None) -> 'TimeRange':
        """Parse natural language time references into a TimeRange.

        Supports various natural language patterns:
        - "last X minutes/hours/days"
        - "X minutes/hours/days ago"
        - "recently" / "just now" / "latest"
        - "today"
        - "since I last asked" (with conversation context)

        Args:
            text: Natural language time reference
            reference_time: Reference point for relative times (defaults to now)

        Returns:
            TimeRange object representing the parsed time period

        Example:
            >>> time_range = TimeRange.from_natural_language("last 5 minutes")
            >>> print(time_range.duration_seconds)
            300.0
        """
        now = reference_time or datetime.now()

        text_lower = text.lower()

        # "last X minutes/hours/days"
        if "last" in text_lower:
            if "minute" in text_lower:
                match = re.search(r'(\d+)\s*minute', text_lower)
                minutes = int(match.group(1)) if match else 5
                start = now - timedelta(minutes=minutes)
            elif "hour" in text_lower:
                match = re.search(r'(\d+)\s*hour', text_lower)
                hours = int(match.group(1)) if match else 1
                start = now - timedelta(hours=hours)
            elif "day" in text_lower:
                match = re.search(r'(\d+)\s*day', text_lower)
                days = int(match.group(1)) if match else 1
                start = now - timedelta(days=days)
            else:
                start = now - timedelta(minutes=5)  # Default

        # "since X minutes/hours ago"
        elif "ago" in text_lower:
            match = re.search(r'(\d+)\s*(\w+)\s*ago', text_lower)
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                if "minute" in unit:
                    start = now - timedelta(minutes=value)
                elif "hour" in unit:
                    start = now - timedelta(hours=value)
                elif "day" in unit:
                    start = now - timedelta(days=value)
                else:
                    start = now - timedelta(minutes=5)
            else:
                start = now - timedelta(minutes=5)

        # "recently" / "just now"
        elif any(word in text_lower for word in ["recently", "just now", "latest"]):
            start = now - timedelta(minutes=2)

        # "today"
        elif "today" in text_lower:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # "since I last asked" - requires conversation context (handled by caller)
        elif "since i" in text_lower or "since you" in text_lower or "since we" in text_lower:
            # Default to 5 minutes, will be overridden by conversation tracker
            start = now - timedelta(minutes=5)

        # Default: last 5 minutes
        else:
            start = now - timedelta(minutes=5)

        return cls(start=start, end=now)


@dataclass
class DetectedChange:
    """Represents a detected change between two states.
    
    Attributes:
        change_type: Type of change detected
        timestamp: When the change occurred
        space_id: ID of the space where change occurred
        description: Human-readable description of the change
        confidence: Confidence score (0.0 to 1.0)
        before_snapshot_id: ID of snapshot before change
        after_snapshot_id: ID of snapshot after change
        diff_regions: List of (x, y, w, h) tuples for visual differences
        metadata: Additional metadata about the change
    """
    change_type: ChangeType
    timestamp: datetime
    space_id: Optional[int]
    description: str
    confidence: float
    before_snapshot_id: Optional[str] = None
    after_snapshot_id: Optional[str] = None
    diff_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalQueryResult:
    """Result of a temporal query.
    
    Attributes:
        query_type: Type of query that was processed
        time_range: Time range that was analyzed
        changes: List of detected changes
        summary: Human-readable summary of results
        timeline: Chronological list of events
        metadata: Additional result metadata
    """
    query_type: TemporalQueryType
    time_range: TimeRange
    changes: List[DetectedChange]
    summary: str
    timeline: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TEMPORAL QUERY HANDLER (INTELLIGENT v3.0)
# ============================================================================

class TemporalQueryHandler:
    """Intelligent temporal query handler with ML-powered analysis.

    This class processes natural language queries about temporal changes and patterns
    in monitored spaces. It integrates with multiple AI and monitoring components to
    provide comprehensive temporal analysis.

    **UPGRADED v3.0 Features**:
    - Integrates with HybridProactiveMonitoringManager (ML-powered alerts!)
    - Pattern-based timeline building (learns event correlations)
    - Predictive event analysis (anticipates future events)
    - Anomaly-aware change detection (detects unusual behavior)
    - Multi-space correlation tracking (cascading failures)
    - Alert quality scoring (prioritizes important changes)
    - Learned pattern integration (uses saved patterns from disk)
    - ImplicitReferenceResolver for natural language understanding
    - ChangeDetectionManager for advanced diffing
    - ConversationTracker for temporal context

    **NEW Query Types**:
    - Pattern Analysis: "What patterns have you noticed?"
    - Predictive Analysis: "Show me predicted events"
    - Anomaly Analysis: "Are there any anomalies?"
    - Correlation Analysis: "How are spaces related?"

    **Intelligence Sources**:
    1. HybridMonitoring ML alerts (fast path + deep path + ML predictions)
    2. Learned pattern rules (saved to ~/.jarvis/monitoring_patterns.json)
    3. Anomaly profiles (statistical baselines)
    4. Alert correlation data (multi-space failures)

    Attributes:
        proactive_monitoring: HybridProactiveMonitoringManager instance
        change_detection: ChangeDetectionManager instance
        implicit_resolver: ImplicitReferenceResolver instance
        conversation_tracker: ConversationTracker instance
        monitoring_alerts: Deque of monitoring alerts for timeline building
        is_hybrid_monitoring: Whether using hybrid monitoring with ML features
        learned_patterns: List of learned patterns from disk
        anomaly_alerts: Deque of anomaly-specific alerts
        predictive_alerts: Deque of predictive alerts
        correlation_alerts: Deque of correlation alerts

    Example:
        >>> handler = TemporalQueryHandler()
        >>> result = await handler.handle_query("What changed in space 3?")
        >>> print(result.summary)
        "Detected 2 change(s) in space 3: Content updated, Error resolved"
    """

    def __init__(
        self,
        proactive_monitoring_manager=None,
        change_detection_manager=None,
        implicit_resolver=None,
        conversation_tracker=None
    ):
        """Initialize Intelligent TemporalQueryHandler v3.0.

        Args:
            proactive_monitoring_manager: HybridProactiveMonitoringManager instance (v2.0)
            change_detection_manager: ChangeDetectionManager instance
            implicit_resolver: ImplicitReferenceResolver instance
            conversation_tracker: ConversationTracker instance
        """
        self.proactive_monitoring = proactive_monitoring_manager
        self.change_detection = change_detection_manager
        self.implicit_resolver = implicit_resolver
        self.conversation_tracker = conversation_tracker

        # Alert history from monitoring (populated by monitoring callback)
        self.monitoring_alerts: deque[Dict[str, Any]] = deque(maxlen=500)  # Increased from 200

        # NEW v3.0: Intelligence data
        self.is_hybrid_monitoring = self._check_if_hybrid_monitoring()
        self.learned_patterns: List[Dict[str, Any]] = []
        self.anomaly_alerts: deque[Dict[str, Any]] = deque(maxlen=100)
        self.predictive_alerts: deque[Dict[str, Any]] = deque(maxlen=100)
        self.correlation_alerts: deque[Dict[str, Any]] = deque(maxlen=100)

        # Load learned patterns if available
        self._load_learned_patterns()

        if self.is_hybrid_monitoring:
            logger.info("[TEMPORAL-HANDLER] ✅ Initialized with HybridProactiveMonitoring (ML-powered!)")
        else:
            logger.info("[TEMPORAL-HANDLER] Initialized with ProactiveMonitoring integration")

    def _check_if_hybrid_monitoring(self) -> bool:
        """Check if we're using HybridProactiveMonitoringManager.
        
        Returns:
            True if using hybrid monitoring with ML features, False otherwise
        """
        if self.proactive_monitoring:
            # Check if it has ML features (hybrid manager)
            return hasattr(self.proactive_monitoring, 'enable_ml') and \
                   hasattr(self.proactive_monitoring, '_pattern_rules')
        return False

    def _load_learned_patterns(self):
        """Load learned patterns from disk (v3.0).
        
        Attempts to load previously saved patterns from ~/.jarvis/learned_patterns.json
        to enable pattern-based predictions and analysis.
        """
        try:
            import os
            patterns_file = os.path.expanduser('~/.jarvis/learned_patterns.json')

            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    self.learned_patterns = json.load(f)
                    logger.info(f"[TEMPORAL-HANDLER] Loaded {len(self.learned_patterns)} learned patterns from disk")
        except Exception as e:
            logger.warning(f"[TEMPORAL-HANDLER] Failed to load learned patterns: {e}")

    def set_proactive_monitoring(self, manager):
        """Inject ProactiveMonitoringManager dependency.
        
        Args:
            manager: ProactiveMonitoringManager or HybridProactiveMonitoringManager instance
        """
        self.proactive_monitoring = manager
        logger.info("[TEMPORAL-HANDLER] ProactiveMonitoringManager injected")

    def set_change_detection(self, manager):
        """Inject ChangeDetectionManager dependency.
        
        Args:
            manager: ChangeDetectionManager instance
        """
        self.change_detection = manager
        logger.info("[TEMPORAL-HANDLER] ChangeDetectionManager injected")

    def set_implicit_resolver(self, resolver):
        """Inject ImplicitReferenceResolver dependency.
        
        Args:
            resolver: ImplicitReferenceResolver instance
        """
        self.implicit_resolver = resolver
        logger.info("[TEMPORAL-HANDLER] ImplicitReferenceResolver injected")

    def set_conversation_tracker(self, tracker):
        """Inject ConversationTracker dependency.
        
        Args:
            tracker: ConversationTracker instance
        """
        self.conversation_tracker = tracker
        logger.info("[TEMPORAL-HANDLER] ConversationTracker injected")

    def register_monitoring_alert(self, alert: Dict[str, Any]):
        """Register a monitoring alert for timeline building (v3.0 Enhanced).

        Called by HybridProactiveMonitoringManager's alert callback to populate
        the alert history used for temporal analysis.

        Args:
            alert: Alert dictionary with keys:
                - space_id: int - ID of the space where alert occurred
                - event_type: str - Type of event (e.g., "error_appeared")
                - message: str - Human-readable alert message
                - priority: str - Alert priority level
                - timestamp: datetime - When the alert occurred
                - metadata: dict - Additional alert metadata
                - detection_method: str - NEW v3.0: "fast", "deep", "ml", "predictive"
                - predicted: bool - NEW v3.0: True if predictive alert
                - correlation_id: str - NEW v3.0: Groups related alerts

        Example:
            >>> alert = {
            ...     'space_id': 3,
            ...     'event_type': 'error_appeared',
            ...     'message': 'Build failed in terminal',
            ...     'priority': 'HIGH',
            ...     'timestamp': datetime.now()
            ... }
            >>> handler.register_monitoring_alert(alert)
        """
        # Add to main alerts
        self.monitoring_alerts.append(alert)

        # NEW v3.0: Categorize alerts by type for faster querying
        event_type = alert.get('event_type', '')
        detection_method = alert.get('metadata', {}).get('detection_method', 'unknown')

        # Anomaly alerts
        if 'anomaly' in event_type.lower():
            self.anomaly_alerts.append(alert)

        # Predictive alerts
        if alert.get('predicted', False) or detection_method == 'predictive':
            self.predictive_alerts.append(alert)

        # Correlation alerts (cascading failures)
        if alert.get('correlation_id') or 'cascading' in event_type.lower():
            self.correlation_alerts.append(alert)

        logger.debug(f"[TEMPORAL-HANDLER] Registered {detection_method} alert: {alert.get('message')}")

    async def handle_query(
        self,
        query: str,
        space_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TemporalQueryResult:
        """Handle a temporal query with full context resolution.

        This is the main entry point for processing natural language temporal queries.
        It performs query classification, reference resolution, and delegates to
        specialized handlers based on query type.

        Args:
            query: Natural language query (e.g., "What changed in space 3?")
            space_id: Optional space ID context for the query
            context: Optional additional context dictionary

        Returns:
            TemporalQueryResult containing changes, timeline, and summary

        Raises:
            Exception: If query processing fails critically

        Example:
            >>> result = await handler.handle_query("What changed in the last 5 minutes?")
            >>> print(result.summary)
            "Found 3 changes in the last 5 minutes: 2 content changes, 1 error resolved"
        """
        logger.info(f"[TEMPORAL-HANDLER] Handling query: '{query}' (space_id={space_id})")

        # Step 1: Classify query type
        query_type = self._classify_query_type(query)
        logger.debug(f"[TEMPORAL-HANDLER] Query type: {query_type.name}")

        # Step 2: Extract time range
        time_range = TimeRange.from_natural_language(query)

        # Step 3: Resolve references using ImplicitReferenceResolver
        resolved_query = await self._resolve_references(query, space_id, context)
        resolved_space_id = resolved_query.get('space_id', space_id)

        # Step 4: Adjust time range based on conversation context
        if self.conversation_tracker:
            time_range = await self._adjust_time_range_from_conversation(
                query, time_range, resolved_query
            )

        logger.debug(f"[TEMPORAL-HANDLER] Time range: {time_range.start} to {time_range.end}")

        # Step 5: Execute query based on type
        if query_type == TemporalQueryType.CHANGE_DETECTION:
            result = await self._handle_change_detection(resolved_query, time_range, resolved_space_id)

        elif query_type == TemporalQueryType.ERROR_TRACKING:
            result = await self._handle_error_tracking(resolved_query, time_range, resolved_space_id)

        elif query_type == TemporalQueryType.TIMELINE:
            result = await self._handle_timeline(resolved_query, time_range, resolved_space_id)

        elif query_type == TemporalQueryType.FIRST_APPEARANCE:
            result = await self._handle_first_appearance(resolved_query, time_range, resolved_space_id)

        elif query_type == TemporalQueryType.COMPARISON:
            result = await self._handle_comparison(resolved_query, time_range, resolved_space_id)

        elif query_type == TemporalQueryType.MONITORING_REPORT:
            result = await self._handle_monitoring_report(resolved_query, time_range, resolved_space_id)

        # NEW v3.0: Intelligent query types
        elif query_type == TemporalQueryType.PATTERN_ANALYSIS:
            result = await self._handle_pattern_analysis(resolved_query, time_range, resolved_space_id)

        elif query_type == TemporalQueryType.PREDICTIVE_ANALYSIS:
            result = await self._handle_predictive_analysis(resolved_query, time_range, resolved_space_id)

        elif query_type == TemporalQueryType.ANOMALY_ANALYSIS:
            result = await self._handle_anomaly_analysis(resolved_query, time_range, resolved_space_id)

        elif query_type == TemporalQueryType.CORRELATION_ANALYSIS:
            result = await self._handle_correlation_analysis(resolved_query, time_range, resolved_space_id)

        else:
            result = await self._handle_generic_temporal_query(resolved_query, time_range, resolved_space_id)

        logger.info(f"[TEMPORAL-HANDLER] Query completed: {result.summary}")
        return result

    def _classify_query_type(self, query: str) -> TemporalQueryType:
        """Classify the type of temporal query (v3.0 Enhanced).

        Uses pattern matching to determine query intent, including NEW intelligent types.
        The classification drives which specialized handler will process the query.

        Args:
            query: Natural language query string

        Returns:
            TemporalQueryType enum value representing the query intent

        Example:
            >>> query_type = handler._classify_query_type("What patterns have you noticed?")
            >>> print(query_type)
            TemporalQueryType.PATTERN_ANALYSIS
        """
        query_lower = query.lower()

        # NEW v3.0: Pattern analysis
        if any(word in query_lower for word in ["pattern", "patterns", "correlation", "noticed", "learns", "relationship"]):
            return TemporalQueryType.PATTERN_ANALYSIS

        # NEW v3.0: Predictive analysis
        if any(word in query_lower for word in ["predict", "predicted", "will", "going to", "expect", "anticipate", "forecast"]):
            return TemporalQueryType.PREDICTIVE_ANALYSIS

        # NEW v3.0: Anomaly analysis
        if any(word in query_lower for word in ["anomaly", "anomalies", "unusual", "strange", "unexpected", "outlier"]):
            return TemporalQueryType.ANOMALY_ANALYSIS

        # NEW v3.0: Correlation analysis
        if any(word in query_lower for word in ["related", "related to", "connection", "cascading", "multi-space", "across spaces"]):
            return TemporalQueryType.CORRELATION_ANALYSIS

        # Monitoring report
        if any(word in query_lower for word in ["what has monitoring", "monitoring detected", "monitoring found", "monitoring report"]):
            return TemporalQueryType.MONITORING_REPORT

        # Change detection
        if any(word in query_lower for word in ["changed", "change", "different", "new"]):
            return TemporalQueryType.CHANGE_DETECTION

        # Error tracking
        if any(word in query_lower for word in ["error", "fixed", "bug", "issue", "resolved"]):
            return TemporalQueryType.ERROR_TRACKING

        # Timeline
        if any(word in query_lower for word in ["timeline", "history", "show me", "what happened"]):
            return TemporalQueryType.TIMELINE

        # First appearance
        if any(word in query_lower for word in ["when", "first", "appeared", "started"]):
            return TemporalQueryType.FIRST_APPEARANCE

        # Comparison
        if any(word in query_lower for word in ["compare", "vs", "versus", "before", "after"]):
            return TemporalQueryType.COMPARISON

        # Default
        return TemporalQueryType.TIMELINE

    async def _resolve_references(
        self,
        query: str,
        space_id: Optional[int],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use ImplicitReferenceResolver to resolve references in query.

        Resolves ambiguous references like:
        - "the error" → specific error ID
        - "it" → most recent entity
        - "that space" → specific space ID
        - "since I last asked" → specific timestamp

        Args:
            query: Original natural language query
            space_id: Optional space ID context
            context: Optional additional context

        Returns:
            Dictionary containing resolved query information with keys:
            - original_query: Original query string
            - intent: Resolved intent
            - referents: List of resolved referents
            - space_id: Resolved space ID
            - resolved_entities: Dictionary of resolved entities
            - temporal_markers: List of temporal markers found

        Example:
            >>> resolved = await handler._resolve_references("Has the error been fixed?", 3, {})
            >>> print(resolved['resolved_entities'])
            {'error': 'build_error_1234'}
        """
        if self.implicit_resolver:
            try:
                resolution = await self.implicit_resolver.resolve_query(
                    query,
                    context=context or {}
                )

                return {
                    'original_query': query,
                    'intent': resolution.get('intent'),
                    'referents': resolution.get('referents', []),
                    'space_id': space_id or resolution.get('space_id'),
                    'resolved_entities': resolution.get('entities', {}),
                    'temporal_markers': resolution.get('temporal_markers', [])
                }
            except Exception as e:
                logger.warning(f"[TEMPORAL-HANDLER] ImplicitReferenceResolver failed: {e}")

        # Fallback: simple resolution
        return {
            'original_query': query,
            'space_id': space_id,
            'referents': [],
            'resolved_entities': {},
            'temporal_markers': []
        }

    async def _adjust_time_range_from_conversation(
        self,
        query: str,
        time_range: TimeRange,
        resolved_query: Dict[str, Any]
    ) -> TimeRange:
        """Adjust time range based on conversation history.

        Handles queries with conversational temporal references like:
        - "since I last asked" → time of last conversation turn
        - "since we talked about X" → time of conversation about X

        Args:
            query: Original query string
            time_range: Initially parsed time range
            resolved_query: Resolved query information

        Returns:
            Adjusted TimeRange based on conversation context

        Example:
            >>> # If user asked something 10 minutes ago
            >>> adjusted = await handler._adjust_time_range_from_conversation(
            ...     "What changed since I last asked?", time_range, resolved_query
            ... )
            >>> print(adjusted.duration_seconds)
            600.0  # 10 minutes
        """
        if not self.conversation_tracker:
            return time_range

        query_lower = query.lower()

        # "since I last asked"
        if "since i" in query_lower or "since you" in query_lower or "since we" in query_lower:
            last_turn = self.conversation_tracker.get_last_turn()
            if last_turn:
                logger.debug(f"[TEMPORAL-HANDLER] Adjusting time range to 'since last turn' at {last_turn.get('timestamp')}")
                time_range = TimeRange(
                    start=last_turn.get('timestamp', time_range.start),
                    end=time_range.end
                )

        return time_range

    async def _handle_change_detection(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """Handle 'What changed?' queries with ML-powered change detection.

        Uses multiple detection strategies:
        1. Monitoring alerts (ML-powered anomaly detection)
        2. State snapshots (cached for instant results)
        3. Pattern recognition (learned behaviors)
        4. Visual diff analysis (content/layout changes)

        Args:
            resolved_query: Resolved query with references
            time_range: Time range for detection
            space_id: Optional space ID filter

        Returns:
            TemporalQueryResult with detected changes and ML insights
        """
        changes = []
        detected_anomalies = []
        recognized_patterns = []
        confidence_scores = []

        try:
            # Strategy 1: Get ML-powered monitoring alerts
            if self.monitoring_manager:
                try:
                    alerts = await self._get_monitoring_alerts(time_range, space_id)
                    for alert in alerts:
                        change = DetectedChange(
                            change_type=self._classify_alert_type(alert),
                            timestamp=alert.get("timestamp", datetime.now()),
                            space_id=alert.get("space_id", space_id),
                            description=alert.get("message", "Change detected by monitoring"),
                            confidence=alert.get("confidence", 0.7),
                            severity=alert.get("severity", "medium"),
                            metadata={
                                "source": "ml_monitoring",
                                "alert_id": alert.get("id"),
                                "pattern_id": alert.get("pattern_id"),
                                "is_anomaly": alert.get("is_anomaly", False)
                            }
                        )
                        changes.append(change)
                        confidence_scores.append(change.confidence)

                        # Track anomalies separately
                        if alert.get("is_anomaly"):
                            detected_anomalies.append(alert)

                        # Track recognized patterns
                        if alert.get("pattern_id"):
                            recognized_patterns.append(alert.get("pattern_id"))

                except Exception as e:
                    logger.warning(f"Failed to get monitoring alerts: {e}")

            # Strategy 2: State snapshot comparison
            if self.monitoring_manager:
                try:
                    snapshots = await self._get_state_snapshots(time_range, space_id)
                    if len(snapshots) >= 2:
                        # Compare first and last snapshot
                        state_changes = await self._compare_snapshots(
                            snapshots[0],
                            snapshots[-1],
                            space_id
                        )
                        changes.extend(state_changes)
                        confidence_scores.extend([c.confidence for c in state_changes])
                except Exception as e:
                    logger.warning(f"Failed snapshot comparison: {e}")

            # Strategy 3: Pattern-based change detection
            if self.monitoring_manager and hasattr(self.monitoring_manager, 'pattern_engine'):
                try:
                    pattern_changes = await self._detect_pattern_changes(
                        time_range,
                        space_id,
                        recognized_patterns
                    )
                    changes.extend(pattern_changes)
                    confidence_scores.extend([c.confidence for c in pattern_changes])
                except Exception as e:
                    logger.warning(f"Failed pattern detection: {e}")

            # Strategy 4: Anomaly-specific analysis
            if detected_anomalies:
                try:
                    anomaly_changes = await self._analyze_anomalies(
                        detected_anomalies,
                        space_id
                    )
                    changes.extend(anomaly_changes)
                    confidence_scores.extend([c.confidence for c in anomaly_changes])
                except Exception as e:
                    logger.warning(f"Failed anomaly analysis: {e}")

            # Calculate overall confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

            # Build summary
            summary = self._build_change_summary(
                changes,
                detected_anomalies,
                recognized_patterns,
                time_range
            )

            # Sort changes by timestamp (newest first)
            changes.sort(key=lambda c: c.timestamp, reverse=True)

            return TemporalQueryResult(
                query_type="change_detection",
                time_range=time_range,
                results=changes,
                confidence=avg_confidence,
                summary=summary,
                metadata={
                    "total_changes": len(changes),
                    "anomalies_detected": len(detected_anomalies),
                    "patterns_recognized": len(set(recognized_patterns)),
                    "time_range_duration": time_range.duration_seconds,
                    "space_id": space_id,
                    "detection_strategies": [
                        "ml_monitoring",
                        "snapshot_comparison",
                        "pattern_recognition",
                        "anomaly_analysis"
                    ]
                }
            )

        except Exception as e:
            logger.error(f"Change detection failed: {e}", exc_info=True)
            return TemporalQueryResult(
                query_type="change_detection",
                time_range=time_range,
                results=[],
                confidence=0.0,
                summary="Failed to detect changes",
                metadata={"error": str(e)}
            )

    async def _get_monitoring_alerts(
        self,
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Get ML-powered monitoring alerts for the time range."""
        alerts = []
        try:
            if hasattr(self.monitoring_manager, 'get_alerts_in_range'):
                alerts = await self.monitoring_manager.get_alerts_in_range(
                    start_time=time_range.start,
                    end_time=time_range.end,
                    space_id=space_id
                )
        except Exception as e:
            logger.warning(f"Failed to get monitoring alerts: {e}")
        return alerts

    async def _get_state_snapshots(
        self,
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Get cached state snapshots from monitoring."""
        snapshots = []
        try:
            if hasattr(self.monitoring_manager, 'get_snapshots'):
                snapshots = await self.monitoring_manager.get_snapshots(
                    start_time=time_range.start,
                    end_time=time_range.end,
                    space_id=space_id
                )
        except Exception as e:
            logger.warning(f"Failed to get snapshots: {e}")
        return snapshots

    async def _compare_snapshots(
        self,
        snapshot1: Dict[str, Any],
        snapshot2: Dict[str, Any],
        space_id: Optional[int]
    ) -> List['DetectedChange']:
        """Compare two snapshots to detect changes."""
        changes = []

        # Compare content
        if snapshot1.get("content") != snapshot2.get("content"):
            changes.append(DetectedChange(
                change_type=ChangeType.CONTENT_CHANGE,
                timestamp=snapshot2.get("timestamp", datetime.now()),
                space_id=space_id,
                description="Content changed",
                confidence=0.9,
                severity="low",
                metadata={
                    "old_hash": hashlib.md5(str(snapshot1.get("content", "")).encode()).hexdigest()[:8],
                    "new_hash": hashlib.md5(str(snapshot2.get("content", "")).encode()).hexdigest()[:8]
                }
            ))

        # Compare layout
        if snapshot1.get("layout") != snapshot2.get("layout"):
            changes.append(DetectedChange(
                change_type=ChangeType.LAYOUT_CHANGE,
                timestamp=snapshot2.get("timestamp", datetime.now()),
                space_id=space_id,
                description="Layout changed",
                confidence=0.85,
                severity="medium",
                metadata={"source": "snapshot_diff"}
            ))

        # Compare status
        if snapshot1.get("status") != snapshot2.get("status"):
            changes.append(DetectedChange(
                change_type=ChangeType.STATUS_CHANGED,
                timestamp=snapshot2.get("timestamp", datetime.now()),
                space_id=space_id,
                description=f"Status changed: {snapshot1.get('status')} → {snapshot2.get('status')}",
                confidence=0.95,
                severity="high",
                metadata={
                    "old_status": snapshot1.get("status"),
                    "new_status": snapshot2.get("status")
                }
            ))

        return changes

    async def _detect_pattern_changes(
        self,
        time_range: TimeRange,
        space_id: Optional[int],
        recognized_patterns: List[str]
    ) -> List['DetectedChange']:
        """Detect changes based on learned patterns."""
        changes = []
        try:
            # Check if patterns have changed behavior
            for pattern_id in set(recognized_patterns):
                changes.append(DetectedChange(
                    change_type=ChangeType.PATTERN_RECOGNIZED,
                    timestamp=datetime.now(),
                    space_id=space_id,
                    description=f"Pattern #{pattern_id} detected",
                    confidence=0.8,
                    severity="low",
                    metadata={"pattern_id": pattern_id}
                ))
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
        return changes

    async def _analyze_anomalies(
        self,
        anomalies: List[Dict[str, Any]],
        space_id: Optional[int]
    ) -> List['DetectedChange']:
        """Analyze detected anomalies for significant changes."""
        changes = []
        for anomaly in anomalies:
            changes.append(DetectedChange(
                change_type=ChangeType.ANOMALY_DETECTED,
                timestamp=anomaly.get("timestamp", datetime.now()),
                space_id=space_id,
                description=f"Anomaly: {anomaly.get('message', 'Unusual behavior detected')}",
                confidence=anomaly.get("confidence", 0.75),
                severity="high",
                metadata={
                    "anomaly_type": anomaly.get("type"),
                    "deviation_score": anomaly.get("deviation_score")
                }
            ))
        return changes

    def _classify_alert_type(self, alert: Dict[str, Any]) -> ChangeType:
        """Classify an alert into a change type."""
        alert_type = alert.get("type", "").lower()
        message = alert.get("message", "").lower()

        if "error" in message and "appeared" in message:
            return ChangeType.ERROR_APPEARED
        elif "error" in message and ("resolved" in message or "fixed" in message):
            return ChangeType.ERROR_RESOLVED
        elif "anomaly" in alert_type:
            return ChangeType.ANOMALY_DETECTED
        elif "pattern" in alert_type:
            return ChangeType.PATTERN_RECOGNIZED
        elif "build" in message and "completed" in message:
            return ChangeType.BUILD_COMPLETED
        elif "build" in message and "failed" in message:
            return ChangeType.BUILD_FAILED
        else:
            return ChangeType.CONTENT_CHANGE

    def _build_change_summary(
        self,
        changes: List['DetectedChange'],
        anomalies: List[Dict],
        patterns: List[str],
        time_range: TimeRange
    ) -> str:
        """Build a human-readable summary of detected changes."""
        if not changes:
            return f"No changes detected in the last {time_range.duration_seconds:.0f} seconds"

        summary_parts = [f"Detected {len(changes)} change(s)"]

        # Group by type
        change_types = {}
        for change in changes:
            change_type = change.change_type.value if hasattr(change.change_type, 'value') else str(change.change_type)
            change_types[change_type] = change_types.get(change_type, 0) + 1

        # Add top change types
        top_types = sorted(change_types.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_types:
            type_desc = ", ".join([f"{count} {ctype.replace('_', ' ')}" for ctype, count in top_types])
            summary_parts.append(type_desc)

        # Add anomaly info
        if anomalies:
            summary_parts.append(f"{len(anomalies)} anomalies")

        # Add pattern info
        if patterns:
            summary_parts.append(f"{len(set(patterns))} patterns recognized")

        return "; ".join(summary_parts)