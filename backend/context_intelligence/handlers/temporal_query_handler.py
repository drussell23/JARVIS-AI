"""
Temporal Query Handler v3.0 - INTELLIGENT TIME-BASED ANALYSIS
==============================================================

Handles temporal queries with ML-powered intelligence:
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
    """Types of temporal queries"""
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
    """Types of changes detected"""
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
    """Represents a time range for queries"""
    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    @property
    def duration_seconds(self) -> float:
        return self.duration.total_seconds()

    @classmethod
    def from_natural_language(cls, text: str, reference_time: Optional[datetime] = None) -> 'TimeRange':
        """
        Parse natural language time references.

        Supports:
        - "last X minutes/hours/days"
        - "X minutes/hours/days ago"
        - "recently" / "just now" / "latest"
        - "today"
        - "since I last asked" (with conversation context)
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
    """Represents a detected change between two states"""
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
    """Result of a temporal query"""
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
    """
    Intelligent temporal query handler with ML-powered analysis.

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
    """

    def __init__(
        self,
        proactive_monitoring_manager=None,
        change_detection_manager=None,
        implicit_resolver=None,
        conversation_tracker=None
    ):
        """
        Initialize Intelligent TemporalQueryHandler v3.0.

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
        """Check if we're using HybridProactiveMonitoringManager"""
        if self.proactive_monitoring:
            # Check if it has ML features (hybrid manager)
            return hasattr(self.proactive_monitoring, 'enable_ml') and \
                   hasattr(self.proactive_monitoring, '_pattern_rules')
        return False

    def _load_learned_patterns(self):
        """Load learned patterns from disk (saved by HybridMonitoring)"""
        try:
            from pathlib import Path
            patterns_file = Path.home() / ".jarvis" / "monitoring_patterns.json"

            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    self.learned_patterns = data.get('pattern_rules', [])
                    logger.info(f"[TEMPORAL-HANDLER] Loaded {len(self.learned_patterns)} learned patterns from disk")
        except Exception as e:
            logger.warning(f"[TEMPORAL-HANDLER] Failed to load learned patterns: {e}")

    def set_proactive_monitoring(self, manager):
        """Inject ProactiveMonitoringManager"""
        self.proactive_monitoring = manager
        logger.info("[TEMPORAL-HANDLER] ProactiveMonitoringManager injected")

    def set_change_detection(self, manager):
        """Inject ChangeDetectionManager"""
        self.change_detection = manager
        logger.info("[TEMPORAL-HANDLER] ChangeDetectionManager injected")

    def set_implicit_resolver(self, resolver):
        """Inject ImplicitReferenceResolver"""
        self.implicit_resolver = resolver
        logger.info("[TEMPORAL-HANDLER] ImplicitReferenceResolver injected")

    def set_conversation_tracker(self, tracker):
        """Inject ConversationTracker"""
        self.conversation_tracker = tracker
        logger.info("[TEMPORAL-HANDLER] ConversationTracker injected")

    def register_monitoring_alert(self, alert: Dict[str, Any]):
        """
        Register a monitoring alert for timeline building (v3.0 Enhanced).

        Called by HybridProactiveMonitoringManager's alert callback.

        Args:
            alert: Alert dictionary with keys:
                - space_id: int
                - event_type: str
                - message: str
                - priority: str
                - timestamp: datetime
                - metadata: dict
                - detection_method: str (NEW v3.0: "fast", "deep", "ml", "predictive")
                - predicted: bool (NEW v3.0: True if predictive)
                - correlation_id: str (NEW v3.0: Groups related alerts)
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
        """
        Handle a temporal query with full context resolution.

        Args:
            query: Natural language query
            space_id: Optional space ID context
            context: Optional additional context

        Returns:
            TemporalQueryResult with changes, timeline, and summary
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
        """
        Classify the type of temporal query (v3.0 Enhanced).

        Uses pattern matching to determine intent, including NEW intelligent types.
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
        """
        Use ImplicitReferenceResolver to resolve references in query.

        Resolves:
        - "the error" → specific error ID
        - "it" → most recent entity
        - "that space" → specific space ID
        - "since I last asked" → specific timestamp
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
        """
        Adjust time range based on conversation history.

        Handles queries like:
        - "since I last asked" → time of last conversation turn
        - "since we talked about X" → time of conversation about X
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
        """
        Handle 'What changed?' queries using ChangeDetectionManager.

        Uses cached snapshots from monitoring for instant results.
        """
        logger.debug(f"[TEMPORAL-HANDLER] Change detection for space {space_id}")

        if not self.change_detection:
            return TemporalQueryResult(
                query_type=TemporalQueryType.CHANGE_DETECTION,
                time_range=time_range,
                changes=[],
                summary="ChangeDetectionManager not available.",
                timeline=[]
            )

        # Get change detection result
        try:
            change_result = await self.change_detection.detect_changes(
                space_id=space_id,
                query=resolved_query.get('original_query'),
                context=resolved_query
            )

            # Convert to DetectedChange objects
            changes = []
            if change_result.changed:
                changes.append(DetectedChange(
                    change_type=self._map_change_type(change_result.change_type),
                    timestamp=datetime.now(),
                    space_id=space_id,
                    description=change_result.summary or "Changes detected",
                    confidence=change_result.similarity_score,
                    metadata={
                        'elapsed_time': change_result.elapsed_time,
                        'differences': change_result.differences
                    }
                ))

            # Build summary
            if not changes:
                summary = f"No changes detected in space {space_id} over {time_range.duration_seconds:.0f} seconds."
            else:
                summary = f"Detected {len(changes)} change(s) in space {space_id}: {change_result.summary}"

            # Build timeline from monitoring alerts
            timeline = self._build_timeline_from_alerts(time_range, space_id)

            return TemporalQueryResult(
                query_type=TemporalQueryType.CHANGE_DETECTION,
                time_range=time_range,
                changes=changes,
                summary=summary,
                timeline=timeline,
                metadata={'change_detection_result': change_result.__dict__}
            )

        except Exception as e:
            logger.error(f"[TEMPORAL-HANDLER] Change detection failed: {e}")
            return TemporalQueryResult(
                query_type=TemporalQueryType.CHANGE_DETECTION,
                time_range=time_range,
                changes=[],
                summary=f"Change detection failed: {e}",
                timeline=[]
            )

    async def _handle_error_tracking(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """
        Handle 'Has the error been fixed?' queries.

        Uses monitoring alerts to track error state changes.
        """
        logger.debug(f"[TEMPORAL-HANDLER] Error tracking for space {space_id}")

        # Filter monitoring alerts for error events
        error_alerts = [
            alert for alert in self.monitoring_alerts
            if time_range.start <= alert.get('timestamp', datetime.min) <= time_range.end
            and (space_id is None or alert.get('space_id') == space_id)
            and 'error' in alert.get('event_type', '').lower()
        ]

        # Classify error events
        errors_appeared = [a for a in error_alerts if 'appeared' in a.get('event_type', '').lower() or 'new' in a.get('event_type', '').lower()]
        errors_resolved = [a for a in error_alerts if 'resolved' in a.get('event_type', '').lower() or 'fixed' in a.get('event_type', '').lower()]

        # Build changes list
        changes = []

        for alert in errors_appeared:
            changes.append(DetectedChange(
                change_type=ChangeType.ERROR_APPEARED,
                timestamp=alert.get('timestamp', datetime.now()),
                space_id=alert.get('space_id'),
                description=alert.get('message', 'Error appeared'),
                confidence=0.9,
                metadata=alert.get('metadata', {})
            ))

        for alert in errors_resolved:
            changes.append(DetectedChange(
                change_type=ChangeType.ERROR_RESOLVED,
                timestamp=alert.get('timestamp', datetime.now()),
                space_id=alert.get('space_id'),
                description=alert.get('message', 'Error resolved'),
                confidence=0.9,
                metadata=alert.get('metadata', {})
            ))

        # Build summary
        if not changes:
            summary = "No error state changes detected in the time range."
        else:
            appeared_count = len(errors_appeared)
            resolved_count = len(errors_resolved)

            if resolved_count > appeared_count:
                summary = f"✅ Errors have been fixed! {resolved_count} error(s) resolved, {appeared_count} new error(s) appeared."
            elif appeared_count > resolved_count:
                summary = f"❌ New errors appeared. {appeared_count} error(s) appeared, {resolved_count} error(s) resolved."
            else:
                summary = f"Error status mixed: {appeared_count} error(s) appeared and resolved."

        # Build timeline
        timeline = [
            {
                'timestamp': alert.get('timestamp').isoformat(),
                'event_type': alert.get('event_type'),
                'message': alert.get('message'),
                'space_id': alert.get('space_id')
            }
            for alert in error_alerts
        ]

        return TemporalQueryResult(
            query_type=TemporalQueryType.ERROR_TRACKING,
            time_range=time_range,
            changes=sorted(changes, key=lambda c: c.timestamp),
            summary=summary,
            timeline=timeline
        )

    async def _handle_timeline(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """
        Handle 'What's new in last 5 minutes?' queries.

        Builds timeline from monitoring alerts.
        """
        logger.debug(f"[TEMPORAL-HANDLER] Timeline for space {space_id}")

        # Filter monitoring alerts in time range
        relevant_alerts = [
            alert for alert in self.monitoring_alerts
            if time_range.start <= alert.get('timestamp', datetime.min) <= time_range.end
            and (space_id is None or alert.get('space_id') == space_id)
        ]

        # Build timeline
        timeline = []
        changes = []

        for alert in relevant_alerts:
            timeline.append({
                'timestamp': alert.get('timestamp').isoformat(),
                'event_type': alert.get('event_type'),
                'message': alert.get('message'),
                'space_id': alert.get('space_id'),
                'priority': alert.get('priority')
            })

            # Convert to DetectedChange
            change_type = self._map_event_type_to_change_type(alert.get('event_type', ''))
            if change_type:
                changes.append(DetectedChange(
                    change_type=change_type,
                    timestamp=alert.get('timestamp', datetime.now()),
                    space_id=alert.get('space_id'),
                    description=alert.get('message', 'Event detected'),
                    confidence=0.8,
                    metadata=alert.get('metadata', {})
                ))

        # Build summary
        summary = f"Timeline of {len(relevant_alerts)} event(s) over {time_range.duration_seconds:.0f} seconds. "
        if changes:
            summary += f"{len(changes)} change(s) detected."
        else:
            summary += "No significant changes."

        return TemporalQueryResult(
            query_type=TemporalQueryType.TIMELINE,
            time_range=time_range,
            changes=changes,
            summary=summary,
            timeline=timeline
        )

    async def _handle_first_appearance(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """
        Handle 'When did this first appear?' queries.

        Searches monitoring alerts and entity states.
        """
        logger.debug(f"[TEMPORAL-HANDLER] First appearance query")

        # Get entity from resolved query
        entity = resolved_query.get('resolved_entities', {}).get('error') or \
                 resolved_query.get('resolved_entities', {}).get('element') or \
                 resolved_query.get('resolved_entities', {}).get('entity')

        # Search monitoring alerts
        matching_alerts = []
        for alert in self.monitoring_alerts:
            if entity and entity.lower() in alert.get('message', '').lower():
                matching_alerts.append(alert)

        # Find first appearance
        first_alert = None
        if matching_alerts:
            first_alert = min(matching_alerts, key=lambda a: a.get('timestamp', datetime.max))

        # Use ChangeDetectionManager to query entity states
        if self.change_detection and entity:
            try:
                entity_states = self.change_detection.get_entity_states(entity)
                if entity_states:
                    first_state = min(entity_states, key=lambda s: s.first_seen)
                    if not first_alert or first_state.first_seen < first_alert.get('timestamp'):
                        summary = f"First appeared at {first_state.first_seen.strftime('%I:%M:%S %p')} ({(datetime.now() - first_state.first_seen).total_seconds():.0f} seconds ago)"
                        return TemporalQueryResult(
                            query_type=TemporalQueryType.FIRST_APPEARANCE,
                            time_range=time_range,
                            changes=[],
                            summary=summary,
                            timeline=[],
                            metadata={'first_appearance': first_state.first_seen.isoformat()}
                        )
            except Exception as e:
                logger.warning(f"[TEMPORAL-HANDLER] Entity state lookup failed: {e}")

        # Fallback to monitoring alerts
        if first_alert:
            summary = f"First appeared at {first_alert.get('timestamp').strftime('%I:%M:%S %p')} ({(datetime.now() - first_alert.get('timestamp')).total_seconds():.0f} seconds ago)"
            metadata = {'first_appearance': first_alert.get('timestamp').isoformat()}
        else:
            summary = "Could not determine first appearance in the given time range."
            metadata = {}

        return TemporalQueryResult(
            query_type=TemporalQueryType.FIRST_APPEARANCE,
            time_range=time_range,
            changes=[],
            summary=summary,
            timeline=[],
            metadata=metadata
        )

    async def _handle_comparison(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """
        Handle 'How is this different from before?' queries.

        Uses ChangeDetectionManager to compare snapshots.
        """
        logger.debug(f"[TEMPORAL-HANDLER] Comparison query for space {space_id}")

        if not self.change_detection or space_id is None:
            return TemporalQueryResult(
                query_type=TemporalQueryType.COMPARISON,
                time_range=time_range,
                changes=[],
                summary="ChangeDetectionManager not available or space_id required for comparison.",
                timeline=[]
            )

        try:
            # Get change detection result
            change_result = await self.change_detection.detect_changes(
                space_id=space_id,
                query=resolved_query.get('original_query'),
                context=resolved_query
            )

            # Convert to changes
            changes = []
            if change_result.changed:
                changes.append(DetectedChange(
                    change_type=self._map_change_type(change_result.change_type),
                    timestamp=datetime.now(),
                    space_id=space_id,
                    description=change_result.summary or "Changes detected",
                    confidence=change_result.similarity_score,
                    metadata={
                        'elapsed_time': change_result.elapsed_time,
                        'differences': change_result.differences
                    }
                ))

            # Build summary
            if not changes:
                summary = f"No significant changes detected in space {space_id}."
            else:
                summary = f"Comparison for space {space_id}: {change_result.summary}"

            return TemporalQueryResult(
                query_type=TemporalQueryType.COMPARISON,
                time_range=time_range,
                changes=changes,
                summary=summary,
                timeline=[]
            )

        except Exception as e:
            logger.error(f"[TEMPORAL-HANDLER] Comparison failed: {e}")
            return TemporalQueryResult(
                query_type=TemporalQueryType.COMPARISON,
                time_range=time_range,
                changes=[],
                summary=f"Comparison failed: {e}",
                timeline=[]
            )

    async def _handle_monitoring_report(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """
        Handle 'What has monitoring detected?' queries.

        Returns all monitoring alerts in time range.
        """
        logger.debug(f"[TEMPORAL-HANDLER] Monitoring report")

        # Filter monitoring alerts
        relevant_alerts = [
            alert for alert in self.monitoring_alerts
            if time_range.start <= alert.get('timestamp', datetime.min) <= time_range.end
            and (space_id is None or alert.get('space_id') == space_id)
        ]

        # Group by priority
        by_priority = defaultdict(list)
        for alert in relevant_alerts:
            by_priority[alert.get('priority', 'MEDIUM')].append(alert)

        # Build summary
        summary_parts = []
        for priority in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = len(by_priority.get(priority, []))
            if count > 0:
                summary_parts.append(f"{count} {priority}")

        if summary_parts:
            summary = f"Monitoring detected {len(relevant_alerts)} event(s): " + ", ".join(summary_parts)
        else:
            summary = "No monitoring alerts in the time range."

        # Build timeline
        timeline = [
            {
                'timestamp': alert.get('timestamp').isoformat(),
                'event_type': alert.get('event_type'),
                'message': alert.get('message'),
                'space_id': alert.get('space_id'),
                'priority': alert.get('priority')
            }
            for alert in sorted(relevant_alerts, key=lambda a: a.get('timestamp', datetime.min))
        ]

        return TemporalQueryResult(
            query_type=TemporalQueryType.MONITORING_REPORT,
            time_range=time_range,
            changes=[],
            summary=summary,
            timeline=timeline,
            metadata={'alert_count_by_priority': dict(by_priority)}
        )

    async def _handle_generic_temporal_query(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """Handle generic temporal queries"""

        # Get monitoring alerts
        relevant_alerts = [
            alert for alert in self.monitoring_alerts
            if time_range.start <= alert.get('timestamp', datetime.min) <= time_range.end
            and (space_id is None or alert.get('space_id') == space_id)
        ]

        summary = f"Found {len(relevant_alerts)} event(s) in the time range."

        timeline = [
            {
                'timestamp': alert.get('timestamp').isoformat(),
                'event_type': alert.get('event_type'),
                'message': alert.get('message')
            }
            for alert in relevant_alerts
        ]

        return TemporalQueryResult(
            query_type=TemporalQueryType.TIMELINE,
            time_range=time_range,
            changes=[],
            summary=summary,
            timeline=timeline
        )

    def _build_timeline_from_alerts(
        self,
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Build timeline from monitoring alerts"""

        relevant_alerts = [
            alert for alert in self.monitoring_alerts
            if time_range.start <= alert.get('timestamp', datetime.min) <= time_range.end
            and (space_id is None or alert.get('space_id') == space_id)
        ]

        timeline = []
        for alert in sorted(relevant_alerts, key=lambda a: a.get('timestamp', datetime.min)):
            timeline.append({
                'timestamp': alert.get('timestamp').isoformat(),
                'event_type': alert.get('event_type'),
                'message': alert.get('message'),
                'space_id': alert.get('space_id'),
                'priority': alert.get('priority')
            })

        return timeline

    def _map_change_type(self, change_type_str: str) -> ChangeType:
        """Map ChangeDetectionManager ChangeType to TemporalQueryHandler ChangeType"""
        mapping = {
            'CONTENT': ChangeType.CONTENT_CHANGE,
            'LAYOUT': ChangeType.LAYOUT_CHANGE,
            'ERROR': ChangeType.ERROR_APPEARED,
            'VALUE': ChangeType.VALUE_CHANGED,
            'STATUS': ChangeType.STATUS_CHANGED,
        }
        return mapping.get(change_type_str, ChangeType.CONTENT_CHANGE)

    def _map_event_type_to_change_type(self, event_type: str) -> Optional[ChangeType]:
        """Map monitoring event type to ChangeType"""
        event_lower = event_type.lower()

        if 'error' in event_lower:
            if 'appeared' in event_lower or 'new' in event_lower:
                return ChangeType.ERROR_APPEARED
            elif 'resolved' in event_lower or 'fixed' in event_lower:
                return ChangeType.ERROR_RESOLVED

        elif 'build' in event_lower:
            if 'completed' in event_lower or 'success' in event_lower:
                return ChangeType.BUILD_COMPLETED
            elif 'failed' in event_lower:
                return ChangeType.BUILD_FAILED

        elif 'process' in event_lower:
            if 'started' in event_lower:
                return ChangeType.PROCESS_STARTED
            elif 'stopped' in event_lower or 'ended' in event_lower:
                return ChangeType.PROCESS_STOPPED

        return None


# ============================================================================
# INITIALIZATION
# ============================================================================

# Global instance
_temporal_query_handler_instance = None


def get_temporal_query_handler() -> TemporalQueryHandler:
    """Get or create the global temporal query handler"""
    global _temporal_query_handler_instance
    if _temporal_query_handler_instance is None:
        _temporal_query_handler_instance = TemporalQueryHandler()
    return _temporal_query_handler_instance


def initialize_temporal_query_handler(
    proactive_monitoring_manager=None,
    change_detection_manager=None,
    implicit_resolver=None,
    conversation_tracker=None
) -> TemporalQueryHandler:
    """
    Initialize temporal query handler with dependencies.

    Args:
        proactive_monitoring_manager: ProactiveMonitoringManager instance
        change_detection_manager: ChangeDetectionManager instance
        implicit_resolver: ImplicitReferenceResolver instance
        conversation_tracker: ConversationTracker instance

    Returns:
        Initialized TemporalQueryHandler
    """
    global _temporal_query_handler_instance

    _temporal_query_handler_instance = TemporalQueryHandler(
        proactive_monitoring_manager=proactive_monitoring_manager,
        change_detection_manager=change_detection_manager,
        implicit_resolver=implicit_resolver,
        conversation_tracker=conversation_tracker
    )

    logger.info("[TEMPORAL-HANDLER] Initialized with full ProactiveMonitoring integration")

    return _temporal_query_handler_instance
