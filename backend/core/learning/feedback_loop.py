"""
User Feedback Learning Loop
Learns from user interactions to improve notification relevance and timing.

Philosophy:
- Learn from dismissals (user doesn't care about this pattern)
- Learn from engagement (user values this pattern)
- Adapt importance thresholds dynamically
- Respect user's time and attention

Design Principles:
- Privacy-first: All learning happens locally
- Transparent: User can see what JARVIS learned
- Reversible: User can reset learned patterns
- Graceful: Degrades to defaults if no data
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from collections import defaultdict, deque
import json
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class UserResponse(str, Enum):
    """User response types to notifications."""
    ENGAGED = "engaged"  # User clicked, asked for details, took action
    DISMISSED = "dismissed"  # User ignored or explicitly dismissed
    DEFERRED = "deferred"  # User said "not now" but interested
    NEGATIVE_FEEDBACK = "negative_feedback"  # User said "stop showing this"


class NotificationPattern(str, Enum):
    """Notification pattern categories for learning."""
    TERMINAL_ERROR = "terminal_error"
    TERMINAL_COMPLETION = "terminal_completion"
    TERMINAL_WARNING = "terminal_warning"
    BROWSER_UPDATE = "browser_update"
    CODE_DIAGNOSTIC = "code_diagnostic"
    WORKFLOW_SUGGESTION = "workflow_suggestion"
    RESOURCE_WARNING = "resource_warning"
    SECURITY_ALERT = "security_alert"
    OTHER = "other"


@dataclass
class FeedbackEvent:
    """Single user feedback event."""
    pattern: NotificationPattern
    response: UserResponse
    timestamp: datetime
    notification_text: str
    context: Dict[str, Any] = field(default_factory=dict)
    time_to_respond: float = 0.0  # Seconds from notification to response

    @property
    def pattern_hash(self) -> str:
        """Generate hash for pattern matching."""
        # Hash based on pattern type + key context attributes
        context_str = f"{self.pattern.value}:{self.context.get('window_type', '')}:{self.context.get('error_type', '')}"
        return hashlib.md5(context_str.encode()).hexdigest()[:12]


@dataclass
class PatternStats:
    """Statistics for a specific notification pattern."""
    pattern: NotificationPattern
    total_shown: int = 0
    engaged_count: int = 0
    dismissed_count: int = 0
    deferred_count: int = 0
    negative_feedback_count: int = 0
    avg_time_to_respond: float = 0.0
    last_shown: Optional[datetime] = None

    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate (0.0-1.0)."""
        if self.total_shown == 0:
            return 0.5  # Default to neutral
        return self.engaged_count / self.total_shown

    @property
    def dismissal_rate(self) -> float:
        """Calculate dismissal rate (0.0-1.0)."""
        if self.total_shown == 0:
            return 0.0
        return self.dismissed_count / self.total_shown

    @property
    def is_valued(self) -> bool:
        """Determine if user values this pattern."""
        # Valued if engagement rate > 40% and shown at least 3 times
        return self.total_shown >= 3 and self.engagement_rate >= 0.4

    @property
    def is_ignored(self) -> bool:
        """Determine if user consistently ignores this pattern."""
        # Ignored if dismissal rate > 70% and shown at least 5 times
        # OR if negative feedback received twice
        return (
            (self.total_shown >= 5 and self.dismissal_rate >= 0.7)
            or (self.negative_feedback_count >= 2)
        )

    @property
    def should_suppress(self) -> bool:
        """Should this pattern be suppressed?"""
        return self.is_ignored or self.negative_feedback_count > 0

    @property
    def importance_multiplier(self) -> float:
        """
        Calculate importance multiplier based on user feedback.

        Returns:
            0.0-2.0 multiplier to apply to base importance
            < 1.0 = reduce importance
            > 1.0 = increase importance
        """
        if self.should_suppress:
            return 0.0  # Completely suppress

        if self.total_shown < 3:
            return 1.0  # Not enough data, use default

        # Base multiplier on engagement rate
        if self.engagement_rate >= 0.8:
            return 1.5  # Highly valued, boost
        elif self.engagement_rate >= 0.5:
            return 1.2  # Valued, slight boost
        elif self.dismissal_rate >= 0.6:
            return 0.7  # Often dismissed, reduce
        else:
            return 1.0  # Neutral


class FeedbackLearningLoop:
    """
    Learns from user feedback to improve notification intelligence.

    Features:
    - Tracks engagement/dismissal patterns
    - Adapts importance thresholds per pattern type
    - Identifies valued vs ignored patterns
    - Learns optimal timing for notifications
    - Exports learned data for transparency
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_history_size: int = 1000,
    ):
        """
        Initialize feedback learning loop.

        Args:
            storage_path: Path to persist learned data (None = memory only)
            max_history_size: Maximum feedback events to retain
        """
        self.storage_path = storage_path or Path.home() / ".jarvis" / "learning" / "feedback.json"
        self.max_history_size = max_history_size

        # Feedback history
        self.feedback_history: deque[FeedbackEvent] = deque(maxlen=max_history_size)

        # Pattern statistics
        self.pattern_stats: Dict[str, PatternStats] = {}  # pattern_hash -> stats

        # Timing insights
        self.timing_stats: Dict[int, int] = defaultdict(int)  # hour_of_day -> engagement_count
        self.best_hours: Set[int] = set()  # Hours with highest engagement
        self.worst_hours: Set[int] = set()  # Hours with highest dismissal

        # Explicit user preferences
        self.suppressed_patterns: Set[str] = set()  # User explicitly disabled
        self.boosted_patterns: Set[str] = set()  # User explicitly enabled

        # Load existing data
        self._load_from_disk()

        logger.info(
            f"[FEEDBACK-LOOP] Initialized with {len(self.feedback_history)} historical events, "
            f"{len(self.pattern_stats)} pattern stats"
        )

    async def record_feedback(
        self,
        pattern: NotificationPattern,
        response: UserResponse,
        notification_text: str,
        context: Optional[Dict[str, Any]] = None,
        time_to_respond: float = 0.0,
    ) -> None:
        """
        Record user feedback for a notification.

        Args:
            pattern: Notification pattern type
            response: How user responded
            notification_text: The notification message shown
            context: Additional context about the notification
            time_to_respond: Seconds from notification to response
        """
        event = FeedbackEvent(
            pattern=pattern,
            response=response,
            timestamp=datetime.now(),
            notification_text=notification_text,
            context=context or {},
            time_to_respond=time_to_respond,
        )

        # Add to history
        self.feedback_history.append(event)

        # Update pattern stats
        pattern_hash = event.pattern_hash
        if pattern_hash not in self.pattern_stats:
            self.pattern_stats[pattern_hash] = PatternStats(pattern=pattern)

        stats = self.pattern_stats[pattern_hash]
        stats.total_shown += 1
        stats.last_shown = event.timestamp

        # Update response counts
        if response == UserResponse.ENGAGED:
            stats.engaged_count += 1
        elif response == UserResponse.DISMISSED:
            stats.dismissed_count += 1
        elif response == UserResponse.DEFERRED:
            stats.deferred_count += 1
        elif response == UserResponse.NEGATIVE_FEEDBACK:
            stats.negative_feedback_count += 1
            # Auto-suppress after negative feedback
            self.suppressed_patterns.add(pattern_hash)

        # Update timing stats
        hour = event.timestamp.hour
        if response == UserResponse.ENGAGED:
            self.timing_stats[hour] += 1

        # Update average response time
        if stats.total_shown > 1:
            stats.avg_time_to_respond = (
                stats.avg_time_to_respond * (stats.total_shown - 1) + time_to_respond
            ) / stats.total_shown
        else:
            stats.avg_time_to_respond = time_to_respond

        logger.info(
            f"[FEEDBACK-LOOP] Recorded {response.value} for {pattern.value} "
            f"(engagement_rate={stats.engagement_rate:.2%}, "
            f"dismissal_rate={stats.dismissal_rate:.2%})"
        )

        # Persist to disk
        await self._save_to_disk()

    def should_show_notification(
        self,
        pattern: NotificationPattern,
        base_importance: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, float]:
        """
        Determine if notification should be shown based on learned patterns.

        Args:
            pattern: Notification pattern type
            base_importance: Base importance score (0.0-1.0)
            context: Additional context for pattern matching

        Returns:
            (should_show, adjusted_importance)
        """
        # Generate pattern hash
        event = FeedbackEvent(
            pattern=pattern,
            response=UserResponse.ENGAGED,  # Dummy for hash generation
            timestamp=datetime.now(),
            notification_text="",
            context=context or {},
        )
        pattern_hash = event.pattern_hash

        # Check explicit suppressions
        if pattern_hash in self.suppressed_patterns:
            logger.debug(f"[FEEDBACK-LOOP] Suppressing {pattern.value} (explicitly disabled)")
            return False, 0.0

        # Get pattern stats
        stats = self.pattern_stats.get(pattern_hash)
        if not stats:
            # No history, allow with base importance
            logger.debug(f"[FEEDBACK-LOOP] No history for {pattern.value}, using base importance")
            return True, base_importance

        # Apply learned multiplier
        multiplier = stats.importance_multiplier
        adjusted_importance = base_importance * multiplier

        # Check if pattern should be suppressed
        if stats.should_suppress:
            logger.debug(
                f"[FEEDBACK-LOOP] Suppressing {pattern.value} "
                f"(dismissal_rate={stats.dismissal_rate:.2%}, "
                f"negative_feedback={stats.negative_feedback_count})"
            )
            return False, 0.0

        # Check if importance is still above threshold
        should_show = adjusted_importance >= 0.4  # Minimum threshold

        logger.debug(
            f"[FEEDBACK-LOOP] {pattern.value}: "
            f"base={base_importance:.2f}, "
            f"multiplier={multiplier:.2f}, "
            f"adjusted={adjusted_importance:.2f}, "
            f"show={should_show}"
        )

        return should_show, adjusted_importance

    def is_good_time_to_notify(self, hour: Optional[int] = None) -> bool:
        """
        Determine if current time is good for notifications based on learned patterns.

        Args:
            hour: Hour to check (0-23), defaults to current hour

        Returns:
            True if this is a good time to notify
        """
        if hour is None:
            hour = datetime.now().hour

        # Need at least 50 feedback events to learn timing patterns
        if len(self.feedback_history) < 50:
            return True  # Not enough data, allow notifications

        # Update best/worst hours if needed
        if not self.best_hours:
            self._calculate_best_hours()

        # If this hour is in worst hours, be cautious
        if hour in self.worst_hours:
            logger.debug(f"[FEEDBACK-LOOP] Hour {hour} is historically low-engagement")
            return False

        return True

    def _calculate_best_hours(self) -> None:
        """Calculate best and worst hours for notifications."""
        if not self.timing_stats:
            return

        # Find hours with highest engagement
        sorted_hours = sorted(
            self.timing_stats.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Top 30% are best hours
        num_best = max(1, len(sorted_hours) // 3)
        self.best_hours = {hour for hour, _ in sorted_hours[:num_best]}

        # Bottom 30% are worst hours
        num_worst = max(1, len(sorted_hours) // 3)
        self.worst_hours = {hour for hour, _ in sorted_hours[-num_worst:]}

        logger.info(
            f"[FEEDBACK-LOOP] Best notification hours: {sorted(self.best_hours)}, "
            f"Worst hours: {sorted(self.worst_hours)}"
        )

    def get_pattern_insights(self, pattern: NotificationPattern) -> Dict[str, Any]:
        """
        Get insights about a specific notification pattern.

        Args:
            pattern: Pattern to analyze

        Returns:
            Dictionary with engagement stats, timing, and recommendations
        """
        # Find all stats for this pattern type
        pattern_stats_list = [
            stats for stats in self.pattern_stats.values()
            if stats.pattern == pattern
        ]

        if not pattern_stats_list:
            return {
                "pattern": pattern.value,
                "has_data": False,
                "recommendation": "No data yet - using defaults",
            }

        # Aggregate stats
        total_shown = sum(s.total_shown for s in pattern_stats_list)
        total_engaged = sum(s.engaged_count for s in pattern_stats_list)
        total_dismissed = sum(s.dismissed_count for s in pattern_stats_list)

        engagement_rate = total_engaged / total_shown if total_shown > 0 else 0.0
        dismissal_rate = total_dismissed / total_shown if total_shown > 0 else 0.0

        # Determine recommendation
        if engagement_rate >= 0.6:
            recommendation = "Highly valued - boost importance"
        elif dismissal_rate >= 0.7:
            recommendation = "Often dismissed - reduce frequency"
        else:
            recommendation = "Neutral - continue as configured"

        return {
            "pattern": pattern.value,
            "has_data": True,
            "total_shown": total_shown,
            "engagement_rate": round(engagement_rate, 3),
            "dismissal_rate": round(dismissal_rate, 3),
            "recommendation": recommendation,
            "is_valued": any(s.is_valued for s in pattern_stats_list),
            "is_ignored": any(s.is_ignored for s in pattern_stats_list),
        }

    def export_learned_data(self) -> Dict[str, Any]:
        """
        Export all learned data for transparency/debugging.

        Returns:
            Dictionary with all learned patterns, timing, and stats
        """
        return {
            "total_feedback_events": len(self.feedback_history),
            "pattern_stats": {
                pattern_hash: {
                    "pattern": stats.pattern.value,
                    "total_shown": stats.total_shown,
                    "engagement_rate": round(stats.engagement_rate, 3),
                    "dismissal_rate": round(stats.dismissal_rate, 3),
                    "is_valued": stats.is_valued,
                    "is_ignored": stats.is_ignored,
                    "importance_multiplier": round(stats.importance_multiplier, 2),
                }
                for pattern_hash, stats in self.pattern_stats.items()
            },
            "timing_insights": {
                "best_hours": sorted(self.best_hours) if self.best_hours else None,
                "worst_hours": sorted(self.worst_hours) if self.worst_hours else None,
                "engagement_by_hour": dict(self.timing_stats),
            },
            "suppressed_patterns": list(self.suppressed_patterns),
            "boosted_patterns": list(self.boosted_patterns),
        }

    async def reset_learning(self, pattern: Optional[NotificationPattern] = None) -> None:
        """
        Reset learned data (for user control).

        Args:
            pattern: If specified, reset only this pattern. Otherwise reset all.
        """
        if pattern is None:
            # Reset everything
            self.feedback_history.clear()
            self.pattern_stats.clear()
            self.timing_stats.clear()
            self.best_hours.clear()
            self.worst_hours.clear()
            self.suppressed_patterns.clear()
            self.boosted_patterns.clear()
            logger.info("[FEEDBACK-LOOP] Reset all learned data")
        else:
            # Reset specific pattern
            pattern_hashes = [
                h for h, s in self.pattern_stats.items()
                if s.pattern == pattern
            ]
            for h in pattern_hashes:
                del self.pattern_stats[h]
                self.suppressed_patterns.discard(h)
                self.boosted_patterns.discard(h)
            logger.info(f"[FEEDBACK-LOOP] Reset learned data for {pattern.value}")

        await self._save_to_disk()

    async def _save_to_disk(self) -> None:
        """Persist learned data to disk."""
        if not self.storage_path:
            return

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "feedback_history": [
                    {
                        "pattern": e.pattern.value,
                        "response": e.response.value,
                        "timestamp": e.timestamp.isoformat(),
                        "notification_text": e.notification_text,
                        "context": e.context,
                        "time_to_respond": e.time_to_respond,
                    }
                    for e in list(self.feedback_history)[-100:]  # Last 100 events
                ],
                "pattern_stats": {
                    h: {
                        "pattern": s.pattern.value,
                        "total_shown": s.total_shown,
                        "engaged_count": s.engaged_count,
                        "dismissed_count": s.dismissed_count,
                        "deferred_count": s.deferred_count,
                        "negative_feedback_count": s.negative_feedback_count,
                        "avg_time_to_respond": s.avg_time_to_respond,
                        "last_shown": s.last_shown.isoformat() if s.last_shown else None,
                    }
                    for h, s in self.pattern_stats.items()
                },
                "timing_stats": dict(self.timing_stats),
                "suppressed_patterns": list(self.suppressed_patterns),
                "boosted_patterns": list(self.boosted_patterns),
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"[FEEDBACK-LOOP] Saved to {self.storage_path}")

        except Exception as e:
            logger.error(f"[FEEDBACK-LOOP] Failed to save: {e}", exc_info=True)

    def _load_from_disk(self) -> None:
        """Load persisted learned data."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Load feedback history
            for event_data in data.get("feedback_history", []):
                event = FeedbackEvent(
                    pattern=NotificationPattern(event_data["pattern"]),
                    response=UserResponse(event_data["response"]),
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    notification_text=event_data["notification_text"],
                    context=event_data.get("context", {}),
                    time_to_respond=event_data.get("time_to_respond", 0.0),
                )
                self.feedback_history.append(event)

            # Load pattern stats
            for pattern_hash, stats_data in data.get("pattern_stats", {}).items():
                stats = PatternStats(
                    pattern=NotificationPattern(stats_data["pattern"]),
                    total_shown=stats_data["total_shown"],
                    engaged_count=stats_data["engaged_count"],
                    dismissed_count=stats_data["dismissed_count"],
                    deferred_count=stats_data.get("deferred_count", 0),
                    negative_feedback_count=stats_data.get("negative_feedback_count", 0),
                    avg_time_to_respond=stats_data.get("avg_time_to_respond", 0.0),
                    last_shown=datetime.fromisoformat(stats_data["last_shown"]) if stats_data.get("last_shown") else None,
                )
                self.pattern_stats[pattern_hash] = stats

            # Load timing stats
            self.timing_stats = defaultdict(int, {
                int(k): v for k, v in data.get("timing_stats", {}).items()
            })

            # Load suppressions/boosts
            self.suppressed_patterns = set(data.get("suppressed_patterns", []))
            self.boosted_patterns = set(data.get("boosted_patterns", []))

            logger.info(
                f"[FEEDBACK-LOOP] Loaded from {self.storage_path}: "
                f"{len(self.feedback_history)} events, {len(self.pattern_stats)} patterns"
            )

        except Exception as e:
            logger.error(f"[FEEDBACK-LOOP] Failed to load: {e}", exc_info=True)


# Global instance
_global_feedback_loop: Optional[FeedbackLearningLoop] = None


def get_feedback_loop() -> FeedbackLearningLoop:
    """Get or create global feedback learning loop."""
    global _global_feedback_loop

    if _global_feedback_loop is None:
        _global_feedback_loop = FeedbackLearningLoop()

    return _global_feedback_loop
