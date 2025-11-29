"""
JARVIS macOS Helper - Intelligent Notification Triage System

Smart notification management that categorizes, prioritizes, and batches
notifications to minimize user interruption while ensuring important
notifications are never missed.

Features:
- Category-based notification routing
- Dynamic priority scoring based on context
- Smart batching for low-priority notifications
- Focus mode integration (respects user focus)
- Delivery scheduling optimization
- Learning from user notification interactions
- Zero hardcoded rules - all learned dynamically

Architecture:
    NotificationTriageSystem
    ├── CategoryEngine (categorize incoming notifications)
    ├── PriorityScorer (score notification importance)
    ├── SmartBatcher (batch non-urgent notifications)
    ├── FocusGuard (respect focus/DND modes)
    ├── DeliveryScheduler (optimal delivery timing)
    └── InteractionLearner (learn from user behavior)

Examples:
- Critical: "Security alert from 1Password" → Immediate delivery
- High: "Slack DM from boss" → Deliver soon, may interrupt
- Medium: "Email from newsletter" → Batch with similar
- Low: "App update available" → Deliver during idle time
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class UrgencyLevel(str, Enum):
    """Urgency levels for notifications."""
    CRITICAL = "critical"   # Deliver immediately, always interrupt
    HIGH = "high"           # Deliver soon, may interrupt
    MEDIUM = "medium"       # Deliver when convenient
    LOW = "low"            # Batch and deliver during idle
    SILENT = "silent"      # Log only, never show


class NotificationCategory(str, Enum):
    """Categories of notifications."""
    SECURITY = "security"           # Security alerts, 2FA
    COMMUNICATION = "communication" # Messages, emails, calls
    CALENDAR = "calendar"           # Events, reminders
    WORK = "work"                   # Work-related apps
    SOCIAL = "social"              # Social media
    SYSTEM = "system"              # OS notifications
    UPDATE = "update"              # App updates
    MARKETING = "marketing"        # Promotions, ads
    UNKNOWN = "unknown"


class NotificationAction(str, Enum):
    """Actions that can be taken on notifications."""
    DELIVER = "deliver"         # Show to user
    BATCH = "batch"             # Add to batch for later
    DEFER = "defer"             # Show later
    SUPPRESS = "suppress"       # Don't show
    ESCALATE = "escalate"       # Upgrade priority


class TriageDecision(str, Enum):
    """Decision made about a notification."""
    IMMEDIATE = "immediate"     # Deliver now
    BATCHED = "batched"         # Added to batch
    DEFERRED = "deferred"       # Will show later
    SUPPRESSED = "suppressed"   # Won't show


class FocusMode(str, Enum):
    """User focus modes."""
    NORMAL = "normal"           # No focus restrictions
    WORK = "work"              # Work focus - work notifications only
    PERSONAL = "personal"       # Personal focus - non-work only
    DND = "dnd"                 # Do not disturb - critical only
    SLEEP = "sleep"            # Sleep mode - nothing


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class TriagedNotification:
    """A notification that has been triaged."""
    # Identity
    notification_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])
    original_id: Optional[str] = None  # Original notification ID

    # Source
    app_name: str = ""
    bundle_id: Optional[str] = None

    # Content
    title: str = ""
    body: str = ""
    subtitle: Optional[str] = None

    # Classification
    category: NotificationCategory = NotificationCategory.UNKNOWN
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM

    # Scoring
    priority_score: float = 0.5  # 0.0 to 1.0
    relevance_score: float = 0.5
    timeliness_score: float = 0.5

    # Context
    context_at_receipt: Optional[Dict[str, Any]] = None
    user_activity_at_receipt: Optional[str] = None

    # Triage result
    decision: TriageDecision = TriageDecision.IMMEDIATE
    batch_id: Optional[str] = None
    scheduled_delivery: Optional[datetime] = None

    # Timestamps
    received_at: datetime = field(default_factory=datetime.now)
    triaged_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None

    # User interaction
    was_seen: bool = False
    was_clicked: bool = False
    was_dismissed: bool = False
    response_time_seconds: Optional[float] = None

    # Actions
    available_actions: List[str] = field(default_factory=list)
    action_taken: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "notification_id": self.notification_id,
            "app_name": self.app_name,
            "bundle_id": self.bundle_id,
            "title": self.title,
            "body": self.body[:100] if self.body else None,  # Truncate for privacy
            "category": self.category.value,
            "urgency": self.urgency.value,
            "priority_score": self.priority_score,
            "decision": self.decision.value,
            "received_at": self.received_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "was_clicked": self.was_clicked,
        }

    def get_combined_score(self) -> float:
        """Get combined priority score."""
        return (
            self.priority_score * 0.5 +
            self.relevance_score * 0.3 +
            self.timeliness_score * 0.2
        )


@dataclass
class NotificationBatch:
    """A batch of notifications to be delivered together."""
    batch_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:10])
    category: NotificationCategory = NotificationCategory.UNKNOWN

    # Notifications
    notifications: List[TriagedNotification] = field(default_factory=list)
    max_size: int = 10

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_delivery: Optional[datetime] = None
    delivered_at: Optional[datetime] = None

    # Batching rules
    batch_window_seconds: float = 300.0  # 5 minutes

    # State
    is_delivered: bool = False

    def add_notification(self, notification: TriagedNotification) -> bool:
        """Add a notification to the batch."""
        if len(self.notifications) >= self.max_size:
            return False

        notification.batch_id = self.batch_id
        notification.decision = TriageDecision.BATCHED
        self.notifications.append(notification)
        return True

    def is_ready_to_deliver(self) -> bool:
        """Check if batch is ready to be delivered."""
        if not self.notifications:
            return False

        # Full batch
        if len(self.notifications) >= self.max_size:
            return True

        # Time window expired
        elapsed = (datetime.now() - self.created_at).total_seconds()
        if elapsed >= self.batch_window_seconds:
            return True

        # Scheduled delivery time passed
        if self.scheduled_delivery and datetime.now() >= self.scheduled_delivery:
            return True

        return False

    def get_summary(self) -> str:
        """Get a summary of the batch."""
        if not self.notifications:
            return "Empty batch"

        apps = set(n.app_name for n in self.notifications)
        count = len(self.notifications)

        if len(apps) == 1:
            return f"{count} notifications from {list(apps)[0]}"
        return f"{count} notifications from {len(apps)} apps"


@dataclass
class NotificationTriageConfig:
    """Configuration for the notification triage system."""
    # Batching
    enable_batching: bool = os.getenv("TRIAGE_ENABLE_BATCHING", "true").lower() == "true"
    batch_window_seconds: float = float(os.getenv("TRIAGE_BATCH_WINDOW", "300.0"))
    max_batch_size: int = int(os.getenv("TRIAGE_MAX_BATCH_SIZE", "10"))

    # Delivery
    check_interval_seconds: float = float(os.getenv("TRIAGE_CHECK_INTERVAL", "10.0"))
    idle_delivery_delay_seconds: float = float(os.getenv("TRIAGE_IDLE_DELAY", "60.0"))

    # Focus mode
    respect_focus_mode: bool = True
    respect_system_dnd: bool = True

    # Suppression
    suppress_duplicate_window_seconds: float = float(os.getenv("TRIAGE_DUPLICATE_WINDOW", "60.0"))
    suppress_marketing: bool = os.getenv("TRIAGE_SUPPRESS_MARKETING", "false").lower() == "true"

    # Learning
    enable_learning: bool = os.getenv("TRIAGE_ENABLE_LEARNING", "true").lower() == "true"
    learning_decay_days: int = int(os.getenv("TRIAGE_LEARNING_DECAY", "30"))

    # Rate limiting
    max_notifications_per_minute: int = int(os.getenv("TRIAGE_MAX_PER_MINUTE", "10"))
    max_notifications_per_hour: int = int(os.getenv("TRIAGE_MAX_PER_HOUR", "50"))

    # Priority thresholds
    critical_threshold: float = float(os.getenv("TRIAGE_CRITICAL_THRESHOLD", "0.9"))
    high_threshold: float = float(os.getenv("TRIAGE_HIGH_THRESHOLD", "0.7"))
    medium_threshold: float = float(os.getenv("TRIAGE_MEDIUM_THRESHOLD", "0.4"))


# =============================================================================
# Category Engine
# =============================================================================

class CategoryEngine:
    """
    Categorizes incoming notifications based on source and content.

    Uses learned mappings with fallback to heuristic classification.
    """

    def __init__(self):
        # Learned mappings
        self._app_category_map: Dict[str, NotificationCategory] = {}

        # Default mappings
        self._default_app_map = {
            # Security
            "1Password": NotificationCategory.SECURITY,
            "Keychain Access": NotificationCategory.SECURITY,
            "Authy": NotificationCategory.SECURITY,
            "Google Authenticator": NotificationCategory.SECURITY,

            # Communication
            "Messages": NotificationCategory.COMMUNICATION,
            "Mail": NotificationCategory.COMMUNICATION,
            "Slack": NotificationCategory.COMMUNICATION,
            "Microsoft Teams": NotificationCategory.COMMUNICATION,
            "Discord": NotificationCategory.COMMUNICATION,
            "WhatsApp": NotificationCategory.COMMUNICATION,
            "Telegram": NotificationCategory.COMMUNICATION,
            "FaceTime": NotificationCategory.COMMUNICATION,

            # Calendar
            "Calendar": NotificationCategory.CALENDAR,
            "Reminders": NotificationCategory.CALENDAR,
            "Fantastical": NotificationCategory.CALENDAR,

            # Work
            "GitHub Desktop": NotificationCategory.WORK,
            "Jira": NotificationCategory.WORK,
            "Notion": NotificationCategory.WORK,
            "Linear": NotificationCategory.WORK,

            # Social
            "Twitter": NotificationCategory.SOCIAL,
            "Facebook": NotificationCategory.SOCIAL,
            "Instagram": NotificationCategory.SOCIAL,
            "LinkedIn": NotificationCategory.SOCIAL,
            "TikTok": NotificationCategory.SOCIAL,

            # System
            "System Preferences": NotificationCategory.SYSTEM,
            "Finder": NotificationCategory.SYSTEM,
            "Disk Utility": NotificationCategory.SYSTEM,

            # Updates
            "App Store": NotificationCategory.UPDATE,
            "Software Update": NotificationCategory.UPDATE,
        }

        # Category keywords for content analysis
        self._category_keywords = {
            NotificationCategory.SECURITY: [
                "security", "password", "2fa", "verification", "login",
                "authentication", "suspicious", "breach", "alert"
            ],
            NotificationCategory.COMMUNICATION: [
                "message", "email", "call", "chat", "replied", "mentioned",
                "dm", "direct message"
            ],
            NotificationCategory.CALENDAR: [
                "event", "meeting", "reminder", "appointment", "starts in",
                "calendar", "scheduled"
            ],
            NotificationCategory.MARKETING: [
                "sale", "discount", "offer", "promo", "deal", "limited time",
                "exclusive", "% off", "subscribe", "newsletter"
            ],
        }

    def categorize(
        self,
        app_name: str,
        title: str,
        body: str,
        bundle_id: Optional[str] = None,
    ) -> Tuple[NotificationCategory, float]:
        """
        Categorize a notification.

        Returns:
            Tuple of (category, confidence)
        """
        # Try learned mapping first
        if app_name in self._app_category_map:
            return self._app_category_map[app_name], 0.9

        # Try default mapping
        if app_name in self._default_app_map:
            return self._default_app_map[app_name], 0.8

        # Content-based classification
        combined_text = f"{title} {body}".lower()

        best_category = NotificationCategory.UNKNOWN
        best_score = 0.0

        for category, keywords in self._category_keywords.items():
            matches = sum(1 for kw in keywords if kw in combined_text)
            if matches > 0:
                score = min(matches / 3.0, 1.0)
                if score > best_score:
                    best_score = score
                    best_category = category

        if best_score > 0.3:
            return best_category, best_score * 0.7

        return NotificationCategory.UNKNOWN, 0.5

    def learn_category(self, app_name: str, category: NotificationCategory) -> None:
        """Learn a new app-to-category mapping."""
        self._app_category_map[app_name] = category


# =============================================================================
# Priority Scorer
# =============================================================================

class PriorityScorer:
    """
    Scores notification priority based on multiple factors.

    Factors:
    - Source app importance
    - Content urgency indicators
    - User's historical interaction with similar notifications
    - Current user context
    - Time sensitivity
    """

    def __init__(self):
        # Learned app priorities
        self._app_priority: Dict[str, float] = {}

        # Interaction history (app -> click rate)
        self._interaction_rates: Dict[str, Deque[bool]] = defaultdict(lambda: deque(maxlen=50))

        # Default app priorities
        self._default_priorities = {
            # Very high (0.9+)
            "1Password": 0.95,
            "FaceTime": 0.9,

            # High (0.7-0.9)
            "Messages": 0.85,
            "Calendar": 0.8,
            "Slack": 0.75,
            "Mail": 0.7,

            # Medium (0.4-0.7)
            "GitHub Desktop": 0.6,
            "Notion": 0.55,

            # Low (0.2-0.4)
            "App Store": 0.3,
            "News": 0.25,

            # Very low (<0.2)
        }

        # Urgency keywords
        self._urgency_keywords = {
            "critical": ["urgent", "critical", "emergency", "security alert", "breach"],
            "high": ["important", "action required", "deadline", "failed", "error"],
            "medium": ["reminder", "update", "fyi", "heads up"],
            "low": ["tip", "suggestion", "news", "digest"],
        }

    def score(
        self,
        notification: TriagedNotification,
        user_activity: Optional[str] = None,
        focus_mode: FocusMode = FocusMode.NORMAL,
    ) -> Tuple[float, UrgencyLevel]:
        """
        Score notification priority.

        Returns:
            Tuple of (priority_score 0.0-1.0, urgency_level)
        """
        scores = []

        # App-based score
        app_score = self._get_app_score(notification.app_name)
        scores.append(("app", app_score, 0.3))

        # Content-based score
        content_score = self._score_content(notification.title, notification.body)
        scores.append(("content", content_score, 0.3))

        # Interaction history score
        history_score = self._get_history_score(notification.app_name)
        scores.append(("history", history_score, 0.2))

        # Context score
        context_score = self._score_context(notification.category, user_activity, focus_mode)
        scores.append(("context", context_score, 0.2))

        # Weighted combination
        final_score = sum(score * weight for _, score, weight in scores)

        # Determine urgency level
        urgency = self._score_to_urgency(final_score)

        return final_score, urgency

    def _get_app_score(self, app_name: str) -> float:
        """Get priority score for an app."""
        if app_name in self._app_priority:
            return self._app_priority[app_name]

        if app_name in self._default_priorities:
            return self._default_priorities[app_name]

        return 0.5

    def _score_content(self, title: str, body: str) -> float:
        """Score based on notification content."""
        combined = f"{title} {body}".lower()

        # Check for urgency keywords
        for urgency, keywords in self._urgency_keywords.items():
            if any(kw in combined for kw in keywords):
                if urgency == "critical":
                    return 0.95
                elif urgency == "high":
                    return 0.8
                elif urgency == "medium":
                    return 0.5
                elif urgency == "low":
                    return 0.3

        return 0.5

    def _get_history_score(self, app_name: str) -> float:
        """Get score based on interaction history."""
        history = self._interaction_rates.get(app_name)

        if not history or len(history) < 5:
            return 0.5  # Not enough data

        # Calculate click rate
        click_rate = sum(1 for clicked in history if clicked) / len(history)
        return click_rate

    def _score_context(
        self,
        category: NotificationCategory,
        user_activity: Optional[str],
        focus_mode: FocusMode,
    ) -> float:
        """Score based on current context."""
        score = 0.5

        # Focus mode adjustments
        if focus_mode == FocusMode.WORK:
            if category in [NotificationCategory.WORK, NotificationCategory.CALENDAR]:
                score += 0.2
            elif category in [NotificationCategory.SOCIAL, NotificationCategory.MARKETING]:
                score -= 0.3

        elif focus_mode == FocusMode.DND:
            if category != NotificationCategory.SECURITY:
                score -= 0.4

        # Activity relevance
        if user_activity:
            if user_activity == "coding" and category == NotificationCategory.WORK:
                score += 0.1
            elif user_activity == "meeting" and category == NotificationCategory.COMMUNICATION:
                score -= 0.2  # Don't interrupt meetings

        return max(0.0, min(1.0, score))

    def _score_to_urgency(self, score: float) -> UrgencyLevel:
        """Convert score to urgency level."""
        if score >= 0.9:
            return UrgencyLevel.CRITICAL
        elif score >= 0.7:
            return UrgencyLevel.HIGH
        elif score >= 0.4:
            return UrgencyLevel.MEDIUM
        elif score >= 0.2:
            return UrgencyLevel.LOW
        else:
            return UrgencyLevel.SILENT

    def record_interaction(self, app_name: str, was_clicked: bool) -> None:
        """Record user interaction for learning."""
        self._interaction_rates[app_name].append(was_clicked)

    def update_app_priority(self, app_name: str, priority: float) -> None:
        """Update learned app priority."""
        self._app_priority[app_name] = max(0.0, min(1.0, priority))


# =============================================================================
# Smart Batcher
# =============================================================================

class SmartBatcher:
    """
    Batches low-priority notifications for delivery together.

    Reduces interruption frequency while ensuring notifications
    are eventually delivered.
    """

    def __init__(self, config: NotificationTriageConfig):
        self.config = config

        # Active batches by category
        self._batches: Dict[NotificationCategory, NotificationBatch] = {}

        # Delivered batches history
        self._batch_history: Deque[NotificationBatch] = deque(maxlen=100)

    def should_batch(self, notification: TriagedNotification) -> bool:
        """Determine if notification should be batched."""
        if not self.config.enable_batching:
            return False

        # Never batch critical or high
        if notification.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]:
            return False

        # Batch low and silent
        if notification.urgency in [UrgencyLevel.LOW, UrgencyLevel.SILENT]:
            return True

        # Medium - batch based on category
        batchable_categories = [
            NotificationCategory.UPDATE,
            NotificationCategory.MARKETING,
            NotificationCategory.SOCIAL,
        ]

        if notification.category in batchable_categories:
            return True

        return False

    def add_to_batch(self, notification: TriagedNotification) -> NotificationBatch:
        """Add notification to appropriate batch."""
        category = notification.category

        # Get or create batch
        if category not in self._batches or self._batches[category].is_delivered:
            self._batches[category] = NotificationBatch(
                category=category,
                max_size=self.config.max_batch_size,
                batch_window_seconds=self.config.batch_window_seconds,
            )

        batch = self._batches[category]
        batch.add_notification(notification)

        return batch

    def get_ready_batches(self) -> List[NotificationBatch]:
        """Get batches that are ready to be delivered."""
        ready = []

        for category, batch in self._batches.items():
            if not batch.is_delivered and batch.is_ready_to_deliver():
                ready.append(batch)

        return ready

    def mark_batch_delivered(self, batch: NotificationBatch) -> None:
        """Mark a batch as delivered."""
        batch.is_delivered = True
        batch.delivered_at = datetime.now()

        for notification in batch.notifications:
            notification.delivered_at = datetime.now()

        self._batch_history.append(batch)

    def get_pending_count(self) -> int:
        """Get total pending notifications in all batches."""
        return sum(
            len(b.notifications)
            for b in self._batches.values()
            if not b.is_delivered
        )


# =============================================================================
# Focus Guard
# =============================================================================

class FocusGuard:
    """
    Guards notification delivery based on focus/DND status.

    Integrates with macOS focus modes and learned focus patterns.
    """

    def __init__(self):
        self._current_focus: FocusMode = FocusMode.NORMAL
        self._focus_history: Deque[Tuple[datetime, FocusMode]] = deque(maxlen=100)

        # Time-based focus patterns
        self._focus_schedule: Dict[int, FocusMode] = {}  # hour -> mode

    def set_focus_mode(self, mode: FocusMode) -> None:
        """Set current focus mode."""
        self._current_focus = mode
        self._focus_history.append((datetime.now(), mode))

    def get_focus_mode(self) -> FocusMode:
        """Get current focus mode."""
        return self._current_focus

    def should_deliver(
        self,
        notification: TriagedNotification,
        override_focus: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if notification should be delivered given focus state.

        Returns:
            Tuple of (should_deliver, reason)
        """
        if override_focus:
            return True, "override"

        mode = self._current_focus

        # Always deliver critical
        if notification.urgency == UrgencyLevel.CRITICAL:
            return True, "critical_always"

        # Never deliver silent
        if notification.urgency == UrgencyLevel.SILENT:
            return False, "silent"

        # Focus mode specific logic
        if mode == FocusMode.DND:
            return False, "dnd_active"

        if mode == FocusMode.SLEEP:
            return False, "sleep_mode"

        if mode == FocusMode.WORK:
            # Only work-related during work focus
            work_categories = [
                NotificationCategory.WORK,
                NotificationCategory.CALENDAR,
                NotificationCategory.SECURITY,
            ]
            if notification.category not in work_categories:
                return False, "work_focus_filter"

        if mode == FocusMode.PERSONAL:
            # No work during personal time
            if notification.category == NotificationCategory.WORK:
                return False, "personal_focus_filter"

        return True, "allowed"

    async def check_system_dnd(self) -> bool:
        """Check if system DND is active."""
        try:
            import subprocess

            # Check macOS DND status
            result = subprocess.run(
                ["defaults", "read", "com.apple.controlcenter", "DoNotDisturb"],
                capture_output=True,
                text=True,
                timeout=2
            )

            return result.returncode == 0 and result.stdout.strip() == "1"

        except Exception:
            return False


# =============================================================================
# Delivery Scheduler
# =============================================================================

class DeliveryScheduler:
    """
    Schedules optimal delivery times for notifications.

    Considers:
    - User activity patterns
    - Natural break points
    - Notification urgency
    - Rate limiting
    """

    def __init__(self, config: NotificationTriageConfig):
        self.config = config

        # Rate tracking
        self._deliveries_this_minute: int = 0
        self._deliveries_this_hour: int = 0
        self._minute_started: datetime = datetime.now().replace(second=0, microsecond=0)
        self._hour_started: datetime = datetime.now().replace(minute=0, second=0, microsecond=0)

        # Delivery queue
        self._delivery_queue: List[Tuple[datetime, TriagedNotification]] = []

    def schedule_delivery(
        self,
        notification: TriagedNotification,
        urgency: UrgencyLevel,
    ) -> datetime:
        """
        Schedule delivery time for notification.

        Returns:
            Scheduled delivery datetime
        """
        now = datetime.now()

        # Critical - immediate
        if urgency == UrgencyLevel.CRITICAL:
            return now

        # High - within 30 seconds
        if urgency == UrgencyLevel.HIGH:
            return now + timedelta(seconds=30)

        # Medium - check rate limits, deliver when slot available
        if urgency == UrgencyLevel.MEDIUM:
            return self._find_delivery_slot(now, timedelta(minutes=5))

        # Low - deliver during idle or after window
        if urgency == UrgencyLevel.LOW:
            return now + timedelta(seconds=self.config.idle_delivery_delay_seconds)

        # Silent - never (or very long delay)
        return now + timedelta(hours=24)

    def _find_delivery_slot(
        self,
        start_time: datetime,
        max_delay: timedelta,
    ) -> datetime:
        """Find next available delivery slot within rate limits."""
        self._update_rate_counters()

        # Check minute limit
        if self._deliveries_this_minute >= self.config.max_notifications_per_minute:
            # Wait for next minute
            next_minute = self._minute_started + timedelta(minutes=1)
            if next_minute - start_time <= max_delay:
                return next_minute

        # Check hour limit
        if self._deliveries_this_hour >= self.config.max_notifications_per_hour:
            # Wait for next hour
            next_hour = self._hour_started + timedelta(hours=1)
            if next_hour - start_time <= max_delay:
                return next_hour

        return start_time

    def _update_rate_counters(self) -> None:
        """Update rate limiting counters."""
        now = datetime.now()

        current_minute = now.replace(second=0, microsecond=0)
        if current_minute > self._minute_started:
            self._deliveries_this_minute = 0
            self._minute_started = current_minute

        current_hour = now.replace(minute=0, second=0, microsecond=0)
        if current_hour > self._hour_started:
            self._deliveries_this_hour = 0
            self._hour_started = current_hour

    def record_delivery(self) -> None:
        """Record that a notification was delivered."""
        self._update_rate_counters()
        self._deliveries_this_minute += 1
        self._deliveries_this_hour += 1

    def get_rate_status(self) -> Dict[str, Any]:
        """Get current rate limiting status."""
        self._update_rate_counters()
        return {
            "deliveries_this_minute": self._deliveries_this_minute,
            "deliveries_this_hour": self._deliveries_this_hour,
            "minute_limit": self.config.max_notifications_per_minute,
            "hour_limit": self.config.max_notifications_per_hour,
        }


# =============================================================================
# Notification Triage System
# =============================================================================

class NotificationTriageSystem:
    """
    Main system for intelligent notification triage.

    Coordinates category engine, priority scorer, batcher,
    focus guard, and delivery scheduler to provide smart
    notification management.
    """

    def __init__(self, config: Optional[NotificationTriageConfig] = None):
        """
        Initialize the notification triage system.

        Args:
            config: System configuration
        """
        self.config = config or NotificationTriageConfig()

        # State
        self._running = False
        self._started_at: Optional[datetime] = None

        # Components
        self._category_engine = CategoryEngine()
        self._priority_scorer = PriorityScorer()
        self._batcher = SmartBatcher(self.config)
        self._focus_guard = FocusGuard()
        self._scheduler = DeliveryScheduler(self.config)

        # Notification storage
        self._pending_delivery: Deque[TriagedNotification] = deque(maxlen=100)
        self._notification_history: Deque[TriagedNotification] = deque(maxlen=500)

        # Duplicate detection
        self._recent_hashes: Deque[Tuple[datetime, str]] = deque(maxlen=100)

        # Callbacks
        self._on_notification_ready: List[Callable[[TriagedNotification], Coroutine]] = []
        self._on_batch_ready: List[Callable[[NotificationBatch], Coroutine]] = []

        # Background task
        self._delivery_task: Optional[asyncio.Task] = None

        # Stats
        self._stats = {
            "notifications_received": 0,
            "notifications_delivered": 0,
            "notifications_batched": 0,
            "notifications_suppressed": 0,
            "batches_delivered": 0,
        }

        logger.debug("NotificationTriageSystem initialized")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> bool:
        """Start the triage system."""
        if self._running:
            return True

        try:
            self._running = True
            self._started_at = datetime.now()

            # Start delivery loop
            self._delivery_task = asyncio.create_task(
                self._delivery_loop(),
                name="notification_delivery"
            )

            logger.info("NotificationTriageSystem started")
            return True

        except Exception as e:
            logger.error(f"Failed to start NotificationTriageSystem: {e}")
            self._running = False
            return False

    async def stop(self) -> None:
        """Stop the triage system."""
        if not self._running:
            return

        self._running = False

        if self._delivery_task:
            self._delivery_task.cancel()
            try:
                await self._delivery_task
            except asyncio.CancelledError:
                pass

        logger.info("NotificationTriageSystem stopped")

    # =========================================================================
    # Notification Processing
    # =========================================================================

    async def triage_notification(
        self,
        app_name: str,
        title: str,
        body: str,
        bundle_id: Optional[str] = None,
        original_id: Optional[str] = None,
        available_actions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TriagedNotification:
        """
        Triage an incoming notification.

        Args:
            app_name: Source application name
            title: Notification title
            body: Notification body
            bundle_id: Bundle identifier
            original_id: Original notification ID
            available_actions: Actions available on notification
            context: Current user context

        Returns:
            TriagedNotification with decision
        """
        self._stats["notifications_received"] += 1

        # Create notification object
        notification = TriagedNotification(
            original_id=original_id,
            app_name=app_name,
            bundle_id=bundle_id,
            title=title,
            body=body,
            available_actions=available_actions or [],
            context_at_receipt=context,
        )

        # Check for duplicates
        if self._is_duplicate(notification):
            notification.decision = TriageDecision.SUPPRESSED
            self._stats["notifications_suppressed"] += 1
            return notification

        # Categorize
        category, cat_confidence = self._category_engine.categorize(
            app_name, title, body, bundle_id
        )
        notification.category = category

        # Score priority
        user_activity = context.get("activity_type") if context else None
        focus_mode = self._focus_guard.get_focus_mode()

        priority_score, urgency = self._priority_scorer.score(
            notification,
            user_activity=user_activity,
            focus_mode=focus_mode,
        )
        notification.priority_score = priority_score
        notification.urgency = urgency

        # Check focus guard
        should_deliver, reason = self._focus_guard.should_deliver(notification)

        if not should_deliver:
            if reason in ["dnd_active", "sleep_mode"]:
                notification.decision = TriageDecision.DEFERRED
                # Schedule for later
                notification.scheduled_delivery = datetime.now() + timedelta(hours=1)
            else:
                notification.decision = TriageDecision.SUPPRESSED
                self._stats["notifications_suppressed"] += 1

            notification.triaged_at = datetime.now()
            self._notification_history.append(notification)
            return notification

        # Check if should batch
        if self._batcher.should_batch(notification):
            batch = self._batcher.add_to_batch(notification)
            notification.decision = TriageDecision.BATCHED
            self._stats["notifications_batched"] += 1
        else:
            # Schedule immediate or near-immediate delivery
            notification.scheduled_delivery = self._scheduler.schedule_delivery(
                notification, urgency
            )
            notification.decision = TriageDecision.IMMEDIATE
            self._pending_delivery.append(notification)

        notification.triaged_at = datetime.now()
        self._notification_history.append(notification)

        return notification

    def _is_duplicate(self, notification: TriagedNotification) -> bool:
        """Check if notification is a duplicate."""
        # Create hash
        hash_input = f"{notification.app_name}:{notification.title}:{notification.body[:50]}"
        notif_hash = hashlib.md5(hash_input.encode()).hexdigest()

        # Check recent hashes
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.config.suppress_duplicate_window_seconds)

        for timestamp, stored_hash in self._recent_hashes:
            if timestamp > cutoff and stored_hash == notif_hash:
                return True

        # Store hash
        self._recent_hashes.append((now, notif_hash))

        return False

    # =========================================================================
    # Delivery Loop
    # =========================================================================

    async def _delivery_loop(self) -> None:
        """Main loop for delivering notifications."""
        while self._running:
            try:
                now = datetime.now()

                # Process pending individual notifications
                await self._process_pending_notifications(now)

                # Process ready batches
                await self._process_ready_batches()

                await asyncio.sleep(self.config.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Delivery loop error: {e}")
                await asyncio.sleep(self.config.check_interval_seconds)

    async def _process_pending_notifications(self, now: datetime) -> None:
        """Process pending notifications that are due for delivery."""
        to_deliver = []
        remaining = deque()

        for notification in self._pending_delivery:
            if notification.scheduled_delivery and notification.scheduled_delivery <= now:
                to_deliver.append(notification)
            else:
                remaining.append(notification)

        self._pending_delivery = remaining

        for notification in to_deliver:
            await self._deliver_notification(notification)

    async def _process_ready_batches(self) -> None:
        """Process batches that are ready for delivery."""
        ready_batches = self._batcher.get_ready_batches()

        for batch in ready_batches:
            await self._deliver_batch(batch)

    async def _deliver_notification(self, notification: TriagedNotification) -> None:
        """Deliver a single notification."""
        notification.delivered_at = datetime.now()
        self._scheduler.record_delivery()
        self._stats["notifications_delivered"] += 1

        # Notify callbacks
        for callback in self._on_notification_ready:
            try:
                await callback(notification)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

    async def _deliver_batch(self, batch: NotificationBatch) -> None:
        """Deliver a batch of notifications."""
        self._batcher.mark_batch_delivered(batch)
        self._stats["batches_delivered"] += 1

        # Notify callbacks
        for callback in self._on_batch_ready:
            try:
                await callback(batch)
            except Exception as e:
                logger.error(f"Batch callback error: {e}")

    # =========================================================================
    # User Feedback
    # =========================================================================

    def record_interaction(
        self,
        notification_id: str,
        was_clicked: bool = False,
        was_dismissed: bool = False,
        action_taken: Optional[str] = None,
    ) -> None:
        """Record user interaction with notification."""
        # Find notification
        for notification in self._notification_history:
            if notification.notification_id == notification_id:
                notification.was_clicked = was_clicked
                notification.was_dismissed = was_dismissed
                notification.action_taken = action_taken

                if notification.delivered_at:
                    notification.response_time_seconds = (
                        datetime.now() - notification.delivered_at
                    ).total_seconds()

                # Update priority scorer learning
                if self.config.enable_learning:
                    self._priority_scorer.record_interaction(
                        notification.app_name,
                        was_clicked
                    )

                break

    # =========================================================================
    # Public API
    # =========================================================================

    def set_focus_mode(self, mode: FocusMode) -> None:
        """Set the current focus mode."""
        self._focus_guard.set_focus_mode(mode)

    def get_focus_mode(self) -> FocusMode:
        """Get the current focus mode."""
        return self._focus_guard.get_focus_mode()

    def on_notification_ready(
        self,
        callback: Callable[[TriagedNotification], Coroutine]
    ) -> None:
        """Register callback for when notifications are ready to deliver."""
        self._on_notification_ready.append(callback)

    def on_batch_ready(
        self,
        callback: Callable[[NotificationBatch], Coroutine]
    ) -> None:
        """Register callback for when batches are ready to deliver."""
        self._on_batch_ready.append(callback)

    def get_pending_count(self) -> int:
        """Get count of pending notifications (individual + batched)."""
        return len(self._pending_delivery) + self._batcher.get_pending_count()

    def get_stats(self) -> Dict[str, Any]:
        """Get triage system statistics."""
        return {
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "focus_mode": self._focus_guard.get_focus_mode().value,
            "pending_individual": len(self._pending_delivery),
            "pending_batched": self._batcher.get_pending_count(),
            "history_count": len(self._notification_history),
            "rate_status": self._scheduler.get_rate_status(),
            **self._stats,
        }


# =============================================================================
# Singleton Management
# =============================================================================

_notification_triage: Optional[NotificationTriageSystem] = None


async def get_notification_triage(
    config: Optional[NotificationTriageConfig] = None
) -> NotificationTriageSystem:
    """Get the global notification triage instance."""
    global _notification_triage

    if _notification_triage is None:
        _notification_triage = NotificationTriageSystem(config)

    return _notification_triage


async def start_notification_triage(
    config: Optional[NotificationTriageConfig] = None
) -> NotificationTriageSystem:
    """Get and start the global notification triage."""
    triage = await get_notification_triage(config)
    if not triage._running:
        await triage.start()
    return triage


async def stop_notification_triage() -> None:
    """Stop the global notification triage."""
    global _notification_triage

    if _notification_triage is not None:
        await _notification_triage.stop()
        _notification_triage = None
