"""
JARVIS macOS Helper - Focus Tracker & Productivity Insights

Tracks user focus patterns, measures productivity, and generates
actionable insights to help users understand and improve their work habits.

Features:
- Focus session detection and tracking
- Deep work vs shallow work classification
- Distraction detection and pattern analysis
- Productivity scoring with contextual adjustment
- Break reminders and wellness suggestions
- Daily/weekly productivity reports
- Learning from user patterns (no hardcoding)

Architecture:
    FocusTracker
    ├── ActivityMonitor (track app/window usage)
    ├── FocusScorer (rate focus quality)
    ├── SessionTracker (manage focus sessions)
    ├── DistractionDetector (identify interruptions)
    ├── InsightEngine (generate productivity insights)
    └── ReportGenerator (create summaries)

Insights generated:
- "Your best focus time is between 9-11 AM"
- "Slack is your biggest distraction source"
- "You've been coding for 2 hours - time for a break"
- "Today's focus score: 78/100"
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class FocusState(str, Enum):
    """User focus states."""
    DEEP_FOCUS = "deep_focus"       # Concentrated, minimal context switches
    LIGHT_FOCUS = "light_focus"     # Working but not deeply focused
    DISTRACTED = "distracted"       # Frequent context switches
    BREAK = "break"                 # Taking a break
    IDLE = "idle"                   # No activity
    MEETING = "meeting"             # In a meeting
    UNKNOWN = "unknown"


class ActivityCategory(str, Enum):
    """Categories of activities for productivity analysis."""
    PRODUCTIVE = "productive"       # Core work activities
    NEUTRAL = "neutral"             # Neither productive nor distracting
    DISTRACTING = "distracting"     # Activities that break focus
    COMMUNICATION = "communication" # Messages, email, calls
    BREAK = "break"                 # Intentional rest


class InsightType(str, Enum):
    """Types of productivity insights."""
    PATTERN = "pattern"             # Observed behavioral pattern
    SUGGESTION = "suggestion"       # Actionable suggestion
    MILESTONE = "milestone"         # Achievement or milestone
    WARNING = "warning"             # Potential issue
    SUMMARY = "summary"             # Summary statistic


class BreakRecommendation(str, Enum):
    """Break recommendation levels."""
    NONE = "none"                   # No break needed
    SUGGESTED = "suggested"         # Break would be helpful
    RECOMMENDED = "recommended"     # Should take a break
    URGENT = "urgent"              # Really need a break


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class FocusSession:
    """A tracked focus session."""
    session_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    duration_minutes: float = 0.0

    # Focus quality
    focus_state: FocusState = FocusState.UNKNOWN
    focus_score: float = 0.0  # 0-100
    deep_focus_minutes: float = 0.0

    # Activity breakdown
    primary_app: Optional[str] = None
    primary_activity: Optional[str] = None
    apps_used: List[str] = field(default_factory=list)
    context_switches: int = 0

    # Interruptions
    distraction_count: int = 0
    distraction_sources: List[str] = field(default_factory=list)
    notification_interruptions: int = 0

    # State
    is_active: bool = True
    was_productive: bool = True

    def end_session(self) -> None:
        """End the focus session."""
        self.ended_at = datetime.now()
        self.is_active = False
        self.duration_minutes = (self.ended_at - self.started_at).total_seconds() / 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_minutes": self.duration_minutes,
            "focus_state": self.focus_state.value,
            "focus_score": self.focus_score,
            "primary_app": self.primary_app,
            "context_switches": self.context_switches,
            "distraction_count": self.distraction_count,
        }


@dataclass
class ProductivityInsight:
    """A productivity insight or suggestion."""
    insight_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:10])
    insight_type: InsightType = InsightType.PATTERN

    # Content
    title: str = ""
    message: str = ""
    detail: Optional[str] = None

    # Relevance
    confidence: float = 0.0
    relevance_score: float = 0.0

    # Context
    related_app: Optional[str] = None
    related_time_range: Optional[str] = None
    data_points: int = 0  # How much data supports this

    # Timing
    generated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # State
    was_shown: bool = False
    was_helpful: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type.value,
            "title": self.title,
            "message": self.message,
            "confidence": self.confidence,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class DailyProductivity:
    """Daily productivity summary."""
    date: date = field(default_factory=date.today)

    # Time tracking
    total_active_minutes: float = 0.0
    productive_minutes: float = 0.0
    deep_focus_minutes: float = 0.0
    break_minutes: float = 0.0

    # Sessions
    session_count: int = 0
    avg_session_duration: float = 0.0
    longest_session_minutes: float = 0.0

    # Quality metrics
    productivity_score: float = 0.0  # 0-100
    focus_score: float = 0.0  # 0-100
    distraction_score: float = 0.0  # Higher = more distracted

    # App usage
    app_time: Dict[str, float] = field(default_factory=dict)
    top_apps: List[Tuple[str, float]] = field(default_factory=list)

    # Patterns
    most_productive_hour: Optional[int] = None
    peak_focus_time: Optional[str] = None

    # Distractions
    total_context_switches: int = 0
    total_distractions: int = 0
    top_distractions: List[str] = field(default_factory=list)


@dataclass
class FocusTrackerConfig:
    """Configuration for the focus tracker."""
    # Session detection
    session_start_threshold_seconds: float = float(os.getenv("FOCUS_SESSION_START", "60.0"))
    session_end_idle_seconds: float = float(os.getenv("FOCUS_SESSION_END_IDLE", "300.0"))
    min_session_duration_minutes: float = float(os.getenv("FOCUS_MIN_SESSION", "5.0"))

    # Focus scoring
    context_switch_penalty: float = float(os.getenv("FOCUS_SWITCH_PENALTY", "2.0"))
    distraction_penalty: float = float(os.getenv("FOCUS_DISTRACTION_PENALTY", "5.0"))
    deep_focus_threshold: float = float(os.getenv("FOCUS_DEEP_THRESHOLD", "70.0"))

    # Breaks
    enable_break_reminders: bool = os.getenv("FOCUS_BREAK_REMINDERS", "true").lower() == "true"
    break_reminder_interval_minutes: float = float(os.getenv("FOCUS_BREAK_INTERVAL", "90.0"))
    min_break_duration_minutes: float = float(os.getenv("FOCUS_MIN_BREAK", "5.0"))

    # Insights
    enable_insights: bool = os.getenv("FOCUS_ENABLE_INSIGHTS", "true").lower() == "true"
    min_data_points_for_insight: int = int(os.getenv("FOCUS_MIN_DATA_POINTS", "5"))
    insight_generation_interval_hours: float = float(os.getenv("FOCUS_INSIGHT_INTERVAL", "4.0"))

    # Tracking intervals
    update_interval_seconds: float = float(os.getenv("FOCUS_UPDATE_INTERVAL", "10.0"))

    # Storage
    max_sessions_history: int = int(os.getenv("FOCUS_MAX_SESSIONS", "100"))
    max_insights_history: int = int(os.getenv("FOCUS_MAX_INSIGHTS", "50"))


# =============================================================================
# Activity Classification
# =============================================================================

class ActivityClassificationEngine:
    """
    Classifies activities as productive, neutral, or distracting.

    Uses learned patterns with fallback to defaults.
    """

    def __init__(self):
        # Learned classifications
        self._app_classifications: Dict[str, ActivityCategory] = {}

        # Default classifications
        self._default_productive = {
            "Cursor", "Visual Studio Code", "Xcode", "Terminal", "iTerm2",
            "Warp", "PyCharm", "IntelliJ IDEA", "WebStorm", "Android Studio",
            "Notion", "Obsidian", "Bear", "Sublime Text", "Figma",
            "Adobe Photoshop", "Adobe Illustrator", "Sketch",
            "Microsoft Word", "Microsoft Excel", "Pages", "Numbers",
        }

        self._default_communication = {
            "Slack", "Microsoft Teams", "Discord", "Messages",
            "Mail", "Outlook", "Spark", "Gmail", "Zoom",
            "FaceTime", "Google Meet", "Webex",
        }

        self._default_distracting = {
            "Twitter", "Facebook", "Instagram", "TikTok",
            "YouTube", "Netflix", "Twitch", "Reddit",
            "News", "Apple News", "Hacker News",
        }

    def classify(
        self,
        app_name: str,
        window_title: Optional[str] = None,
        time_spent_seconds: float = 0.0,
    ) -> Tuple[ActivityCategory, float]:
        """
        Classify an app/activity.

        Returns:
            Tuple of (category, confidence)
        """
        # Check learned classification
        if app_name in self._app_classifications:
            return self._app_classifications[app_name], 0.9

        # Check defaults
        if app_name in self._default_productive:
            return ActivityCategory.PRODUCTIVE, 0.8

        if app_name in self._default_communication:
            # Communication can be productive or distracting
            # Short checks are usually productive, long ones less so
            if time_spent_seconds < 120:  # < 2 minutes
                return ActivityCategory.COMMUNICATION, 0.7
            elif time_spent_seconds > 600:  # > 10 minutes
                return ActivityCategory.DISTRACTING, 0.6
            else:
                return ActivityCategory.NEUTRAL, 0.5

        if app_name in self._default_distracting:
            return ActivityCategory.DISTRACTING, 0.8

        # Heuristics based on window title
        if window_title:
            title_lower = window_title.lower()

            # Coding indicators
            code_indicators = [".py", ".js", ".ts", ".go", ".rs", "git", "github"]
            if any(ind in title_lower for ind in code_indicators):
                return ActivityCategory.PRODUCTIVE, 0.7

            # Entertainment indicators
            entertainment = ["video", "stream", "episode", "movie"]
            if any(ind in title_lower for ind in entertainment):
                return ActivityCategory.DISTRACTING, 0.7

        return ActivityCategory.NEUTRAL, 0.5

    def learn_classification(
        self,
        app_name: str,
        category: ActivityCategory,
    ) -> None:
        """Learn a new app classification."""
        self._app_classifications[app_name] = category


# =============================================================================
# Focus Scorer
# =============================================================================

class FocusScorer:
    """
    Scores focus quality based on activity patterns.

    Considers:
    - Time in productive apps
    - Context switch frequency
    - Distraction exposure
    - Session continuity
    """

    def __init__(self, config: FocusTrackerConfig):
        self.config = config

        # Scoring history for calibration
        self._score_history: Deque[float] = deque(maxlen=100)

    def score_session(
        self,
        productive_minutes: float,
        total_minutes: float,
        context_switches: int,
        distractions: int,
    ) -> float:
        """
        Calculate focus score for a session.

        Returns:
            Score from 0-100
        """
        if total_minutes <= 0:
            return 0.0

        # Base score from productive time ratio
        productive_ratio = productive_minutes / total_minutes
        base_score = productive_ratio * 100

        # Penalty for context switches
        switch_rate = context_switches / max(total_minutes, 1)
        switch_penalty = switch_rate * self.config.context_switch_penalty * 10

        # Penalty for distractions
        distraction_rate = distractions / max(total_minutes, 1)
        distraction_penalty = distraction_rate * self.config.distraction_penalty * 10

        # Calculate final score
        final_score = max(0, base_score - switch_penalty - distraction_penalty)
        final_score = min(100, final_score)

        self._score_history.append(final_score)

        return round(final_score, 1)

    def score_focus_state(
        self,
        recent_switches: int,
        window_seconds: float,
        in_productive_app: bool,
    ) -> FocusState:
        """Determine current focus state."""
        if window_seconds <= 0:
            return FocusState.UNKNOWN

        switch_rate = recent_switches / (window_seconds / 60)  # per minute

        if switch_rate > 3:
            return FocusState.DISTRACTED
        elif switch_rate > 1:
            return FocusState.LIGHT_FOCUS
        elif in_productive_app and switch_rate < 0.5:
            return FocusState.DEEP_FOCUS
        else:
            return FocusState.LIGHT_FOCUS

    def get_average_score(self) -> float:
        """Get average focus score from history."""
        if not self._score_history:
            return 50.0
        return sum(self._score_history) / len(self._score_history)


# =============================================================================
# Distraction Detector
# =============================================================================

class DistractionDetector:
    """
    Detects when user is distracted and identifies distraction sources.

    Tracks:
    - Quick app switches (context switch storms)
    - Time spent in distracting apps
    - Notification-triggered distractions
    - Pattern of distraction behavior
    """

    def __init__(self, config: FocusTrackerConfig):
        self.config = config

        # Recent activity tracking
        self._recent_apps: Deque[Tuple[datetime, str]] = deque(maxlen=50)
        self._recent_notifications: Deque[Tuple[datetime, str]] = deque(maxlen=20)

        # Distraction statistics
        self._distraction_sources: Dict[str, int] = defaultdict(int)
        self._hourly_distractions: Dict[int, int] = defaultdict(int)

    def record_app_switch(self, app_name: str) -> None:
        """Record an app switch."""
        self._recent_apps.append((datetime.now(), app_name))

    def record_notification(self, app_name: str) -> None:
        """Record a notification."""
        self._recent_notifications.append((datetime.now(), app_name))

    def detect_distraction(
        self,
        current_app: str,
        classification_engine: ActivityClassificationEngine,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if current state indicates distraction.

        Returns:
            Tuple of (is_distracted, distraction_source)
        """
        now = datetime.now()

        # Check if current app is distracting
        category, _ = classification_engine.classify(current_app)
        if category == ActivityCategory.DISTRACTING:
            self._distraction_sources[current_app] += 1
            self._hourly_distractions[now.hour] += 1
            return True, current_app

        # Check for context switch storm (many switches in short time)
        recent_window = timedelta(minutes=2)
        recent_switches = [
            app for ts, app in self._recent_apps
            if now - ts < recent_window
        ]

        if len(set(recent_switches)) > 5:
            return True, "context_switch_storm"

        # Check for notification-triggered distraction
        recent_notif_window = timedelta(seconds=30)
        recent_notifs = [
            app for ts, app in self._recent_notifications
            if now - ts < recent_notif_window
        ]

        if recent_notifs and current_app in self._recent_apps:
            # User switched to app right after notification
            return True, f"notification_from_{recent_notifs[-1]}"

        return False, None

    def get_top_distractions(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get top distraction sources."""
        sorted_sources = sorted(
            self._distraction_sources.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_sources[:limit]

    def get_most_distracting_hours(self) -> List[Tuple[int, int]]:
        """Get hours with most distractions."""
        sorted_hours = sorted(
            self._hourly_distractions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_hours[:5]


# =============================================================================
# Insight Engine
# =============================================================================

class InsightEngine:
    """
    Generates productivity insights from tracked data.

    Analyzes patterns to provide actionable suggestions.
    """

    def __init__(self, config: FocusTrackerConfig):
        self.config = config
        self._generated_insights: Deque[ProductivityInsight] = deque(
            maxlen=config.max_insights_history
        )

    def generate_insights(
        self,
        sessions: List[FocusSession],
        daily_data: Optional[DailyProductivity] = None,
        distraction_detector: Optional[DistractionDetector] = None,
    ) -> List[ProductivityInsight]:
        """Generate insights from available data."""
        insights = []

        if len(sessions) < self.config.min_data_points_for_insight:
            return insights

        # Analyze best focus times
        time_insight = self._analyze_best_focus_times(sessions)
        if time_insight:
            insights.append(time_insight)

        # Analyze distraction patterns
        if distraction_detector:
            distraction_insight = self._analyze_distraction_patterns(distraction_detector)
            if distraction_insight:
                insights.append(distraction_insight)

        # Analyze session patterns
        session_insight = self._analyze_session_patterns(sessions)
        if session_insight:
            insights.append(session_insight)

        # Generate daily summary if available
        if daily_data:
            summary_insight = self._generate_daily_summary(daily_data)
            if summary_insight:
                insights.append(summary_insight)

        self._generated_insights.extend(insights)
        return insights

    def _analyze_best_focus_times(
        self,
        sessions: List[FocusSession],
    ) -> Optional[ProductivityInsight]:
        """Analyze when user focuses best."""
        if len(sessions) < 5:
            return None

        # Group sessions by hour
        hour_scores: Dict[int, List[float]] = defaultdict(list)

        for session in sessions:
            hour = session.started_at.hour
            hour_scores[hour].append(session.focus_score)

        # Find best hour
        avg_by_hour = {
            hour: sum(scores) / len(scores)
            for hour, scores in hour_scores.items()
            if len(scores) >= 2
        }

        if not avg_by_hour:
            return None

        best_hour = max(avg_by_hour.keys(), key=lambda h: avg_by_hour[h])
        best_score = avg_by_hour[best_hour]

        # Format time
        time_str = datetime.now().replace(hour=best_hour, minute=0).strftime("%-I %p")

        return ProductivityInsight(
            insight_type=InsightType.PATTERN,
            title="Peak Focus Time",
            message=f"Your best focus time is around {time_str} "
                    f"(avg score: {best_score:.0f}/100)",
            confidence=0.7 if len(sessions) >= 10 else 0.5,
            related_time_range=f"{best_hour}:00",
            data_points=len(sessions),
        )

    def _analyze_distraction_patterns(
        self,
        detector: DistractionDetector,
    ) -> Optional[ProductivityInsight]:
        """Analyze distraction patterns."""
        top_distractions = detector.get_top_distractions(3)

        if not top_distractions:
            return None

        top_source, count = top_distractions[0]

        if count < 3:
            return None

        return ProductivityInsight(
            insight_type=InsightType.SUGGESTION,
            title="Distraction Source",
            message=f"{top_source} is your top distraction source "
                    f"({count} times today). Consider blocking during focus time.",
            confidence=0.8,
            related_app=top_source,
            data_points=count,
        )

    def _analyze_session_patterns(
        self,
        sessions: List[FocusSession],
    ) -> Optional[ProductivityInsight]:
        """Analyze focus session patterns."""
        if len(sessions) < 5:
            return None

        # Calculate average session duration
        durations = [s.duration_minutes for s in sessions if s.duration_minutes > 0]
        if not durations:
            return None

        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)

        if avg_duration < 30:
            return ProductivityInsight(
                insight_type=InsightType.SUGGESTION,
                title="Short Sessions",
                message=f"Your avg focus session is {avg_duration:.0f} min. "
                        "Try extending to 45-90 min for deeper work.",
                confidence=0.7,
                data_points=len(durations),
            )
        elif avg_duration > 120:
            return ProductivityInsight(
                insight_type=InsightType.WARNING,
                title="Long Sessions",
                message=f"Sessions averaging {avg_duration:.0f} min. "
                        "Consider taking more breaks to maintain quality.",
                confidence=0.7,
                data_points=len(durations),
            )

        return None

    def _generate_daily_summary(
        self,
        daily: DailyProductivity,
    ) -> Optional[ProductivityInsight]:
        """Generate daily productivity summary."""
        if daily.total_active_minutes < 30:
            return None

        return ProductivityInsight(
            insight_type=InsightType.SUMMARY,
            title="Daily Summary",
            message=f"Today: {daily.productive_minutes:.0f}m productive, "
                    f"{daily.deep_focus_minutes:.0f}m deep focus. "
                    f"Score: {daily.productivity_score:.0f}/100",
            confidence=0.9,
            data_points=daily.session_count,
        )

    def get_recent_insights(self, limit: int = 10) -> List[ProductivityInsight]:
        """Get recent insights."""
        return list(self._generated_insights)[-limit:]


# =============================================================================
# Break Manager
# =============================================================================

class BreakManager:
    """
    Manages break reminders and tracking.

    Monitors continuous work time and suggests breaks.
    """

    def __init__(self, config: FocusTrackerConfig):
        self.config = config

        self._continuous_work_started: Optional[datetime] = None
        self._last_break_ended: Optional[datetime] = None
        self._breaks_taken_today: int = 0

    def record_activity_start(self) -> None:
        """Record start of activity after break/idle."""
        if self._continuous_work_started is None:
            self._continuous_work_started = datetime.now()

    def record_break_start(self) -> None:
        """Record start of a break."""
        self._continuous_work_started = None

    def record_break_end(self) -> None:
        """Record end of a break."""
        self._last_break_ended = datetime.now()
        self._breaks_taken_today += 1
        self._continuous_work_started = datetime.now()

    def get_continuous_work_minutes(self) -> float:
        """Get minutes since last break."""
        if self._continuous_work_started is None:
            return 0.0

        return (datetime.now() - self._continuous_work_started).total_seconds() / 60

    def check_break_needed(self) -> Tuple[BreakRecommendation, str]:
        """
        Check if user should take a break.

        Returns:
            Tuple of (recommendation_level, message)
        """
        if not self.config.enable_break_reminders:
            return BreakRecommendation.NONE, ""

        work_minutes = self.get_continuous_work_minutes()

        if work_minutes >= self.config.break_reminder_interval_minutes * 1.5:
            return (
                BreakRecommendation.URGENT,
                f"You've been working for {work_minutes:.0f} minutes - "
                "time to rest your eyes and stretch!"
            )
        elif work_minutes >= self.config.break_reminder_interval_minutes:
            return (
                BreakRecommendation.RECOMMENDED,
                f"{work_minutes:.0f} minutes of continuous work. "
                "A short break would help maintain focus."
            )
        elif work_minutes >= self.config.break_reminder_interval_minutes * 0.75:
            return (
                BreakRecommendation.SUGGESTED,
                "Consider taking a break in the next 15-20 minutes."
            )

        return BreakRecommendation.NONE, ""


# =============================================================================
# Focus Tracker
# =============================================================================

class FocusTracker:
    """
    Main focus tracking and productivity analysis system.

    Coordinates all tracking components and provides unified API.
    """

    def __init__(self, config: Optional[FocusTrackerConfig] = None):
        """
        Initialize the focus tracker.

        Args:
            config: Tracker configuration
        """
        self.config = config or FocusTrackerConfig()

        # State
        self._running = False
        self._started_at: Optional[datetime] = None

        # Components
        self._classification_engine = ActivityClassificationEngine()
        self._focus_scorer = FocusScorer(self.config)
        self._distraction_detector = DistractionDetector(self.config)
        self._insight_engine = InsightEngine(self.config)
        self._break_manager = BreakManager(self.config)

        # Current state
        self._current_session: Optional[FocusSession] = None
        self._current_focus_state: FocusState = FocusState.UNKNOWN
        self._current_app: Optional[str] = None
        self._last_activity_time: Optional[datetime] = None

        # History
        self._session_history: Deque[FocusSession] = deque(
            maxlen=self.config.max_sessions_history
        )

        # Daily tracking
        self._daily_data: DailyProductivity = DailyProductivity()
        self._last_daily_reset: date = date.today()

        # Tracking data
        self._app_time_today: Dict[str, float] = defaultdict(float)
        self._context_switches_today: int = 0

        # Callbacks
        self._on_focus_changed: List[Callable[[FocusState, FocusState], Coroutine]] = []
        self._on_session_ended: List[Callable[[FocusSession], Coroutine]] = []
        self._on_insight_generated: List[Callable[[ProductivityInsight], Coroutine]] = []
        self._on_break_recommended: List[Callable[[BreakRecommendation, str], Coroutine]] = []

        # Background task
        self._update_task: Optional[asyncio.Task] = None
        self._insight_task: Optional[asyncio.Task] = None

        # Stats
        self._stats = {
            "sessions_tracked": 0,
            "total_focus_minutes": 0.0,
            "insights_generated": 0,
            "breaks_taken": 0,
        }

        logger.debug("FocusTracker initialized")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> bool:
        """Start the focus tracker."""
        if self._running:
            return True

        try:
            self._running = True
            self._started_at = datetime.now()

            # Start update loop
            self._update_task = asyncio.create_task(
                self._update_loop(),
                name="focus_update"
            )

            # Start insight generation loop
            if self.config.enable_insights:
                self._insight_task = asyncio.create_task(
                    self._insight_loop(),
                    name="focus_insights"
                )

            logger.info("FocusTracker started")
            return True

        except Exception as e:
            logger.error(f"Failed to start FocusTracker: {e}")
            self._running = False
            return False

    async def stop(self) -> None:
        """Stop the focus tracker."""
        if not self._running:
            return

        self._running = False

        # End current session
        if self._current_session:
            await self._end_session()

        # Cancel tasks
        for task in [self._update_task, self._insight_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("FocusTracker stopped")

    # =========================================================================
    # Activity Updates
    # =========================================================================

    def update_activity(
        self,
        app_name: Optional[str] = None,
        window_title: Optional[str] = None,
        is_idle: bool = False,
    ) -> None:
        """
        Update with current activity information.

        Called periodically by external context sources.
        """
        now = datetime.now()

        # Check for day rollover
        if now.date() != self._last_daily_reset:
            self._reset_daily_data()

        # Handle idle
        if is_idle:
            self._handle_idle()
            return

        # Record activity
        self._last_activity_time = now
        self._break_manager.record_activity_start()

        # Check for app change
        if app_name and app_name != self._current_app:
            self._handle_app_change(app_name)

        self._current_app = app_name

        # Update app time tracking
        if app_name:
            self._app_time_today[app_name] += self.config.update_interval_seconds / 60

        # Update current session if exists
        if self._current_session:
            self._update_current_session(app_name, window_title)
        else:
            # Start new session if active
            self._start_session(app_name)

    def _handle_app_change(self, new_app: str) -> None:
        """Handle application change."""
        self._context_switches_today += 1
        self._distraction_detector.record_app_switch(new_app)

        if self._current_session:
            self._current_session.context_switches += 1
            if new_app not in self._current_session.apps_used:
                self._current_session.apps_used.append(new_app)

    def _handle_idle(self) -> None:
        """Handle idle state."""
        if self._current_session:
            idle_duration = 0.0
            if self._last_activity_time:
                idle_duration = (datetime.now() - self._last_activity_time).total_seconds()

            if idle_duration >= self.config.session_end_idle_seconds:
                # End session due to idle
                asyncio.create_task(self._end_session())
                self._break_manager.record_break_start()

        self._current_focus_state = FocusState.IDLE

    # =========================================================================
    # Session Management
    # =========================================================================

    def _start_session(self, app_name: Optional[str]) -> None:
        """Start a new focus session."""
        self._current_session = FocusSession(
            primary_app=app_name,
            apps_used=[app_name] if app_name else [],
        )

    async def _end_session(self) -> None:
        """End current focus session."""
        if not self._current_session:
            return

        session = self._current_session
        session.end_session()

        # Calculate scores
        productive_mins = self._calculate_productive_time(session)
        session.focus_score = self._focus_scorer.score_session(
            productive_minutes=productive_mins,
            total_minutes=session.duration_minutes,
            context_switches=session.context_switches,
            distractions=session.distraction_count,
        )

        # Determine if session meets minimum duration
        if session.duration_minutes >= self.config.min_session_duration_minutes:
            self._session_history.append(session)
            self._stats["sessions_tracked"] += 1
            self._stats["total_focus_minutes"] += session.duration_minutes

            # Update daily data
            self._daily_data.session_count += 1
            self._daily_data.productive_minutes += productive_mins
            self._daily_data.total_active_minutes += session.duration_minutes

            if session.focus_score >= self.config.deep_focus_threshold:
                self._daily_data.deep_focus_minutes += session.duration_minutes

            # Notify callbacks
            for callback in self._on_session_ended:
                try:
                    await callback(session)
                except Exception as e:
                    logger.error(f"Session ended callback error: {e}")

        self._current_session = None

    def _update_current_session(
        self,
        app_name: Optional[str],
        window_title: Optional[str],
    ) -> None:
        """Update current session metrics."""
        if not self._current_session:
            return

        session = self._current_session

        # Update duration
        session.duration_minutes = (
            datetime.now() - session.started_at
        ).total_seconds() / 60

        # Check for distraction
        if app_name:
            is_distracted, source = self._distraction_detector.detect_distraction(
                app_name,
                self._classification_engine,
            )
            if is_distracted:
                session.distraction_count += 1
                if source and source not in session.distraction_sources:
                    session.distraction_sources.append(source)

        # Update primary app (most used)
        if session.apps_used:
            primary = max(
                session.apps_used,
                key=lambda a: self._app_time_today.get(a, 0)
            )
            session.primary_app = primary

    def _calculate_productive_time(self, session: FocusSession) -> float:
        """Calculate productive time in session."""
        productive = 0.0

        for app in session.apps_used:
            category, _ = self._classification_engine.classify(app)
            if category == ActivityCategory.PRODUCTIVE:
                productive += self._app_time_today.get(app, 0)

        # Cap at session duration
        return min(productive, session.duration_minutes)

    # =========================================================================
    # Background Loops
    # =========================================================================

    async def _update_loop(self) -> None:
        """Main update loop."""
        while self._running:
            try:
                now = datetime.now()

                # Update focus state
                await self._update_focus_state()

                # Check break recommendation
                await self._check_break_recommendation()

                await asyncio.sleep(self.config.update_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Focus update loop error: {e}")
                await asyncio.sleep(self.config.update_interval_seconds)

    async def _update_focus_state(self) -> None:
        """Update current focus state."""
        if not self._current_session:
            new_state = FocusState.IDLE
        else:
            # Determine focus state from recent activity
            recent_switches = self._current_session.context_switches
            session_duration = (
                datetime.now() - self._current_session.started_at
            ).total_seconds()

            category, _ = self._classification_engine.classify(
                self._current_app or ""
            )
            in_productive = category == ActivityCategory.PRODUCTIVE

            new_state = self._focus_scorer.score_focus_state(
                recent_switches=recent_switches,
                window_seconds=session_duration,
                in_productive_app=in_productive,
            )

        # Notify if changed
        if new_state != self._current_focus_state:
            old_state = self._current_focus_state
            self._current_focus_state = new_state

            for callback in self._on_focus_changed:
                try:
                    await callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"Focus changed callback error: {e}")

    async def _check_break_recommendation(self) -> None:
        """Check and notify about break recommendations."""
        recommendation, message = self._break_manager.check_break_needed()

        if recommendation != BreakRecommendation.NONE:
            for callback in self._on_break_recommended:
                try:
                    await callback(recommendation, message)
                except Exception as e:
                    logger.error(f"Break recommendation callback error: {e}")

    async def _insight_loop(self) -> None:
        """Periodic insight generation loop."""
        interval = self.config.insight_generation_interval_hours * 3600

        while self._running:
            try:
                await asyncio.sleep(interval)

                # Generate insights
                insights = self._insight_engine.generate_insights(
                    sessions=list(self._session_history),
                    daily_data=self._daily_data,
                    distraction_detector=self._distraction_detector,
                )

                self._stats["insights_generated"] += len(insights)

                # Notify callbacks
                for insight in insights:
                    for callback in self._on_insight_generated:
                        try:
                            await callback(insight)
                        except Exception as e:
                            logger.error(f"Insight callback error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Insight loop error: {e}")

    # =========================================================================
    # Daily Reset
    # =========================================================================

    def _reset_daily_data(self) -> None:
        """Reset daily tracking data."""
        # Store yesterday's data if needed
        self._daily_data = DailyProductivity()
        self._app_time_today.clear()
        self._context_switches_today = 0
        self._last_daily_reset = date.today()
        self._break_manager._breaks_taken_today = 0

    # =========================================================================
    # Public API
    # =========================================================================

    def get_current_focus_state(self) -> FocusState:
        """Get current focus state."""
        return self._current_focus_state

    def get_current_session(self) -> Optional[FocusSession]:
        """Get current active session."""
        return self._current_session

    def get_session_history(self, limit: int = 10) -> List[FocusSession]:
        """Get recent session history."""
        return list(self._session_history)[-limit:]

    def get_daily_productivity(self) -> DailyProductivity:
        """Get today's productivity data."""
        return self._daily_data

    def get_focus_score(self) -> float:
        """Get current focus score."""
        if self._current_session:
            return self._current_session.focus_score
        return 0.0

    def get_productivity_score(self) -> float:
        """Get today's productivity score."""
        if self._daily_data.total_active_minutes <= 0:
            return 0.0

        return min(100, (
            self._daily_data.productive_minutes /
            self._daily_data.total_active_minutes * 100
        ))

    def record_break(self, duration_minutes: float) -> None:
        """Record that user took a break."""
        self._break_manager.record_break_end()
        self._daily_data.break_minutes += duration_minutes
        self._stats["breaks_taken"] += 1

    def on_focus_changed(
        self,
        callback: Callable[[FocusState, FocusState], Coroutine]
    ) -> None:
        """Register callback for focus state changes."""
        self._on_focus_changed.append(callback)

    def on_session_ended(
        self,
        callback: Callable[[FocusSession], Coroutine]
    ) -> None:
        """Register callback for session end."""
        self._on_session_ended.append(callback)

    def on_insight_generated(
        self,
        callback: Callable[[ProductivityInsight], Coroutine]
    ) -> None:
        """Register callback for new insights."""
        self._on_insight_generated.append(callback)

    def on_break_recommended(
        self,
        callback: Callable[[BreakRecommendation, str], Coroutine]
    ) -> None:
        """Register callback for break recommendations."""
        self._on_break_recommended.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "current_focus_state": self._current_focus_state.value,
            "current_session_active": self._current_session is not None,
            "session_history_count": len(self._session_history),
            "today_productive_minutes": self._daily_data.productive_minutes,
            "today_session_count": self._daily_data.session_count,
            "continuous_work_minutes": self._break_manager.get_continuous_work_minutes(),
            **self._stats,
        }


# =============================================================================
# Singleton Management
# =============================================================================

_focus_tracker: Optional[FocusTracker] = None


async def get_focus_tracker(
    config: Optional[FocusTrackerConfig] = None
) -> FocusTracker:
    """Get the global focus tracker instance."""
    global _focus_tracker

    if _focus_tracker is None:
        _focus_tracker = FocusTracker(config)

    return _focus_tracker


async def start_focus_tracker(
    config: Optional[FocusTrackerConfig] = None
) -> FocusTracker:
    """Get and start the global focus tracker."""
    tracker = await get_focus_tracker(config)
    if not tracker._running:
        await tracker.start()
    return tracker


async def stop_focus_tracker() -> None:
    """Stop the global focus tracker."""
    global _focus_tracker

    if _focus_tracker is not None:
        await _focus_tracker.stop()
        _focus_tracker = None
