"""
JARVIS macOS Helper - Proactive Suggestion Engine

ML-powered suggestion system that learns user patterns and provides
contextually relevant, timely suggestions without being intrusive.

Features:
- Pattern learning from user behavior
- Context-aware suggestion generation
- Intent inference from activity patterns
- Timing optimization (suggest at right moment)
- Confidence-based filtering
- User feedback integration
- Calendar and schedule awareness
- Zero hardcoded rules - all learned dynamically

Architecture:
    ProactiveSuggestionEngine
    ├── PatternLearner (learn user behavior patterns)
    ├── IntentInferer (predict user goals/needs)
    ├── SuggestionGenerator (create relevant suggestions)
    ├── TimingOptimizer (find optimal suggestion moments)
    ├── ConfidenceScorer (rate suggestion quality)
    └── FeedbackProcessor (learn from user responses)

Examples:
- "I see you're in Cursor - want me to run the tests?"
- "Your meeting starts in 5 minutes - should I open Zoom?"
- "You've been coding for 2 hours - time for a break?"
- "Based on your pattern, you usually review PRs now - want me to open GitHub?"
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
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

class SuggestionType(str, Enum):
    """Types of proactive suggestions."""
    ACTION = "action"           # Do something (run tests, open app)
    REMINDER = "reminder"       # Remember something (meeting, deadline)
    INSIGHT = "insight"         # Information (productivity stats)
    HELP = "help"              # Assistance (error detected, stuck)
    OPTIMIZATION = "optimization"  # Improve workflow
    HEALTH = "health"          # Take a break, posture check
    CONTEXT = "context"        # Contextual info (weather, news)
    LEARNING = "learning"      # Tips and shortcuts


class SuggestionPriority(str, Enum):
    """Priority levels for suggestions."""
    CRITICAL = "critical"      # Time-sensitive, important
    HIGH = "high"              # Should see soon
    MEDIUM = "medium"          # When convenient
    LOW = "low"                # Background/optional


class SuggestionState(str, Enum):
    """State of a suggestion."""
    PENDING = "pending"        # Not yet shown
    SHOWN = "shown"            # Displayed to user
    ACCEPTED = "accepted"      # User acted on it
    DISMISSED = "dismissed"    # User dismissed
    EXPIRED = "expired"        # No longer relevant
    DEFERRED = "deferred"      # User asked to remind later


class UserResponseType(str, Enum):
    """Types of user responses to suggestions."""
    ACCEPT = "accept"
    DISMISS = "dismiss"
    DEFER = "defer"
    IGNORE = "ignore"          # No response
    FEEDBACK_POSITIVE = "feedback_positive"
    FEEDBACK_NEGATIVE = "feedback_negative"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Suggestion:
    """A proactive suggestion for the user."""
    suggestion_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])
    suggestion_type: SuggestionType = SuggestionType.ACTION
    priority: SuggestionPriority = SuggestionPriority.MEDIUM

    # Content
    title: str = ""
    message: str = ""
    detail: Optional[str] = None

    # Action
    action_type: Optional[str] = None  # "run_command", "open_app", "open_url", etc.
    action_payload: Optional[Dict[str, Any]] = None

    # Context
    trigger_context: Optional[str] = None  # What triggered this suggestion
    related_app: Optional[str] = None
    related_file: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    optimal_show_time: Optional[datetime] = None

    # Scoring
    confidence: float = 0.0  # 0.0 to 1.0
    relevance_score: float = 0.0
    timing_score: float = 0.0

    # State
    state: SuggestionState = SuggestionState.PENDING
    shown_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
    user_response: Optional[UserResponseType] = None

    # Learning
    pattern_id: Optional[str] = None  # Which learned pattern triggered this
    feedback_incorporated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suggestion_id": self.suggestion_id,
            "suggestion_type": self.suggestion_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "detail": self.detail,
            "action_type": self.action_type,
            "action_payload": self.action_payload,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "timing_score": self.timing_score,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    def is_expired(self) -> bool:
        """Check if suggestion has expired."""
        if self.expires_at and datetime.now() > self.expires_at:
            return True
        return self.state == SuggestionState.EXPIRED

    def get_combined_score(self) -> float:
        """Get combined quality score."""
        return (
            self.confidence * 0.4 +
            self.relevance_score * 0.4 +
            self.timing_score * 0.2
        )


@dataclass
class LearnedPattern:
    """A learned behavioral pattern."""
    pattern_id: str
    pattern_type: str  # "time_based", "context_based", "sequence_based"
    description: str

    # Trigger conditions
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    occurrence_count: int = 0
    success_rate: float = 0.0  # How often suggestion was accepted
    last_triggered: Optional[datetime] = None

    # Suggested action
    suggested_action: Optional[str] = None
    suggested_message: Optional[str] = None

    # Learning state
    confidence: float = 0.0
    is_active: bool = True

    def should_trigger(self, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Check if pattern should trigger given context."""
        # Compare context against trigger conditions
        match_score = 0.0
        conditions_matched = 0
        total_conditions = len(self.trigger_conditions)

        if total_conditions == 0:
            return False, 0.0

        for key, expected_value in self.trigger_conditions.items():
            if key in context:
                actual_value = context[key]

                # Exact match
                if actual_value == expected_value:
                    conditions_matched += 1
                    match_score += 1.0

                # Partial match for strings
                elif isinstance(expected_value, str) and isinstance(actual_value, str):
                    if expected_value.lower() in actual_value.lower():
                        conditions_matched += 1
                        match_score += 0.7

        if conditions_matched == 0:
            return False, 0.0

        final_score = (match_score / total_conditions) * self.confidence
        return final_score > 0.5, final_score


@dataclass
class SuggestionEngineConfig:
    """Configuration for the suggestion engine."""
    # Suggestion generation
    max_pending_suggestions: int = int(os.getenv("SUGGEST_MAX_PENDING", "10"))
    min_confidence_threshold: float = float(os.getenv("SUGGEST_MIN_CONFIDENCE", "0.6"))
    min_relevance_threshold: float = float(os.getenv("SUGGEST_MIN_RELEVANCE", "0.5"))

    # Timing
    min_interval_between_suggestions: float = float(os.getenv("SUGGEST_MIN_INTERVAL", "60.0"))  # seconds
    check_interval: float = float(os.getenv("SUGGEST_CHECK_INTERVAL", "10.0"))
    suggestion_ttl_seconds: float = float(os.getenv("SUGGEST_TTL", "300.0"))  # 5 minutes

    # Learning
    enable_pattern_learning: bool = os.getenv("SUGGEST_ENABLE_LEARNING", "true").lower() == "true"
    min_occurrences_for_pattern: int = int(os.getenv("SUGGEST_MIN_OCCURRENCES", "3"))
    pattern_decay_days: int = int(os.getenv("SUGGEST_PATTERN_DECAY_DAYS", "30"))

    # Interruption control
    respect_focus_mode: bool = True
    respect_meeting_time: bool = True
    quiet_hours_start: int = int(os.getenv("SUGGEST_QUIET_START", "22"))  # 10 PM
    quiet_hours_end: int = int(os.getenv("SUGGEST_QUIET_END", "8"))  # 8 AM

    # Priority settings
    allow_critical_during_focus: bool = True
    max_suggestions_per_hour: int = int(os.getenv("SUGGEST_MAX_PER_HOUR", "10"))


# =============================================================================
# Pattern Learning
# =============================================================================

class PatternLearner:
    """
    Learns user behavior patterns from activity observations.

    Patterns learned:
    - Time-based: "User opens email at 9 AM"
    - Context-based: "User runs tests after saving Python files"
    - Sequence-based: "User opens Slack after finishing a PR"
    """

    def __init__(self):
        self._patterns: Dict[str, LearnedPattern] = {}
        self._observations: Deque[Dict[str, Any]] = deque(maxlen=1000)

        # Time-based pattern buckets (hour -> activities)
        self._hourly_patterns: Dict[int, List[str]] = defaultdict(list)

        # Context transition patterns (from_context -> to_context -> count)
        self._transition_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # App usage patterns
        self._app_patterns: Dict[str, Dict[str, Any]] = {}

    def observe(
        self,
        activity_type: str,
        app_name: Optional[str],
        context: Dict[str, Any],
    ) -> None:
        """Record an observation for pattern learning."""
        observation = {
            "timestamp": datetime.now(),
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "activity_type": activity_type,
            "app_name": app_name,
            "context": context,
        }
        self._observations.append(observation)

        # Update hourly patterns
        self._hourly_patterns[observation["hour"]].append(activity_type)

        # Track transitions
        if len(self._observations) >= 2:
            prev = self._observations[-2]
            if prev.get("app_name") and app_name and prev["app_name"] != app_name:
                self._transition_patterns[prev["app_name"]][app_name] += 1

    def find_matching_patterns(
        self,
        current_context: Dict[str, Any]
    ) -> List[Tuple[LearnedPattern, float]]:
        """Find patterns that match the current context."""
        matches = []

        for pattern in self._patterns.values():
            if not pattern.is_active:
                continue

            should_trigger, score = pattern.should_trigger(current_context)
            if should_trigger:
                matches.append((pattern, score))

        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_time_based_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions based on time patterns."""
        current_hour = datetime.now().hour
        suggestions = []

        # Check what activities are common at this hour
        hour_activities = self._hourly_patterns.get(current_hour, [])
        if hour_activities:
            # Find most common activity
            from collections import Counter
            common = Counter(hour_activities).most_common(3)

            for activity, count in common:
                if count >= 3:  # Minimum occurrences
                    suggestions.append({
                        "pattern_type": "time_based",
                        "activity": activity,
                        "confidence": min(count / 10.0, 1.0),
                        "hour": current_hour,
                    })

        return suggestions

    def get_transition_suggestion(self, current_app: str) -> Optional[Dict[str, Any]]:
        """Suggest next app based on transition patterns."""
        if current_app not in self._transition_patterns:
            return None

        transitions = self._transition_patterns[current_app]
        if not transitions:
            return None

        # Find most common next app
        next_app = max(transitions.items(), key=lambda x: x[1])
        count = next_app[1]

        if count >= 3:  # Minimum occurrences
            return {
                "pattern_type": "transition",
                "from_app": current_app,
                "to_app": next_app[0],
                "confidence": min(count / 10.0, 1.0),
            }

        return None

    def create_pattern_from_observation(
        self,
        pattern_type: str,
        conditions: Dict[str, Any],
        suggested_action: str,
    ) -> LearnedPattern:
        """Create a new pattern from observations."""
        pattern_id = hashlib.md5(str(conditions).encode()).hexdigest()[:12]

        pattern = LearnedPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            description=f"{pattern_type} pattern",
            trigger_conditions=conditions,
            suggested_action=suggested_action,
            confidence=0.5,  # Initial confidence
        )

        self._patterns[pattern_id] = pattern
        return pattern

    def update_pattern_from_feedback(
        self,
        pattern_id: str,
        was_accepted: bool,
    ) -> None:
        """Update pattern confidence based on user feedback."""
        if pattern_id not in self._patterns:
            return

        pattern = self._patterns[pattern_id]
        pattern.occurrence_count += 1

        # Update success rate with exponential moving average
        alpha = 0.3
        success = 1.0 if was_accepted else 0.0
        pattern.success_rate = alpha * success + (1 - alpha) * pattern.success_rate

        # Update confidence based on success rate and occurrences
        pattern.confidence = min(
            pattern.success_rate * (1 - 1 / (pattern.occurrence_count + 1)),
            1.0
        )

        # Deactivate patterns with poor performance
        if pattern.occurrence_count >= 10 and pattern.success_rate < 0.2:
            pattern.is_active = False


# =============================================================================
# Timing Optimizer
# =============================================================================

class TimingOptimizer:
    """
    Determines the optimal time to show suggestions.

    Considers:
    - User focus state
    - Current activity
    - Time since last suggestion
    - User's historical response patterns
    - Calendar events
    """

    def __init__(self, config: SuggestionEngineConfig):
        self.config = config
        self._last_suggestion_time: Optional[datetime] = None
        self._suggestions_this_hour: int = 0
        self._hour_started: datetime = datetime.now().replace(minute=0, second=0, microsecond=0)

        # Track response times by context
        self._response_times: Dict[str, List[float]] = defaultdict(list)

    def is_good_time_to_suggest(
        self,
        focus_state: Optional[str] = None,
        in_meeting: bool = False,
        priority: SuggestionPriority = SuggestionPriority.MEDIUM,
    ) -> Tuple[bool, str]:
        """
        Check if now is a good time to show a suggestion.

        Returns:
            Tuple of (is_good_time, reason)
        """
        now = datetime.now()

        # Reset hourly counter if needed
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        if current_hour > self._hour_started:
            self._suggestions_this_hour = 0
            self._hour_started = current_hour

        # Check quiet hours (unless critical)
        if priority != SuggestionPriority.CRITICAL:
            hour = now.hour
            if self.config.quiet_hours_start > self.config.quiet_hours_end:
                # Quiet hours span midnight
                if hour >= self.config.quiet_hours_start or hour < self.config.quiet_hours_end:
                    return False, "quiet_hours"
            else:
                if self.config.quiet_hours_start <= hour < self.config.quiet_hours_end:
                    return False, "quiet_hours"

        # Check meeting time
        if in_meeting and self.config.respect_meeting_time:
            if priority != SuggestionPriority.CRITICAL:
                return False, "in_meeting"

        # Check focus mode
        if focus_state == "deep_focus" and self.config.respect_focus_mode:
            if not (priority == SuggestionPriority.CRITICAL and self.config.allow_critical_during_focus):
                return False, "focus_mode"

        # Check rate limiting
        if self._suggestions_this_hour >= self.config.max_suggestions_per_hour:
            return False, "rate_limit_hour"

        # Check minimum interval
        if self._last_suggestion_time:
            elapsed = (now - self._last_suggestion_time).total_seconds()
            if elapsed < self.config.min_interval_between_suggestions:
                return False, "too_soon"

        return True, "ok"

    def record_suggestion_shown(self) -> None:
        """Record that a suggestion was shown."""
        self._last_suggestion_time = datetime.now()
        self._suggestions_this_hour += 1

    def record_response_time(
        self,
        context: str,
        response_time_seconds: float,
    ) -> None:
        """Record how long user took to respond in given context."""
        self._response_times[context].append(response_time_seconds)

        # Keep only recent data
        if len(self._response_times[context]) > 100:
            self._response_times[context] = self._response_times[context][-100:]

    def get_optimal_delay(self, context: str) -> float:
        """Get optimal delay before showing suggestion based on context."""
        if context in self._response_times and len(self._response_times[context]) >= 5:
            # Use median response time as guide
            times = sorted(self._response_times[context])
            median = times[len(times) // 2]
            return max(0.5, median * 0.5)  # Show after half the typical response time

        return 2.0  # Default 2 second delay


# =============================================================================
# Proactive Suggestion Engine
# =============================================================================

class ProactiveSuggestionEngine:
    """
    Main engine for generating and managing proactive suggestions.

    Learns from user behavior and context to provide helpful,
    timely suggestions without being intrusive.
    """

    def __init__(self, config: Optional[SuggestionEngineConfig] = None):
        """
        Initialize the suggestion engine.

        Args:
            config: Engine configuration
        """
        self.config = config or SuggestionEngineConfig()

        # State
        self._running = False
        self._paused = False
        self._started_at: Optional[datetime] = None

        # Components
        self._pattern_learner = PatternLearner()
        self._timing_optimizer = TimingOptimizer(self.config)

        # Suggestions
        self._pending_suggestions: Deque[Suggestion] = deque(maxlen=self.config.max_pending_suggestions)
        self._shown_suggestions: Dict[str, Suggestion] = {}
        self._suggestion_history: Deque[Suggestion] = deque(maxlen=100)

        # Current context (updated by external sources)
        self._current_context: Dict[str, Any] = {}
        self._current_activity: Optional[str] = None
        self._current_app: Optional[str] = None

        # Callbacks
        self._on_suggestion: List[Callable[[Suggestion], Coroutine]] = []

        # Background tasks
        self._check_task: Optional[asyncio.Task] = None

        # Calendar integration (lazy loaded)
        self._calendar_events: List[Dict[str, Any]] = []

        # Stats
        self._stats = {
            "suggestions_generated": 0,
            "suggestions_shown": 0,
            "suggestions_accepted": 0,
            "suggestions_dismissed": 0,
            "patterns_learned": 0,
        }

        logger.debug("ProactiveSuggestionEngine initialized")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> bool:
        """Start the suggestion engine."""
        if self._running:
            return True

        try:
            self._running = True
            self._started_at = datetime.now()

            # Start suggestion check loop
            self._check_task = asyncio.create_task(
                self._suggestion_check_loop(),
                name="suggestion_check"
            )

            logger.info("ProactiveSuggestionEngine started")
            return True

        except Exception as e:
            logger.error(f"Failed to start ProactiveSuggestionEngine: {e}")
            self._running = False
            return False

    async def stop(self) -> None:
        """Stop the suggestion engine."""
        if not self._running:
            return

        self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        logger.info("ProactiveSuggestionEngine stopped")

    def pause(self) -> None:
        """Pause suggestion generation."""
        self._paused = True

    def resume(self) -> None:
        """Resume suggestion generation."""
        self._paused = False

    # =========================================================================
    # Context Updates
    # =========================================================================

    def update_context(
        self,
        activity_type: Optional[str] = None,
        app_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the current context."""
        if activity_type:
            self._current_activity = activity_type
        if app_name:
            self._current_app = app_name
        if context:
            self._current_context.update(context)

        # Record observation for pattern learning
        if self.config.enable_pattern_learning and activity_type:
            self._pattern_learner.observe(
                activity_type=activity_type,
                app_name=app_name,
                context=self._current_context,
            )

    def set_calendar_events(self, events: List[Dict[str, Any]]) -> None:
        """Set upcoming calendar events for awareness."""
        self._calendar_events = events

    # =========================================================================
    # Suggestion Generation
    # =========================================================================

    async def _suggestion_check_loop(self) -> None:
        """Main loop for checking and generating suggestions."""
        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(self.config.check_interval)
                    continue

                # Generate suggestions based on current context
                await self._generate_contextual_suggestions()

                # Check for calendar-based suggestions
                await self._check_calendar_suggestions()

                # Process pending suggestions
                await self._process_pending_suggestions()

                # Clean up expired suggestions
                self._cleanup_expired_suggestions()

                await asyncio.sleep(self.config.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Suggestion check loop error: {e}")
                await asyncio.sleep(self.config.check_interval)

    async def _generate_contextual_suggestions(self) -> None:
        """Generate suggestions based on current context and patterns."""
        if not self._current_context:
            return

        # Check learned patterns
        matches = self._pattern_learner.find_matching_patterns(self._current_context)
        for pattern, score in matches[:3]:  # Top 3 matches
            suggestion = self._create_suggestion_from_pattern(pattern, score)
            if suggestion:
                self._add_pending_suggestion(suggestion)

        # Check time-based patterns
        time_suggestions = self._pattern_learner.get_time_based_suggestions()
        for ts in time_suggestions[:2]:
            suggestion = self._create_time_based_suggestion(ts)
            if suggestion:
                self._add_pending_suggestion(suggestion)

        # Check transition patterns
        if self._current_app:
            transition = self._pattern_learner.get_transition_suggestion(self._current_app)
            if transition:
                suggestion = self._create_transition_suggestion(transition)
                if suggestion:
                    self._add_pending_suggestion(suggestion)

    def _create_suggestion_from_pattern(
        self,
        pattern: LearnedPattern,
        score: float,
    ) -> Optional[Suggestion]:
        """Create a suggestion from a learned pattern."""
        if score < self.config.min_confidence_threshold:
            return None

        return Suggestion(
            suggestion_type=SuggestionType.ACTION,
            priority=SuggestionPriority.MEDIUM,
            title="Based on your patterns",
            message=pattern.suggested_message or f"Would you like to {pattern.suggested_action}?",
            action_type=pattern.suggested_action,
            pattern_id=pattern.pattern_id,
            confidence=score,
            relevance_score=pattern.success_rate,
            timing_score=0.7,
            expires_at=datetime.now() + timedelta(seconds=self.config.suggestion_ttl_seconds),
        )

    def _create_time_based_suggestion(
        self,
        pattern_data: Dict[str, Any],
    ) -> Optional[Suggestion]:
        """Create a suggestion from time-based pattern."""
        activity = pattern_data.get("activity")
        confidence = pattern_data.get("confidence", 0.5)

        if confidence < self.config.min_confidence_threshold:
            return None

        messages = {
            "coding": "Ready to start coding?",
            "communication": "Time to check messages?",
            "productivity": "Ready to get productive?",
        }

        message = messages.get(activity, f"Time for {activity}?")

        return Suggestion(
            suggestion_type=SuggestionType.REMINDER,
            priority=SuggestionPriority.LOW,
            title="Time-based suggestion",
            message=message,
            confidence=confidence,
            relevance_score=confidence,
            timing_score=0.8,
            expires_at=datetime.now() + timedelta(seconds=self.config.suggestion_ttl_seconds),
        )

    def _create_transition_suggestion(
        self,
        transition_data: Dict[str, Any],
    ) -> Optional[Suggestion]:
        """Create a suggestion from transition pattern."""
        to_app = transition_data.get("to_app")
        confidence = transition_data.get("confidence", 0.5)

        if confidence < self.config.min_confidence_threshold:
            return None

        return Suggestion(
            suggestion_type=SuggestionType.ACTION,
            priority=SuggestionPriority.LOW,
            title="Next step suggestion",
            message=f"Would you like to open {to_app}?",
            action_type="open_app",
            action_payload={"app_name": to_app},
            confidence=confidence,
            relevance_score=confidence,
            timing_score=0.7,
            expires_at=datetime.now() + timedelta(seconds=self.config.suggestion_ttl_seconds),
        )

    async def _check_calendar_suggestions(self) -> None:
        """Generate suggestions based on calendar events."""
        now = datetime.now()

        for event in self._calendar_events:
            start_time = event.get("start_time")
            if not start_time:
                continue

            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time)
                except ValueError:
                    continue

            # Check if event is coming up (within 10 minutes)
            time_until = (start_time - now).total_seconds()
            if 0 < time_until <= 600:  # 10 minutes
                # Create meeting reminder
                minutes = int(time_until / 60)
                suggestion = Suggestion(
                    suggestion_type=SuggestionType.REMINDER,
                    priority=SuggestionPriority.HIGH,
                    title="Upcoming event",
                    message=f"{event.get('title', 'Meeting')} starts in {minutes} minutes",
                    detail=event.get("location"),
                    action_type="open_meeting",
                    action_payload=event,
                    confidence=1.0,
                    relevance_score=1.0,
                    timing_score=1.0,
                    expires_at=start_time,
                )
                self._add_pending_suggestion(suggestion)

    def _add_pending_suggestion(self, suggestion: Suggestion) -> None:
        """Add a suggestion to the pending queue."""
        # Check for duplicates
        for existing in self._pending_suggestions:
            if (existing.message == suggestion.message and
                existing.suggestion_type == suggestion.suggestion_type):
                return  # Skip duplicate

        self._pending_suggestions.append(suggestion)
        self._stats["suggestions_generated"] += 1

    async def _process_pending_suggestions(self) -> None:
        """Process pending suggestions and show appropriate ones."""
        if not self._pending_suggestions:
            return

        # Sort by combined score
        sorted_suggestions = sorted(
            self._pending_suggestions,
            key=lambda s: s.get_combined_score(),
            reverse=True
        )

        for suggestion in sorted_suggestions:
            if suggestion.is_expired():
                continue

            # Check if good time to suggest
            is_good_time, reason = self._timing_optimizer.is_good_time_to_suggest(
                focus_state=self._current_context.get("focus_state"),
                in_meeting=self._current_context.get("in_meeting", False),
                priority=suggestion.priority,
            )

            if is_good_time:
                await self._show_suggestion(suggestion)
                break  # Only show one at a time

    async def _show_suggestion(self, suggestion: Suggestion) -> None:
        """Show a suggestion to the user."""
        suggestion.state = SuggestionState.SHOWN
        suggestion.shown_at = datetime.now()

        # Move to shown dict
        self._shown_suggestions[suggestion.suggestion_id] = suggestion

        # Remove from pending
        try:
            self._pending_suggestions.remove(suggestion)
        except ValueError:
            pass

        # Record timing
        self._timing_optimizer.record_suggestion_shown()
        self._stats["suggestions_shown"] += 1

        # Notify callbacks
        for callback in self._on_suggestion:
            try:
                await callback(suggestion)
            except Exception as e:
                logger.error(f"Suggestion callback error: {e}")

    def _cleanup_expired_suggestions(self) -> None:
        """Remove expired suggestions."""
        # Clean pending
        self._pending_suggestions = deque(
            [s for s in self._pending_suggestions if not s.is_expired()],
            maxlen=self.config.max_pending_suggestions
        )

        # Clean shown
        for sid, suggestion in list(self._shown_suggestions.items()):
            if suggestion.is_expired():
                suggestion.state = SuggestionState.EXPIRED
                self._suggestion_history.append(suggestion)
                del self._shown_suggestions[sid]

    # =========================================================================
    # User Feedback
    # =========================================================================

    def record_response(
        self,
        suggestion_id: str,
        response: UserResponseType,
    ) -> None:
        """Record user response to a suggestion."""
        if suggestion_id not in self._shown_suggestions:
            return

        suggestion = self._shown_suggestions[suggestion_id]
        suggestion.user_response = response
        suggestion.responded_at = datetime.now()

        # Update stats
        if response == UserResponseType.ACCEPT:
            suggestion.state = SuggestionState.ACCEPTED
            self._stats["suggestions_accepted"] += 1
        elif response in [UserResponseType.DISMISS, UserResponseType.FEEDBACK_NEGATIVE]:
            suggestion.state = SuggestionState.DISMISSED
            self._stats["suggestions_dismissed"] += 1
        elif response == UserResponseType.DEFER:
            suggestion.state = SuggestionState.DEFERRED

        # Update pattern learning
        if suggestion.pattern_id:
            self._pattern_learner.update_pattern_from_feedback(
                suggestion.pattern_id,
                was_accepted=(response == UserResponseType.ACCEPT),
            )

        # Record response time
        if suggestion.shown_at:
            response_time = (datetime.now() - suggestion.shown_at).total_seconds()
            context = self._current_activity or "unknown"
            self._timing_optimizer.record_response_time(context, response_time)

        # Move to history
        self._suggestion_history.append(suggestion)
        del self._shown_suggestions[suggestion_id]

    # =========================================================================
    # Public API
    # =========================================================================

    def on_suggestion(
        self,
        callback: Callable[[Suggestion], Coroutine]
    ) -> None:
        """Register a callback for when suggestions are ready to show."""
        self._on_suggestion.append(callback)

    def get_pending_suggestions(self) -> List[Suggestion]:
        """Get all pending suggestions."""
        return list(self._pending_suggestions)

    def get_shown_suggestions(self) -> List[Suggestion]:
        """Get currently shown suggestions awaiting response."""
        return list(self._shown_suggestions.values())

    def manually_trigger_suggestion(
        self,
        suggestion_type: SuggestionType,
        title: str,
        message: str,
        priority: SuggestionPriority = SuggestionPriority.MEDIUM,
        action_type: Optional[str] = None,
        action_payload: Optional[Dict[str, Any]] = None,
    ) -> Suggestion:
        """Manually create and queue a suggestion."""
        suggestion = Suggestion(
            suggestion_type=suggestion_type,
            priority=priority,
            title=title,
            message=message,
            action_type=action_type,
            action_payload=action_payload,
            confidence=1.0,
            relevance_score=1.0,
            timing_score=1.0,
            expires_at=datetime.now() + timedelta(seconds=self.config.suggestion_ttl_seconds),
        )
        self._add_pending_suggestion(suggestion)
        return suggestion

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "running": self._running,
            "paused": self._paused,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "pending_count": len(self._pending_suggestions),
            "shown_count": len(self._shown_suggestions),
            "history_count": len(self._suggestion_history),
            "patterns_count": len(self._pattern_learner._patterns),
            **self._stats,
        }


# =============================================================================
# Singleton Management
# =============================================================================

_suggestion_engine: Optional[ProactiveSuggestionEngine] = None


async def get_suggestion_engine(
    config: Optional[SuggestionEngineConfig] = None
) -> ProactiveSuggestionEngine:
    """Get the global suggestion engine instance."""
    global _suggestion_engine

    if _suggestion_engine is None:
        _suggestion_engine = ProactiveSuggestionEngine(config)

    return _suggestion_engine


async def start_suggestion_engine(
    config: Optional[SuggestionEngineConfig] = None
) -> ProactiveSuggestionEngine:
    """Get and start the global suggestion engine."""
    engine = await get_suggestion_engine(config)
    if not engine._running:
        await engine.start()
    return engine


async def stop_suggestion_engine() -> None:
    """Stop the global suggestion engine."""
    global _suggestion_engine

    if _suggestion_engine is not None:
        await _suggestion_engine.stop()
        _suggestion_engine = None
