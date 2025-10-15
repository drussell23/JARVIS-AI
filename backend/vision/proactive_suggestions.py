"""
Proactive Suggestion System for JARVIS Vision
Anticipates user needs and suggests actions before being asked.

Examples:
- "I notice Space 2 has an error - want me to analyze it?"
- "You've been working for 2 hours - want a workspace summary?"
- "Space 3 has been inactive - should I check on it?"
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class SuggestionType(Enum):
    """Types of proactive suggestions"""
    ERROR_DETECTED = "error_detected"  # Error/warning on a space
    INACTIVE_SPACE = "inactive_space"  # Space hasn't been used in a while
    WORK_SESSION = "work_session"  # Long work session, suggest summary
    CONTEXT_SWITCH = "context_switch"  # User switched spaces frequently
    PATTERN_BREAK = "pattern_break"  # User broke their usual pattern
    OPPORTUNITY = "opportunity"  # General opportunity for analysis


class SuggestionPriority(Enum):
    """Priority levels for suggestions"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class ProactiveSuggestion:
    """A proactive suggestion for the user"""
    suggestion_id: str
    type: SuggestionType
    priority: SuggestionPriority
    message: str  # What to suggest to the user
    action: str  # What action to take if accepted
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # Tracking
    shown_to_user: bool = False
    user_accepted: bool = False
    user_dismissed: bool = False


class ProactiveSuggestionSystem:
    """
    Generates and manages proactive suggestions.
    Learns which suggestions users find helpful.
    """

    def __init__(self):
        """Initialize proactive suggestion system"""
        # Suggestion queue
        self._active_suggestions: List[ProactiveSuggestion] = []
        self._suggestion_history: List[ProactiveSuggestion] = []

        # Configuration
        self.max_active_suggestions = 3
        self.suggestion_cooldown = timedelta(minutes=15)  # Wait 15min between suggestions
        self.last_suggestion_time: Optional[datetime] = None

        # Learning
        self._suggestion_acceptance_rate: Dict[SuggestionType, float] = {}
        self._suggestion_counts: Dict[SuggestionType, int] = {}

        # Detection state
        self._last_space_activity: Dict[int, datetime] = {}
        self._work_session_start: Optional[datetime] = None
        self._space_switch_count = 0
        self._last_space_switch_time: Optional[datetime] = None

        logger.info("[PROACTIVE] Proactive suggestion system initialized")

    async def analyze_and_suggest(
        self,
        current_context: Dict[str, Any],
        yabai_data: Optional[Dict[str, Any]] = None
    ) -> Optional[ProactiveSuggestion]:
        """
        Analyze current state and generate proactive suggestions

        Args:
            current_context: Current query context from context manager
            yabai_data: Optional Yabai workspace data

        Returns:
            ProactiveSuggestion if one should be shown, None otherwise
        """
        # Check cooldown
        if not self._should_suggest():
            return None

        # Generate suggestions based on various signals
        suggestions = []

        # 1. Error detection (from Yabai or screen analysis)
        if yabai_data:
            error_suggestion = self._detect_errors(yabai_data, current_context)
            if error_suggestion:
                suggestions.append(error_suggestion)

        # 2. Inactive space detection
        inactive_suggestion = self._detect_inactive_spaces(yabai_data, current_context)
        if inactive_suggestion:
            suggestions.append(inactive_suggestion)

        # 3. Work session summary
        work_session_suggestion = self._detect_work_session(current_context)
        if work_session_suggestion:
            suggestions.append(work_session_suggestion)

        # 4. Context switching pattern
        context_switch_suggestion = self._detect_context_switching(current_context)
        if context_switch_suggestion:
            suggestions.append(context_switch_suggestion)

        # 5. Pattern break detection
        pattern_break_suggestion = self._detect_pattern_break(current_context)
        if pattern_break_suggestion:
            suggestions.append(pattern_break_suggestion)

        # Select best suggestion
        if not suggestions:
            return None

        # Sort by priority and select highest
        suggestions.sort(key=lambda s: s.priority.value, reverse=True)
        best_suggestion = suggestions[0]

        # Add to active suggestions
        self._active_suggestions.append(best_suggestion)
        self.last_suggestion_time = datetime.now()

        logger.info(
            f"[PROACTIVE] Generated suggestion: {best_suggestion.type.value} "
            f"(priority: {best_suggestion.priority.value})"
        )

        return best_suggestion

    def _should_suggest(self) -> bool:
        """Check if we should show a suggestion now"""
        # Too many active suggestions
        if len(self._active_suggestions) >= self.max_active_suggestions:
            return False

        # Recent suggestion shown
        if self.last_suggestion_time:
            time_since = datetime.now() - self.last_suggestion_time
            if time_since < self.suggestion_cooldown:
                return False

        return True

    def _detect_errors(
        self,
        yabai_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[ProactiveSuggestion]:
        """Detect errors or warnings in workspace"""
        # This would analyze Yabai data or recent screen captures
        # for error indicators (red text, error dialogs, etc.)

        # Placeholder logic
        spaces = yabai_data.get('spaces', {})
        for space_id, space_info in spaces.items():
            # Check for error keywords in window titles
            if isinstance(space_info, dict):
                apps = space_info.get('applications', [])
                if any('error' in str(app).lower() or 'warning' in str(app).lower() for app in apps):
                    return ProactiveSuggestion(
                        suggestion_id=f"error_{space_id}_{int(time.time())}",
                        type=SuggestionType.ERROR_DETECTED,
                        priority=SuggestionPriority.HIGH,
                        message=f"I notice there might be an error in Space {space_id}. Would you like me to analyze it?",
                        action=f"analyze_space_{space_id}",
                        context={'space_id': space_id, 'apps': apps},
                        expires_at=datetime.now() + timedelta(minutes=30)
                    )

        return None

    def _detect_inactive_spaces(
        self,
        yabai_data: Optional[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Optional[ProactiveSuggestion]:
        """Detect spaces that haven't been used in a while"""
        if not yabai_data:
            return None

        current_time = datetime.now()
        active_space = context.get('active_space')

        # Track space activity
        if active_space:
            self._last_space_activity[active_space] = current_time

        # Find inactive spaces
        spaces = yabai_data.get('spaces', {})
        for space_id in spaces.keys():
            if space_id == active_space:
                continue

            last_activity = self._last_space_activity.get(space_id)
            if last_activity:
                inactive_duration = current_time - last_activity
                # If space inactive for >1 hour
                if inactive_duration > timedelta(hours=1):
                    return ProactiveSuggestion(
                        suggestion_id=f"inactive_{space_id}_{int(time.time())}",
                        type=SuggestionType.INACTIVE_SPACE,
                        priority=SuggestionPriority.LOW,
                        message=f"Space {space_id} has been inactive for {inactive_duration.seconds//3600} hours. Want me to check on it?",
                        action=f"check_space_{space_id}",
                        context={'space_id': space_id, 'inactive_duration': inactive_duration.seconds},
                        expires_at=datetime.now() + timedelta(hours=2)
                    )

        return None

    def _detect_work_session(
        self,
        context: Dict[str, Any]
    ) -> Optional[ProactiveSuggestion]:
        """Detect long work sessions and suggest summaries"""
        session_duration = context.get('session_duration_minutes', 0)

        # Track work session
        if not self._work_session_start:
            self._work_session_start = datetime.now()

        # If working for >2 hours
        if session_duration > 120:
            return ProactiveSuggestion(
                suggestion_id=f"work_session_{int(time.time())}",
                type=SuggestionType.WORK_SESSION,
                priority=SuggestionPriority.MEDIUM,
                message=f"You've been working for {int(session_duration/60)} hours. Would you like a workspace summary?",
                action="workspace_summary",
                context={'session_duration_minutes': session_duration},
                expires_at=datetime.now() + timedelta(minutes=30)
            )

        return None

    def _detect_context_switching(
        self,
        context: Dict[str, Any]
    ) -> Optional[ProactiveSuggestion]:
        """Detect frequent context switching"""
        # Track space switches
        if context.get('active_space'):
            current_time = datetime.now()
            if self._last_space_switch_time:
                time_since_switch = current_time - self._last_space_switch_time
                # If switched within 5 minutes
                if time_since_switch < timedelta(minutes=5):
                    self._space_switch_count += 1
                else:
                    self._space_switch_count = 0

            self._last_space_switch_time = current_time

            # If switched >3 times in 5 minutes
            if self._space_switch_count > 3:
                return ProactiveSuggestion(
                    suggestion_id=f"context_switch_{int(time.time())}",
                    type=SuggestionType.CONTEXT_SWITCH,
                    priority=SuggestionPriority.MEDIUM,
                    message="I notice you're switching between spaces frequently. Want me to analyze your workflow?",
                    action="analyze_workflow",
                    context={'switch_count': self._space_switch_count},
                    expires_at=datetime.now() + timedelta(minutes=15)
                )

        return None

    def _detect_pattern_break(
        self,
        context: Dict[str, Any]
    ) -> Optional[ProactiveSuggestion]:
        """Detect when user breaks their usual pattern"""
        detected_pattern = context.get('user_pattern', 'unknown')
        time_of_day = context.get('time_of_day', 'unknown')

        # Example: If user is a "morning_overview" type but hasn't asked for overview
        if detected_pattern == 'morning_overview' and time_of_day == 'morning':
            recent_intents = context.get('recent_intents', [])
            if 'metadata_only' not in recent_intents and 'deep_analysis' not in recent_intents:
                return ProactiveSuggestion(
                    suggestion_id=f"pattern_break_{int(time.time())}",
                    type=SuggestionType.PATTERN_BREAK,
                    priority=SuggestionPriority.LOW,
                    message="Good morning, Sir. Would you like your usual workspace overview?",
                    action="workspace_overview",
                    context={'pattern': detected_pattern, 'time': time_of_day},
                    expires_at=datetime.now() + timedelta(hours=1)
                )

        return None

    async def record_user_response(
        self,
        suggestion_id: str,
        accepted: bool
    ):
        """
        Record user's response to a suggestion

        Args:
            suggestion_id: ID of the suggestion
            accepted: Whether user accepted or dismissed
        """
        # Find suggestion
        suggestion = None
        for s in self._active_suggestions:
            if s.suggestion_id == suggestion_id:
                suggestion = s
                break

        if not suggestion:
            logger.warning(f"[PROACTIVE] Unknown suggestion: {suggestion_id}")
            return

        # Update suggestion
        suggestion.user_accepted = accepted
        suggestion.user_dismissed = not accepted

        # Move to history
        self._active_suggestions.remove(suggestion)
        self._suggestion_history.append(suggestion)

        # Update learning metrics
        if suggestion.type not in self._suggestion_counts:
            self._suggestion_counts[suggestion.type] = 0
            self._suggestion_acceptance_rate[suggestion.type] = 0.0

        self._suggestion_counts[suggestion.type] += 1

        # Update acceptance rate (exponential moving average)
        current_rate = self._suggestion_acceptance_rate[suggestion.type]
        new_rate = current_rate * 0.9 + (1.0 if accepted else 0.0) * 0.1
        self._suggestion_acceptance_rate[suggestion.type] = new_rate

        logger.info(
            f"[PROACTIVE] User {'accepted' if accepted else 'dismissed'} suggestion: "
            f"{suggestion.type.value} (acceptance rate: {new_rate:.1%})"
        )

    def get_active_suggestions(self) -> List[ProactiveSuggestion]:
        """Get all active suggestions"""
        # Remove expired suggestions
        current_time = datetime.now()
        self._active_suggestions = [
            s for s in self._active_suggestions
            if not s.expires_at or s.expires_at > current_time
        ]
        return self._active_suggestions.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get suggestion statistics"""
        return {
            'active_suggestions': len(self._active_suggestions),
            'total_suggestions_generated': len(self._suggestion_history),
            'acceptance_rates': {
                stype.value: self._suggestion_acceptance_rate.get(stype, 0.0)
                for stype in SuggestionType
            },
            'suggestion_counts': {
                stype.value: self._suggestion_counts.get(stype, 0)
                for stype in SuggestionType
            },
            'last_suggestion_time': (
                self.last_suggestion_time.isoformat()
                if self.last_suggestion_time
                else None
            )
        }

    def clear_suggestions(self):
        """Clear all active suggestions"""
        self._active_suggestions.clear()
        logger.info("[PROACTIVE] All suggestions cleared")


# Singleton instance
_proactive_system: Optional[ProactiveSuggestionSystem] = None


def get_proactive_system() -> ProactiveSuggestionSystem:
    """Get or create the singleton proactive system"""
    global _proactive_system

    if _proactive_system is None:
        _proactive_system = ProactiveSuggestionSystem()

    return _proactive_system
