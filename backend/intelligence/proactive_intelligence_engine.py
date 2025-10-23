#!/usr/bin/env python3
"""
Proactive Intelligence Engine - Phase 4
========================================

Advanced proactive communication system that uses behavioral learning
to provide natural, context-aware suggestions and automation.

Features:
- Natural language suggestions based on learned patterns
- Workflow optimization analysis and recommendations
- Predictive app launching with user confirmation
- Smart Space switching based on context
- Human-like proactive communication
- Adaptive timing based on user focus
- Multi-modal communication (voice + visual)

Author: Derek J. Russell
Date: October 2025
Version: 4.0.0 - Proactive Intelligence with Behavioral Learning
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import random

logger = logging.getLogger(__name__)


# ============================================================================
# Communication Enums and Data Models
# ============================================================================

class SuggestionType(Enum):
    """Types of proactive suggestions"""
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    PREDICTIVE_APP_LAUNCH = "predictive_app_launch"
    SMART_SPACE_SWITCH = "smart_space_switch"
    PATTERN_REMINDER = "pattern_reminder"
    EFFICIENCY_TIP = "efficiency_tip"
    BEHAVIORAL_INSIGHT = "behavioral_insight"
    AUTOMATION_OPPORTUNITY = "automation_opportunity"


class CommunicationPriority(Enum):
    """Priority levels for proactive communication"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class UserFocusLevel(Enum):
    """Inferred user focus level"""
    DEEP_WORK = "deep_work"          # Don't interrupt
    FOCUSED = "focused"              # Minimal interruptions
    CASUAL = "casual"                # Normal communication
    IDLE = "idle"                    # Can communicate freely


@dataclass
class ProactiveSuggestion:
    """A proactive suggestion to present to the user"""
    suggestion_id: str
    suggestion_type: SuggestionType
    priority: CommunicationPriority
    confidence: float

    # Content
    title: str
    message: str
    reasoning: str

    # Action
    action_type: str
    action_params: Dict[str, Any]
    requires_confirmation: bool = True

    # Context
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)

    # User interaction
    presented: bool = False
    user_response: Optional[str] = None  # 'accepted', 'rejected', 'ignored'
    response_time: Optional[float] = None


@dataclass
class CommunicationContext:
    """Current context for communication decisions"""
    user_focus_level: UserFocusLevel
    current_app: Optional[str]
    current_space: Optional[int]
    recent_activity: List[str]
    time_since_last_message: float
    pending_suggestions_count: int
    user_is_idle: bool = False


# ============================================================================
# Proactive Intelligence Engine
# ============================================================================

class ProactiveIntelligenceEngine:
    """
    Advanced proactive communication engine powered by behavioral learning

    Integrates with:
    - Learning Database (behavioral patterns)
    - Pattern Learner (ML predictions)
    - Yabai (spatial context)
    - UAE (decision fusion)
    """

    def __init__(
        self,
        learning_db=None,
        pattern_learner=None,
        yabai_intelligence=None,
        uae_engine=None,
        voice_callback: Optional[Callable] = None,
        notification_callback: Optional[Callable] = None
    ):
        """
        Initialize Proactive Intelligence Engine

        Args:
            learning_db: Learning Database instance
            pattern_learner: Workspace Pattern Learner instance
            yabai_intelligence: Yabai Spatial Intelligence instance
            uae_engine: UAE engine instance
            voice_callback: Async function for voice output
            notification_callback: Async function for notifications
        """
        self.learning_db = learning_db
        self.pattern_learner = pattern_learner
        self.yabai = yabai_intelligence
        self.uae = uae_engine
        self.voice_callback = voice_callback
        self.notification_callback = notification_callback

        # Communication state
        self.is_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.communication_interval = 30.0  # Check every 30 seconds

        # Suggestion management
        self.pending_suggestions: deque = deque(maxlen=20)
        self.suggestion_history: deque = deque(maxlen=100)
        self.last_suggestion_time: Optional[float] = None
        self.min_suggestion_interval = 300.0  # 5 minutes minimum between suggestions

        # Context tracking
        self.current_context = CommunicationContext(
            user_focus_level=UserFocusLevel.CASUAL,
            current_app=None,
            current_space=None,
            recent_activity=[],
            time_since_last_message=float('inf'),
            pending_suggestions_count=0
        )

        # User preferences (learned over time)
        self.user_preferences = {
            'voice_enabled': True,
            'interruption_threshold': 0.7,  # Min confidence for interruption
            'quiet_hours_start': 22,  # 10 PM
            'quiet_hours_end': 8,     # 8 AM
            'personality_level': 0.8,  # How casual/human-like to be
            'max_suggestions_per_hour': 6
        }

        # Natural language templates
        self.message_templates = self._initialize_templates()

        # Statistics
        self.stats = {
            'suggestions_generated': 0,
            'suggestions_presented': 0,
            'suggestions_accepted': 0,
            'suggestions_rejected': 0,
            'suggestions_ignored': 0,
            'voice_messages_sent': 0,
            'notifications_sent': 0
        }

        logger.info("[PIE] Proactive Intelligence Engine initialized")
        logger.info(f"[PIE] Voice enabled: {voice_callback is not None}")
        logger.info(f"[PIE] Notifications enabled: {notification_callback is not None}")

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def start(self):
        """Start proactive intelligence monitoring"""
        if self.is_active:
            logger.warning("[PIE] Already active")
            return

        logger.info("[PIE] Starting proactive intelligence engine...")
        self.is_active = True

        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._proactive_monitoring_loop())

        logger.info("[PIE] ✅ Proactive intelligence active")
        logger.info("[PIE] 📢 Ready to communicate naturally based on learned patterns")

    async def stop(self):
        """Stop proactive intelligence"""
        if not self.is_active:
            return

        logger.info("[PIE] Stopping proactive intelligence...")
        self.is_active = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("[PIE] ✅ Proactive intelligence stopped")
        logger.info(f"[PIE] Total suggestions generated: {self.stats['suggestions_generated']}")
        logger.info(f"[PIE] Acceptance rate: {self._calculate_acceptance_rate():.1%}")

    # ========================================================================
    # Proactive Monitoring Loop
    # ========================================================================

    async def _proactive_monitoring_loop(self):
        """Main monitoring loop for proactive suggestions"""
        logger.info("[PIE] Proactive monitoring loop started")

        while self.is_active:
            try:
                # Update context
                await self._update_context()

                # Check if it's appropriate to communicate
                if not self._should_communicate():
                    await asyncio.sleep(self.communication_interval)
                    continue

                # Generate suggestions based on patterns
                suggestions = await self._generate_proactive_suggestions()

                # Filter and prioritize suggestions
                suggestions = self._filter_suggestions(suggestions)

                # Present suggestions
                if suggestions:
                    await self._present_suggestions(suggestions)

                # Sleep until next check
                await asyncio.sleep(self.communication_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[PIE] Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(self.communication_interval)

    async def _update_context(self):
        """Update current communication context"""
        try:
            # Get current spatial context from Yabai
            if self.yabai and self.yabai.yabai_available:
                focused_space = self.yabai.current_focused_space
                if focused_space and focused_space in self.yabai.current_spaces:
                    space = self.yabai.current_spaces[focused_space]
                    if space.focused_window:
                        self.current_context.current_app = space.focused_window.app_name
                        self.current_context.current_space = focused_space

            # Infer user focus level
            self.current_context.user_focus_level = await self._infer_focus_level()

            # Update time since last message
            if self.last_suggestion_time:
                self.current_context.time_since_last_message = time.time() - self.last_suggestion_time

            # Update pending count
            self.current_context.pending_suggestions_count = len(self.pending_suggestions)

        except Exception as e:
            logger.error(f"[PIE] Error updating context: {e}", exc_info=True)

    async def _infer_focus_level(self) -> UserFocusLevel:
        """Infer user's current focus level from activity"""
        try:
            # Check if in quiet hours
            now = datetime.now()
            if self.user_preferences['quiet_hours_start'] <= now.hour or now.hour < self.user_preferences['quiet_hours_end']:
                return UserFocusLevel.DEEP_WORK

            # Check current app type
            if self.current_context.current_app:
                focus_apps = ['xcode', 'pycharm', 'vscode', 'terminal', 'iterm', 'slack']
                if any(app in self.current_context.current_app.lower() for app in focus_apps):
                    return UserFocusLevel.FOCUSED

            # Check activity level from Yabai
            if self.yabai:
                recent_space_changes = self.yabai.metrics.get('total_space_changes', 0)
                if recent_space_changes > 10:  # Active switching
                    return UserFocusLevel.CASUAL

            # Default to casual
            return UserFocusLevel.CASUAL

        except Exception as e:
            logger.error(f"[PIE] Error inferring focus level: {e}", exc_info=True)
            return UserFocusLevel.CASUAL

    def _should_communicate(self) -> bool:
        """Determine if it's appropriate to communicate now"""
        # Check minimum time interval
        if self.last_suggestion_time:
            time_since_last = time.time() - self.last_suggestion_time
            if time_since_last < self.min_suggestion_interval:
                return False

        # Check focus level
        if self.current_context.user_focus_level == UserFocusLevel.DEEP_WORK:
            return False

        # Check max suggestions per hour
        recent_suggestions = [
            s for s in self.suggestion_history
            if time.time() - s.created_at < 3600
        ]
        if len(recent_suggestions) >= self.user_preferences['max_suggestions_per_hour']:
            return False

        return True

    # ========================================================================
    # Suggestion Generation (Pattern-Based)
    # ========================================================================

    async def _generate_proactive_suggestions(self) -> List[ProactiveSuggestion]:
        """Generate suggestions based on learned behavioral patterns"""
        suggestions = []

        try:
            # Get behavioral insights from Learning DB
            if self.learning_db:
                insights = await self.learning_db.get_behavioral_insights()

                # Generate workflow optimization suggestions
                workflow_suggestions = await self._generate_workflow_suggestions(insights)
                suggestions.extend(workflow_suggestions)

                # Generate predictive app launch suggestions
                predictive_suggestions = await self._generate_predictive_suggestions(insights)
                suggestions.extend(predictive_suggestions)

                # Generate smart space switching suggestions
                space_suggestions = await self._generate_space_suggestions(insights)
                suggestions.extend(space_suggestions)

                # Generate pattern-based reminders
                reminder_suggestions = await self._generate_pattern_reminders(insights)
                suggestions.extend(reminder_suggestions)

            # Get ML predictions from Pattern Learner
            if self.pattern_learner:
                ml_suggestions = await self._generate_ml_predictions()
                suggestions.extend(ml_suggestions)

            self.stats['suggestions_generated'] += len(suggestions)

        except Exception as e:
            logger.error(f"[PIE] Error generating suggestions: {e}", exc_info=True)

        return suggestions

    async def _generate_workflow_suggestions(self, insights: Dict[str, Any]) -> List[ProactiveSuggestion]:
        """Generate workflow optimization suggestions"""
        suggestions = []

        try:
            # Analyze common workflows
            workflows = insights.get('common_workflows', [])

            for workflow in workflows[:3]:  # Top 3 workflows
                if workflow['success_rate'] < 0.8:
                    # Suggest workflow improvement
                    suggestion = ProactiveSuggestion(
                        suggestion_id=f"workflow_opt_{workflow['name']}_{int(time.time())}",
                        suggestion_type=SuggestionType.WORKFLOW_OPTIMIZATION,
                        priority=CommunicationPriority.MEDIUM,
                        confidence=workflow['confidence'],
                        title="Workflow Optimization",
                        message=self._generate_workflow_message(workflow),
                        reasoning=f"Your {workflow['name']} workflow has a {workflow['success_rate']:.0%} success rate. I can help optimize it.",
                        action_type="optimize_workflow",
                        action_params={'workflow_name': workflow['name']},
                        context={'workflow': workflow}
                    )
                    suggestions.append(suggestion)

        except Exception as e:
            logger.error(f"[PIE] Error generating workflow suggestions: {e}", exc_info=True)

        return suggestions

    async def _generate_predictive_suggestions(self, insights: Dict[str, Any]) -> List[ProactiveSuggestion]:
        """Generate predictive app launch suggestions"""
        suggestions = []

        try:
            # Get current context for predictions
            if not self.learning_db:
                return suggestions

            current_context = {
                'current_app': self.current_context.current_app,
                'current_space': self.current_context.current_space,
                'time_of_day': datetime.now().hour
            }

            predictions = await self.learning_db.predict_next_action(current_context)

            for prediction in predictions[:2]:  # Top 2 predictions
                if prediction['confidence'] >= 0.7 and prediction['action_type'] == 'switch_app':
                    suggestion = ProactiveSuggestion(
                        suggestion_id=f"predict_app_{prediction['target']}_{int(time.time())}",
                        suggestion_type=SuggestionType.PREDICTIVE_APP_LAUNCH,
                        priority=CommunicationPriority.LOW,
                        confidence=prediction['confidence'],
                        title="Ready to work?",
                        message=self._generate_predictive_message(prediction),
                        reasoning=prediction['reasoning'],
                        action_type="launch_app",
                        action_params={'app_name': prediction['target']},
                        context={'prediction': prediction}
                    )
                    suggestions.append(suggestion)

        except Exception as e:
            logger.error(f"[PIE] Error generating predictive suggestions: {e}", exc_info=True)

        return suggestions

    async def _generate_space_suggestions(self, insights: Dict[str, Any]) -> List[ProactiveSuggestion]:
        """Generate smart space switching suggestions"""
        suggestions = []

        try:
            # Analyze space transitions
            transitions = insights.get('space_transitions', [])

            # Find common transitions from current space
            if self.current_context.current_space:
                for trans in transitions:
                    if trans['from'] == self.current_context.current_space and trans['frequency'] > 5:
                        suggestion = ProactiveSuggestion(
                            suggestion_id=f"space_switch_{trans['to']}_{int(time.time())}",
                            suggestion_type=SuggestionType.SMART_SPACE_SWITCH,
                            priority=CommunicationPriority.LOW,
                            confidence=0.6,
                            title="Switch to another Space?",
                            message=self._generate_space_switch_message(trans),
                            reasoning=f"You usually switch to Space {trans['to']} around this time",
                            action_type="switch_space",
                            action_params={'target_space': trans['to']},
                            context={'transition': trans}
                        )
                        suggestions.append(suggestion)
                        break  # Only one space suggestion at a time

        except Exception as e:
            logger.error(f"[PIE] Error generating space suggestions: {e}", exc_info=True)

        return suggestions

    async def _generate_pattern_reminders(self, insights: Dict[str, Any]) -> List[ProactiveSuggestion]:
        """Generate reminders based on temporal patterns"""
        suggestions = []

        try:
            now = datetime.now()

            # Check temporal habits
            habits = insights.get('temporal_habits', [])

            for habit in habits:
                if habit['hour'] == now.hour and habit['occurrences'] > 5:
                    suggestion = ProactiveSuggestion(
                        suggestion_id=f"reminder_{habit['action']}_{int(time.time())}",
                        suggestion_type=SuggestionType.PATTERN_REMINDER,
                        priority=CommunicationPriority.MEDIUM,
                        confidence=habit['confidence'],
                        title="Your usual routine",
                        message=self._generate_reminder_message(habit),
                        reasoning=f"You typically {habit['action']} around this time",
                        action_type="reminder",
                        action_params={'action': habit['action']},
                        context={'habit': habit}
                    )
                    suggestions.append(suggestion)

        except Exception as e:
            logger.error(f"[PIE] Error generating pattern reminders: {e}", exc_info=True)

        return suggestions

    async def _generate_ml_predictions(self) -> List[ProactiveSuggestion]:
        """Generate suggestions from ML Pattern Learner"""
        suggestions = []

        try:
            if not self.pattern_learner:
                return suggestions

            # Get learned patterns
            patterns = self.pattern_learner.get_learned_patterns(min_confidence=0.7)

            for pattern in patterns[:2]:  # Top 2 patterns
                if pattern.pattern_type.value == 'workflow':
                    suggestion = ProactiveSuggestion(
                        suggestion_id=f"ml_pattern_{pattern.pattern_id}",
                        suggestion_type=SuggestionType.BEHAVIORAL_INSIGHT,
                        priority=CommunicationPriority.LOW,
                        confidence=pattern.confidence,
                        title="Pattern Detected",
                        message=f"I've noticed you frequently {pattern.pattern_id}. Want me to automate this?",
                        reasoning=f"ML detected pattern with {pattern.confidence:.0%} confidence",
                        action_type="create_automation",
                        action_params={'pattern_id': pattern.pattern_id},
                        context={'pattern': pattern}
                    )
                    suggestions.append(suggestion)

        except Exception as e:
            logger.error(f"[PIE] Error generating ML predictions: {e}", exc_info=True)

        return suggestions

    # ========================================================================
    # Suggestion Filtering and Prioritization
    # ========================================================================

    def _filter_suggestions(self, suggestions: List[ProactiveSuggestion]) -> List[ProactiveSuggestion]:
        """Filter and prioritize suggestions"""
        if not suggestions:
            return []

        # Filter by confidence threshold
        threshold = self.user_preferences['interruption_threshold']
        filtered = [s for s in suggestions if s.confidence >= threshold]

        # Remove duplicates
        seen = set()
        unique = []
        for sug in filtered:
            key = (sug.suggestion_type, sug.action_type, str(sug.action_params))
            if key not in seen:
                seen.add(key)
                unique.append(sug)

        # Sort by priority and confidence
        unique.sort(key=lambda s: (s.priority.value, s.confidence), reverse=True)

        # Limit based on focus level
        if self.current_context.user_focus_level == UserFocusLevel.FOCUSED:
            return unique[:1]  # Only most important
        elif self.current_context.user_focus_level == UserFocusLevel.CASUAL:
            return unique[:2]  # Top 2
        else:
            return unique[:3]  # Top 3

    # ========================================================================
    # Suggestion Presentation
    # ========================================================================

    async def _present_suggestions(self, suggestions: List[ProactiveSuggestion]):
        """Present suggestions to user via voice/notifications"""
        for suggestion in suggestions:
            try:
                # Mark as presented
                suggestion.presented = True
                self.pending_suggestions.append(suggestion)
                self.stats['suggestions_presented'] += 1

                # Choose communication method
                if self.user_preferences['voice_enabled'] and self.voice_callback:
                    await self._present_via_voice(suggestion)
                elif self.notification_callback:
                    await self._present_via_notification(suggestion)

                # Update last suggestion time
                self.last_suggestion_time = time.time()

                logger.info(f"[PIE] Presented suggestion: {suggestion.title}")

            except Exception as e:
                logger.error(f"[PIE] Error presenting suggestion: {e}", exc_info=True)

    async def _present_via_voice(self, suggestion: ProactiveSuggestion):
        """Present suggestion via voice"""
        try:
            if not self.voice_callback:
                return

            # Generate natural voice message
            voice_message = self._generate_voice_message(suggestion)

            # Call voice callback
            await self.voice_callback(voice_message)

            self.stats['voice_messages_sent'] += 1

        except Exception as e:
            logger.error(f"[PIE] Error in voice presentation: {e}", exc_info=True)

    async def _present_via_notification(self, suggestion: ProactiveSuggestion):
        """Present suggestion via notification"""
        try:
            if not self.notification_callback:
                return

            notification = {
                'title': suggestion.title,
                'message': suggestion.message,
                'priority': suggestion.priority.value,
                'action_type': suggestion.action_type,
                'action_params': suggestion.action_params
            }

            await self.notification_callback(notification)

            self.stats['notifications_sent'] += 1

        except Exception as e:
            logger.error(f"[PIE] Error in notification presentation: {e}", exc_info=True)

    # ========================================================================
    # Natural Language Generation
    # ========================================================================

    def _generate_voice_message(self, suggestion: ProactiveSuggestion) -> str:
        """Generate natural, human-like voice message"""
        personality = self.user_preferences['personality_level']

        # Add casual intro based on personality level
        intros = []
        if personality > 0.7:
            intros = ["Hey", "So", "I noticed", "Just so you know", "By the way"]
        elif personality > 0.4:
            intros = ["I see", "I noticed", "It looks like"]
        else:
            intros = [""]

        intro = random.choice(intros) if intros else ""

        # Main message
        message = suggestion.message

        # Add reasoning if high confidence
        if suggestion.confidence > 0.8 and personality > 0.5:
            reasoning = f" {suggestion.reasoning}"
        else:
            reasoning = ""

        # Combine
        if intro:
            return f"{intro}, {message.lower()}{reasoning}"
        else:
            return f"{message}{reasoning}"

    def _generate_workflow_message(self, workflow: Dict) -> str:
        """Generate message for workflow optimization"""
        templates = [
            f"I've been watching your {workflow['name']} workflow. Want me to help optimize it?",
            f"Your {workflow['name']} workflow could be more efficient. I have some ideas.",
            f"I notice you do {workflow['name']} often. I can help make it smoother."
        ]
        return random.choice(templates)

    def _generate_predictive_message(self, prediction: Dict) -> str:
        """Generate message for predictive app launch"""
        app = prediction['target']
        templates = [
            f"You usually open {app} around now. Want me to launch it?",
            f"Ready to work on {app}? I can open it for you.",
            f"Based on your pattern, you're about to use {app}. Should I open it?"
        ]
        return random.choice(templates)

    def _generate_space_switch_message(self, transition: Dict) -> str:
        """Generate message for space switching"""
        to_space = transition['to']
        templates = [
            f"You typically switch to Space {to_space} around now. Want me to switch?",
            f"Ready to move to Space {to_space}? That's your usual next step.",
            f"Based on your pattern, you usually go to Space {to_space} next."
        ]
        return random.choice(templates)

    def _generate_reminder_message(self, habit: Dict) -> str:
        """Generate message for pattern reminder"""
        action = habit['action']
        templates = [
            f"Just a heads up - you usually {action} around this time.",
            f"Reminder: you typically {action} now based on your habits.",
            f"This is when you normally {action}. Just thought you should know."
        ]
        return random.choice(templates)

    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Initialize message templates"""
        return {
            'greeting': [
                "Hey there",
                "Hi",
                "Hello",
                "Good {time_of_day}"
            ],
            'acknowledgment': [
                "Got it",
                "Understood",
                "Okay",
                "Sure thing",
                "Will do"
            ],
            'confirmation': [
                "Done",
                "All set",
                "Completed",
                "Finished"
            ]
        }

    # ========================================================================
    # User Response Handling
    # ========================================================================

    async def handle_user_response(
        self,
        suggestion_id: str,
        response: str,  # 'accepted', 'rejected', 'ignored'
        feedback: Optional[str] = None
    ):
        """Handle user response to a suggestion"""
        try:
            # Find suggestion
            suggestion = None
            for s in self.pending_suggestions:
                if s.suggestion_id == suggestion_id:
                    suggestion = s
                    break

            if not suggestion:
                logger.warning(f"[PIE] Suggestion not found: {suggestion_id}")
                return

            # Update suggestion
            suggestion.user_response = response
            suggestion.response_time = time.time()

            # Update stats
            if response == 'accepted':
                self.stats['suggestions_accepted'] += 1

                # Execute action if accepted
                await self._execute_suggestion_action(suggestion)

            elif response == 'rejected':
                self.stats['suggestions_rejected'] += 1
            else:
                self.stats['suggestions_ignored'] += 1

            # Store feedback to Learning DB
            if self.learning_db and response != 'ignored':
                # This helps JARVIS learn what suggestions are valuable
                accepted = (response == 'accepted')
                # Note: We'd need to add a method to track suggestion feedback
                logger.info(f"[PIE] User {response} suggestion: {suggestion.title}")

            # Move to history
            self.suggestion_history.append(suggestion)
            self.pending_suggestions.remove(suggestion)

        except Exception as e:
            logger.error(f"[PIE] Error handling user response: {e}", exc_info=True)

    async def _execute_suggestion_action(self, suggestion: ProactiveSuggestion):
        """Execute the action for an accepted suggestion"""
        try:
            action_type = suggestion.action_type
            params = suggestion.action_params

            logger.info(f"[PIE] Executing action: {action_type} with params: {params}")

            # Route to appropriate handler
            if action_type == "launch_app":
                await self._execute_launch_app(params)
            elif action_type == "switch_space":
                await self._execute_switch_space(params)
            elif action_type == "optimize_workflow":
                await self._execute_optimize_workflow(params)

        except Exception as e:
            logger.error(f"[PIE] Error executing action: {e}", exc_info=True)

    async def _execute_launch_app(self, params: Dict):
        """Execute app launch"""
        app_name = params.get('app_name')
        logger.info(f"[PIE] Launching app: {app_name}")
        # Implementation would use macOS APIs to launch app

    async def _execute_switch_space(self, params: Dict):
        """Execute space switch"""
        target_space = params.get('target_space')
        logger.info(f"[PIE] Switching to Space: {target_space}")
        # Implementation would use Yabai to switch spaces

    async def _execute_optimize_workflow(self, params: Dict):
        """Execute workflow optimization"""
        workflow_name = params.get('workflow_name')
        logger.info(f"[PIE] Optimizing workflow: {workflow_name}")
        # Implementation would analyze and suggest workflow improvements

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    def _calculate_acceptance_rate(self) -> float:
        """Calculate suggestion acceptance rate"""
        total_responded = self.stats['suggestions_accepted'] + self.stats['suggestions_rejected']
        if total_responded == 0:
            return 0.0
        return self.stats['suggestions_accepted'] / total_responded

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            **self.stats,
            'acceptance_rate': self._calculate_acceptance_rate(),
            'pending_suggestions': len(self.pending_suggestions),
            'current_focus_level': self.current_context.user_focus_level.value,
            'time_since_last_suggestion': self.current_context.time_since_last_message
        }


# ============================================================================
# Factory Functions
# ============================================================================

_pie_instance: Optional[ProactiveIntelligenceEngine] = None


async def initialize_proactive_intelligence(
    learning_db=None,
    pattern_learner=None,
    yabai_intelligence=None,
    uae_engine=None,
    voice_callback: Optional[Callable] = None,
    notification_callback: Optional[Callable] = None
) -> ProactiveIntelligenceEngine:
    """Initialize and start Proactive Intelligence Engine"""
    global _pie_instance

    if _pie_instance is None:
        _pie_instance = ProactiveIntelligenceEngine(
            learning_db=learning_db,
            pattern_learner=pattern_learner,
            yabai_intelligence=yabai_intelligence,
            uae_engine=uae_engine,
            voice_callback=voice_callback,
            notification_callback=notification_callback
        )

        await _pie_instance.start()
        logger.info("[PIE] Proactive Intelligence Engine initialized and started")

    return _pie_instance


def get_proactive_intelligence() -> Optional[ProactiveIntelligenceEngine]:
    """Get existing Proactive Intelligence Engine instance"""
    return _pie_instance


async def shutdown_proactive_intelligence():
    """Shutdown Proactive Intelligence Engine"""
    global _pie_instance

    if _pie_instance:
        await _pie_instance.stop()
        _pie_instance = None
        logger.info("[PIE] Proactive Intelligence Engine shutdown complete")
