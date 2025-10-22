#!/usr/bin/env python3
"""
Autonomous Decision Engine for JARVIS
Makes intelligent decisions based on workspace state without hardcoding
"""

import asyncio
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path

# Import existing vision components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from vision.workspace_analyzer import WorkspaceAnalysis
from vision.window_detector import WindowInfo
from vision.smart_query_router import SmartQueryRouter, QueryIntent

# Import Goal Inference System for predictive automation
from vision.intelligence.goal_inference_system import (
    GoalInferenceEngine, Goal, GoalLevel,
    get_goal_inference_engine
)

logger = logging.getLogger(__name__)

class ActionPriority(Enum):
    """Priority levels for autonomous actions"""
    CRITICAL = 1      # Immediate action required (security, urgent messages)
    HIGH = 2          # Important but not immediate (meeting prep, deadlines)
    MEDIUM = 3        # Standard actions (routine messages, organization)
    LOW = 4           # Nice to have (cleanup, optimization)
    BACKGROUND = 5    # Can wait indefinitely

class ActionCategory(Enum):
    """Categories of autonomous actions"""
    COMMUNICATION = "communication"
    CALENDAR = "calendar"
    NOTIFICATION = "notification"
    ORGANIZATION = "organization"
    SECURITY = "security"
    WORKFLOW = "workflow"
    MAINTENANCE = "maintenance"

@dataclass
class AutonomousAction:
    """Represents an autonomous action JARVIS can take"""
    action_type: str
    target: str
    params: Dict[str, Any]
    priority: ActionPriority
    confidence: float
    category: ActionCategory
    reasoning: str
    requires_permission: bool = field(init=False)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Determine if permission is required based on confidence and category"""
        # Critical security actions always require permission
        if self.category == ActionCategory.SECURITY and self.priority == ActionPriority.CRITICAL:
            self.requires_permission = True
        # High confidence actions don't require permission (unless security)
        elif self.confidence >= 0.85:
            self.requires_permission = False
        # Medium confidence requires permission for important actions
        elif self.confidence >= 0.7:
            self.requires_permission = self.priority in [ActionPriority.CRITICAL, ActionPriority.HIGH]
        # Low confidence always requires permission
        else:
            self.requires_permission = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'action_type': self.action_type,
            'target': self.target,
            'params': self.params,
            'priority': self.priority.name,
            'confidence': self.confidence,
            'category': self.category.value,
            'reasoning': self.reasoning,
            'requires_permission': self.requires_permission,
            'timestamp': self.timestamp.isoformat()
        }

class PatternMatcher:
    """Dynamic pattern matching for identifying actionable situations"""
    
    def __init__(self):
        # Dynamic patterns that learn and adapt
        self.notification_patterns = [
            r'\((\d+)\)',                    # (5) style
            r'\[(\d+)\]',                    # [3] style
            r'(\d+)\s+new',                  # 5 new
            r'(\d+)\s+unread',               # 3 unread
            r'(\d+)\s+notification',         # 2 notifications
            r'(\d+)\s+message',              # 4 messages
            r'•{2,}',                        # ••• dots
            r'!+',                           # !!! urgency
            r'🔴|🟡|🔵',                     # Color indicators
        ]
        
        self.urgency_indicators = [
            'urgent', 'asap', 'important', 'critical', 'emergency',
            'deadline', 'overdue', 'expired', 'ending soon', 'final',
            'immediate', 'priority', 'action required', 'response needed'
        ]
        
        self.meeting_patterns = [
            r'meeting\s+in\s+(\d+)\s+min',
            r'starts?\s+in\s+(\d+)',
            r'beginning\s+soon',
            r'about\s+to\s+start',
            'zoom', 'teams', 'meet', 'webex', 'conference'
        ]
        
        self.security_patterns = [
            'password', 'credential', 'api key', 'secret', 'token',
            'authentication', 'login', 'sign in', 'verify', '2fa',
            'bank', 'financial', 'ssn', 'credit card'
        ]
    
    def extract_notification_count(self, text: str) -> Optional[int]:
        """Extract notification count from text dynamically"""
        for pattern in self.notification_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.groups():
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    pass
        return None
    
    def calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score based on indicators (0-1)"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        
        # Check urgency indicators
        for indicator in self.urgency_indicators:
            if indicator in text_lower:
                score += 0.2
        
        # Check for time pressure
        time_match = re.search(r'(\d+)\s*(min|hour|hr)', text_lower)
        if time_match:
            time_value = int(time_match.group(1))
            unit = time_match.group(2)
            if 'min' in unit and time_value <= 30:
                score += 0.3
            elif 'hour' in unit and time_value <= 2:
                score += 0.2
        
        # Check for caps (indicating urgency)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            score += 0.1
        
        # Check for multiple exclamation marks
        if text.count('!') >= 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def is_meeting_related(self, text: str) -> Tuple[bool, Optional[int]]:
        """Check if text is meeting related and extract time if available"""
        text_lower = text.lower()
        
        for pattern in self.meeting_patterns:
            if isinstance(pattern, str) and pattern in text_lower:
                return True, None
            else:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        minutes = int(match.group(1))
                        return True, minutes
                    except:
                        return True, None
        
        return False, None
    
    def contains_sensitive_content(self, text: str) -> bool:
        """Check if text contains sensitive information"""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.security_patterns)

class AutonomousDecisionEngine:
    """Makes autonomous decisions based on workspace state with Goal Inference integration"""

    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.action_history = []
        self.learned_patterns = self._load_learned_patterns()
        self.query_router = SmartQueryRouter()

        # Initialize Goal Inference Engine for predictive automation
        self.goal_inference = get_goal_inference_engine()

        # Decision handlers for different contexts
        self.decision_handlers = {}

        # Goal-based action predictors
        self.goal_action_mappings = self._initialize_goal_action_mappings()

        # Decision thresholds (can be adjusted based on learning)
        self.thresholds = {
            'notification_action': 3,      # Act on 3+ notifications
            'urgency_threshold': 0.6,      # Urgency score for high priority
            'meeting_prep_time': 5,        # Minutes before meeting to prepare
            'pattern_confidence': 0.7,     # Confidence needed for learned patterns
            'goal_confidence': 0.75,       # Confidence needed for goal-based actions
        }

        # Action templates (dynamically expandable)
        self.action_templates = self._load_action_templates()

        logger.info("Autonomous Decision Engine initialized with Goal Inference")
    
    def register_decision_handler(self, handler_name: str, handler_func):
        """Register a decision handler for specific contexts"""
        self.decision_handlers[handler_name] = handler_func
        logger.info(f"Registered decision handler: {handler_name}")
    
    async def process_decision_handlers(self, context: Dict[str, Any]) -> List[AutonomousAction]:
        """Process all registered decision handlers with the given context"""
        all_actions = []
        
        for handler_name, handler_func in self.decision_handlers.items():
            try:
                handler_actions = await handler_func(context)
                if handler_actions:
                    all_actions.extend(handler_actions)
                    logger.debug(f"Handler {handler_name} generated {len(handler_actions)} actions")
            except Exception as e:
                logger.error(f"Error in decision handler {handler_name}: {e}")
        
        return all_actions
    
    def _load_learned_patterns(self) -> Dict[str, Any]:
        """Load learned patterns from persistent storage"""
        patterns_file = Path("backend/data/learned_patterns.json")
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load learned patterns: {e}")
        
        return {
            'app_behaviors': {},
            'user_preferences': {},
            'action_success_rates': {},
            'timing_patterns': {}
        }
    
    def _initialize_goal_action_mappings(self) -> Dict[str, List[str]]:
        """Initialize mappings between goals and potential actions"""
        return {
            # High-level goal mappings
            'project_completion': ['open_application', 'organize_workspace', 'connect_display'],
            'problem_solving': ['open_application', 'search_information', 'organize_workspace'],
            'information_gathering': ['search_information', 'open_browser', 'take_notes'],
            'communication': ['handle_notifications', 'respond_message', 'open_communication_app'],
            'learning_research': ['open_browser', 'take_notes', 'organize_workspace'],

            # Intermediate goal mappings
            'feature_implementation': ['open_application', 'connect_display', 'organize_workspace'],
            'bug_fixing': ['open_debugger', 'search_error', 'view_logs'],
            'document_preparation': ['open_document', 'organize_workspace', 'connect_display'],
            'meeting_preparation': ['prepare_meeting', 'open_calendar', 'connect_display'],
            'response_composition': ['respond_message', 'open_communication_app'],

            # Immediate goal mappings
            'find_information': ['search_information', 'open_browser'],
            'fix_error': ['view_logs', 'open_debugger', 'search_error'],
            'complete_form': ['open_browser', 'fill_form'],
            'send_message': ['respond_message', 'open_communication_app'],
            'review_content': ['open_document', 'organize_workspace'],

            # Display-specific goal (primary focus for integration)
            'connect_display': ['connect_display', 'configure_display_mode']
        }

    def _load_action_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load action templates (can be extended dynamically)"""
        return {
            'handle_notifications': {
                'description': 'Process notifications in {app}',
                'params': ['app', 'count', 'window_id'],
                'category': ActionCategory.NOTIFICATION
            },
            'prepare_meeting': {
                'description': 'Prepare workspace for meeting',
                'params': ['meeting_info', 'minutes_until'],
                'category': ActionCategory.CALENDAR
            },
            'organize_workspace': {
                'description': 'Organize windows for {task}',
                'params': ['task', 'window_arrangement'],
                'category': ActionCategory.ORGANIZATION
            },
            'security_alert': {
                'description': 'Handle security concern in {app}',
                'params': ['app', 'concern_type', 'window_id'],
                'category': ActionCategory.SECURITY
            },
            'respond_message': {
                'description': 'Respond to message in {app}',
                'params': ['app', 'message_preview', 'suggested_response'],
                'category': ActionCategory.COMMUNICATION
            },
            'cleanup_workspace': {
                'description': 'Clean up {type} windows',
                'params': ['type', 'window_ids'],
                'category': ActionCategory.MAINTENANCE
            },
            'connect_display': {
                'description': 'Connect to {display_name}',
                'params': ['display_name', 'connection_type', 'mode'],
                'category': ActionCategory.WORKFLOW
            },
            'open_application': {
                'description': 'Open {app_name}',
                'params': ['app_name', 'reason'],
                'category': ActionCategory.WORKFLOW
            }
        }
    
    async def analyze_and_decide(self, workspace_state: WorkspaceAnalysis,
                                 windows: List[WindowInfo]) -> List[AutonomousAction]:
        """Analyze workspace and determine autonomous actions with Goal Inference"""
        actions = []

        # STEP 1: Infer user goals from context
        goal_context = self._create_goal_context(workspace_state, windows)
        inferred_goals = await self.goal_inference.infer_goals(goal_context)

        # STEP 2: Generate goal-based autonomous actions
        goal_actions = await self._generate_goal_based_actions(inferred_goals, workspace_state)
        actions.extend(goal_actions)

        # STEP 3: Analyze each window for actionable situations
        for window in windows:
            window_actions = await self._analyze_window(window, workspace_state)
            actions.extend(window_actions)

        # STEP 4: Analyze overall workspace patterns
        workspace_actions = await self._analyze_workspace_patterns(workspace_state, windows)
        actions.extend(workspace_actions)

        # STEP 5: Analyze temporal patterns (time-based actions)
        temporal_actions = await self._analyze_temporal_patterns(workspace_state, windows)
        actions.extend(temporal_actions)

        # STEP 6: Process registered decision handlers with goals
        context = {
            'workspace_state': workspace_state,
            'windows': windows,
            'inferred_goals': inferred_goals,
            'detected_notifications': []  # Will be populated by handlers
        }
        handler_actions = await self.process_decision_handlers(context)
        actions.extend(handler_actions)

        # STEP 7: Apply learned optimizations
        actions = self._apply_learned_optimizations(actions)

        # STEP 8: Deduplicate and prioritize actions
        actions = self._deduplicate_actions(actions)

        # Sort by priority and confidence
        actions.sort(key=lambda a: (a.priority.value, -a.confidence))

        # Record decisions for learning
        self._record_decisions(actions)

        # Update goal progress based on actions
        await self._update_goal_progress(actions, inferred_goals)

        return actions
    
    async def _analyze_window(self, window: WindowInfo, 
                            workspace_state: WorkspaceAnalysis) -> List[AutonomousAction]:
        """Analyze individual window for autonomous actions"""
        actions = []
        
        # Extract window title and content
        title = window.window_title.lower() if window.window_title else ""
        app_name = window.app_name
        
        # Check for notifications
        notification_count = self.pattern_matcher.extract_notification_count(window.window_title)
        if notification_count and notification_count >= self.thresholds['notification_action']:
            action = AutonomousAction(
                action_type='handle_notifications',
                target=app_name,
                params={
                    'app': app_name,
                    'count': notification_count,
                    'window_id': window.window_id
                },
                priority=ActionPriority.MEDIUM,
                confidence=0.8,
                category=ActionCategory.NOTIFICATION,
                reasoning=f"Detected {notification_count} notifications in {app_name}"
            )
            actions.append(action)
        
        # Check urgency
        urgency_score = self.pattern_matcher.calculate_urgency_score(window.window_title)
        if urgency_score >= self.thresholds['urgency_threshold']:
            priority = ActionPriority.HIGH if urgency_score >= 0.8 else ActionPriority.MEDIUM
            action = AutonomousAction(
                action_type='handle_urgent_item',
                target=app_name,
                params={
                    'app': app_name,
                    'urgency_score': urgency_score,
                    'window_id': window.window_id,
                    'title': window.window_title
                },
                priority=priority,
                confidence=urgency_score,
                category=ActionCategory.COMMUNICATION,
                reasoning=f"High urgency detected in {app_name}: {window.window_title}"
            )
            actions.append(action)
        
        # Check for meetings
        is_meeting, minutes = self.pattern_matcher.is_meeting_related(window.window_title)
        if is_meeting and minutes and minutes <= self.thresholds['meeting_prep_time']:
            action = AutonomousAction(
                action_type='prepare_meeting',
                target='calendar',
                params={
                    'meeting_info': window.window_title,
                    'minutes_until': minutes,
                    'source_app': app_name
                },
                priority=ActionPriority.HIGH,
                confidence=0.9,
                category=ActionCategory.CALENDAR,
                reasoning=f"Meeting starting in {minutes} minutes"
            )
            actions.append(action)
        
        # Check for security concerns
        if self.pattern_matcher.contains_sensitive_content(window.window_title):
            action = AutonomousAction(
                action_type='security_alert',
                target=app_name,
                params={
                    'app': app_name,
                    'concern_type': 'sensitive_content',
                    'window_id': window.window_id
                },
                priority=ActionPriority.CRITICAL,
                confidence=0.95,
                category=ActionCategory.SECURITY,
                reasoning=f"Sensitive content detected in {app_name}"
            )
            actions.append(action)
        
        return actions
    
    async def _analyze_workspace_patterns(self, workspace_state: WorkspaceAnalysis,
                                        windows: List[WindowInfo]) -> List[AutonomousAction]:
        """Analyze overall workspace patterns for actions"""
        actions = []
        
        # Group windows by application type
        app_groups = {}
        for window in windows:
            app_type = self._classify_app_dynamically(window)
            if app_type not in app_groups:
                app_groups[app_type] = []
            app_groups[app_type].append(window)
        
        # Check for workspace organization opportunities
        if len(windows) > 10:  # Many windows open
            action = AutonomousAction(
                action_type='organize_workspace',
                target='workspace',
                params={
                    'task': workspace_state.focused_task,
                    'window_arrangement': self._suggest_arrangement(app_groups),
                    'window_count': len(windows)
                },
                priority=ActionPriority.LOW,
                confidence=0.7,
                category=ActionCategory.ORGANIZATION,
                reasoning=f"Detected {len(windows)} windows - suggesting organization"
            )
            actions.append(action)
        
        # Check for distraction patterns
        if 'entertainment' in app_groups and 'productivity' in app_groups:
            if workspace_state.focused_task and 'work' in workspace_state.focused_task.lower():
                action = AutonomousAction(
                    action_type='minimize_distractions',
                    target='workspace',
                    params={
                        'distraction_apps': [w.app_name for w in app_groups.get('entertainment', [])],
                        'focus_task': workspace_state.focused_task
                    },
                    priority=ActionPriority.MEDIUM,
                    confidence=0.75,
                    category=ActionCategory.WORKFLOW,
                    reasoning="Detected potential distractions during focused work"
                )
                actions.append(action)
        
        return actions
    
    async def _analyze_temporal_patterns(self, workspace_state: WorkspaceAnalysis,
                                       windows: List[WindowInfo]) -> List[AutonomousAction]:
        """Analyze time-based patterns for actions"""
        actions = []
        current_time = datetime.now()
        
        # Check learned timing patterns
        hour = current_time.hour
        day = current_time.weekday()
        
        timing_key = f"{day}_{hour}"
        if timing_key in self.learned_patterns.get('timing_patterns', {}):
            pattern = self.learned_patterns['timing_patterns'][timing_key]
            if pattern['confidence'] >= self.thresholds['pattern_confidence']:
                action = AutonomousAction(
                    action_type='routine_automation',
                    target='learned_routine',
                    params={
                        'routine_name': pattern['name'],
                        'expected_apps': pattern['apps'],
                        'typical_duration': pattern['duration']
                    },
                    priority=ActionPriority.LOW,
                    confidence=pattern['confidence'],
                    category=ActionCategory.WORKFLOW,
                    reasoning=f"Time for usual {pattern['name']} routine"
                )
                actions.append(action)
        
        # Check for end-of-day cleanup
        if hour >= 17 and len(windows) > 15:
            action = AutonomousAction(
                action_type='cleanup_workspace',
                target='workspace',
                params={
                    'type': 'end_of_day',
                    'window_ids': [w.window_id for w in windows if not self._is_essential(w)]
                },
                priority=ActionPriority.LOW,
                confidence=0.6,
                category=ActionCategory.MAINTENANCE,
                reasoning="End of day workspace cleanup suggested"
            )
            actions.append(action)
        
        return actions
    
    def _classify_app_dynamically(self, window: WindowInfo) -> str:
        """Classify app type dynamically based on patterns"""
        app_name_lower = window.app_name.lower()
        title_lower = (window.window_title or "").lower()
        
        # Communication patterns
        comm_patterns = ['chat', 'message', 'mail', 'slack', 'discord', 'teams', 'zoom']
        if any(p in app_name_lower or p in title_lower for p in comm_patterns):
            return 'communication'
        
        # Productivity patterns
        prod_patterns = ['code', 'editor', 'ide', 'terminal', 'console', 'jupyter']
        if any(p in app_name_lower or p in title_lower for p in prod_patterns):
            return 'productivity'
        
        # Browser patterns - check content
        browser_patterns = ['chrome', 'safari', 'firefox', 'edge', 'browser']
        if any(p in app_name_lower for p in browser_patterns):
            # Further classify based on title
            if any(p in title_lower for p in ['github', 'stackoverflow', 'docs']):
                return 'productivity'
            elif any(p in title_lower for p in ['youtube', 'netflix', 'reddit']):
                return 'entertainment'
            else:
                return 'browser'
        
        # Entertainment patterns
        ent_patterns = ['spotify', 'music', 'video', 'game', 'play']
        if any(p in app_name_lower or p in title_lower for p in ent_patterns):
            return 'entertainment'
        
        # Default
        return 'other'
    
    def _suggest_arrangement(self, app_groups: Dict[str, List[WindowInfo]]) -> Dict[str, Any]:
        """Suggest window arrangement based on app groups"""
        arrangement = {
            'primary_focus': [],
            'secondary': [],
            'minimize': []
        }
        
        # Prioritize productivity apps
        if 'productivity' in app_groups:
            arrangement['primary_focus'] = [w.window_id for w in app_groups['productivity'][:2]]
        
        # Communication as secondary
        if 'communication' in app_groups:
            arrangement['secondary'] = [w.window_id for w in app_groups['communication'][:2]]
        
        # Minimize entertainment during work
        if 'entertainment' in app_groups:
            arrangement['minimize'] = [w.window_id for w in app_groups['entertainment']]
        
        return arrangement
    
    def _is_essential(self, window: WindowInfo) -> bool:
        """Determine if window is essential and shouldn't be closed"""
        essential_patterns = ['finder', 'system', 'security', 'vpn', 'password']
        return any(p in window.app_name.lower() for p in essential_patterns)
    
    def _apply_learned_optimizations(self, actions: List[AutonomousAction]) -> List[AutonomousAction]:
        """Apply learned patterns to optimize actions"""
        optimized = []
        
        for action in actions:
            # Check success rate of similar actions
            action_key = f"{action.action_type}:{action.category.value}"
            if action_key in self.learned_patterns.get('action_success_rates', {}):
                success_rate = self.learned_patterns['action_success_rates'][action_key]
                
                # Adjust confidence based on historical success
                action.confidence *= success_rate
                
                # Skip low-success actions
                if action.confidence < 0.5:
                    logger.info(f"Skipping low-success action: {action.action_type}")
                    continue
            
            optimized.append(action)
        
        return optimized
    
    def _record_decisions(self, actions: List[AutonomousAction]):
        """Record decisions for future learning"""
        for action in actions:
            self.action_history.append({
                'timestamp': action.timestamp.isoformat(),
                'action': action.to_dict(),
                'workspace_context': {
                    'time': datetime.now().isoformat(),
                    'day': datetime.now().weekday(),
                    'hour': datetime.now().hour
                }
            })
        
        # Keep only recent history (last 1000 actions)
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]
    
    def _create_goal_context(self, workspace_state: WorkspaceAnalysis,
                           windows: List[WindowInfo]) -> Dict[str, Any]:
        """Create context for goal inference"""
        # Extract application names
        active_apps = list(set(w.app_name for w in windows if w.app_name))

        # Extract recent actions from workspace state
        recent_actions = []
        if hasattr(workspace_state, 'suggestions'):
            recent_actions = workspace_state.suggestions[:5]

        # Build context for goal inference
        return {
            'active_applications': active_apps,
            'recent_actions': recent_actions,
            'content': {
                'type': self._infer_content_type(windows),
                'focused_task': workspace_state.focused_task if workspace_state else None
            },
            'time_context': {
                'time_of_day': self._get_time_period(),
                'day_of_week': 'weekday' if datetime.now().weekday() < 5 else 'weekend'
            },
            'workspace_context': workspace_state.workspace_context if workspace_state else None
        }

    def _infer_content_type(self, windows: List[WindowInfo]) -> str:
        """Infer content type from window titles"""
        titles = ' '.join(w.window_title for w in windows if w.window_title)
        titles_lower = titles.lower()

        if any(term in titles_lower for term in ['code', '.py', '.js', '.cpp', 'vscode']):
            return 'code'
        elif any(term in titles_lower for term in ['doc', 'presentation', 'slides']):
            return 'document'
        elif any(term in titles_lower for term in ['email', 'message', 'chat']):
            return 'communication'
        else:
            return 'general'

    def _get_time_period(self) -> str:
        """Get current time period"""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    async def _generate_goal_based_actions(self, inferred_goals: Dict[GoalLevel, List[Goal]],
                                          workspace_state: WorkspaceAnalysis) -> List[AutonomousAction]:
        """Generate autonomous actions based on inferred goals"""
        actions = []

        # Process goals at each level
        for level, goals in inferred_goals.items():
            for goal in goals:
                # Only act on high-confidence goals
                if goal.confidence < self.thresholds['goal_confidence']:
                    continue

                # Get potential actions for this goal type
                potential_actions = self.goal_action_mappings.get(goal.goal_type, [])

                for action_type in potential_actions:
                    # Special handling for display connection (our primary focus)
                    if action_type == 'connect_display':
                        action = await self._create_display_action_from_goal(goal, workspace_state)
                        if action:
                            actions.append(action)

                    # Handle other action types
                    elif action_type in self.action_templates:
                        action = self._create_action_from_goal(goal, action_type, workspace_state)
                        if action:
                            actions.append(action)

        return actions

    async def _create_display_action_from_goal(self, goal: Goal,
                                              workspace_state: WorkspaceAnalysis) -> Optional[AutonomousAction]:
        """Create display connection action based on goal"""
        # Check if display connection is relevant to the goal
        display_relevant_goals = [
            'project_completion', 'feature_implementation',
            'document_preparation', 'meeting_preparation'
        ]

        if goal.goal_type not in display_relevant_goals:
            return None

        # Determine display target dynamically
        display_name = await self._predict_display_target(goal, workspace_state)

        if not display_name:
            display_name = "Living Room TV"  # Default fallback

        # Calculate confidence based on goal and context
        confidence = goal.confidence * 0.9  # Slightly reduce confidence

        # Check if it's a good time to connect
        if await self._should_auto_connect_display(goal, workspace_state):
            return AutonomousAction(
                action_type='connect_display',
                target=display_name,
                params={
                    'display_name': display_name,
                    'connection_type': 'auto',
                    'mode': 'extend',  # or 'mirror' based on goal
                    'goal_id': goal.goal_id,
                    'goal_type': goal.goal_type
                },
                priority=ActionPriority.MEDIUM,
                confidence=confidence,
                category=ActionCategory.WORKFLOW,
                reasoning=f"Goal '{goal.description}' suggests connecting to {display_name}"
            )

        return None

    async def _predict_display_target(self, goal: Goal,
                                     workspace_state: WorkspaceAnalysis) -> Optional[str]:
        """Predict which display to connect to based on goal"""
        # Check for display references in goal evidence
        for evidence in goal.evidence:
            if 'display' in str(evidence.get('data', '')).lower():
                # Extract display name from evidence
                data_str = str(evidence['data'])
                if 'Living Room TV' in data_str:
                    return 'Living Room TV'
                elif 'monitor' in data_str.lower():
                    return 'External Monitor'

        # Check workspace state for display hints
        if workspace_state and workspace_state.workspace_context:
            context_lower = workspace_state.workspace_context.lower()
            if 'presentation' in context_lower or 'meeting' in context_lower:
                return 'Living Room TV'
            elif 'coding' in context_lower or 'development' in context_lower:
                return 'External Monitor'

        # Use learned patterns
        if goal.goal_type in self.learned_patterns.get('display_preferences', {}):
            return self.learned_patterns['display_preferences'][goal.goal_type]

        return None

    async def _should_auto_connect_display(self, goal: Goal,
                                          workspace_state: WorkspaceAnalysis) -> bool:
        """Determine if display should be auto-connected based on goal"""
        # High confidence immediate goals should trigger auto-connect
        if goal.level == GoalLevel.IMMEDIATE and goal.confidence > 0.85:
            return True

        # Meeting preparation goals with time pressure
        if goal.goal_type == 'meeting_preparation':
            # Check for meeting timing in evidence
            for evidence in goal.evidence:
                if 'time' in evidence.get('source', ''):
                    return True

        # Project completion with high progress
        if goal.goal_type == 'project_completion' and goal.progress > 0.7:
            return True

        # Check learned patterns for this goal type
        pattern_key = f"auto_connect_{goal.goal_type}"
        if pattern_key in self.learned_patterns.get('user_preferences', {}):
            return self.learned_patterns['user_preferences'][pattern_key]

        return False

    def _create_action_from_goal(self, goal: Goal, action_type: str,
                                workspace_state: WorkspaceAnalysis) -> Optional[AutonomousAction]:
        """Create generic action from goal"""
        if action_type not in self.action_templates:
            return None

        template = self.action_templates[action_type]

        # Build parameters based on goal and template
        params = {}
        for param in template['params']:
            if param == 'app' or param == 'app_name':
                params[param] = self._infer_app_from_goal(goal)
            elif param == 'task':
                params[param] = goal.description
            elif param == 'reason':
                params[param] = f"Supporting goal: {goal.description}"
            else:
                params[param] = 'auto'

        return AutonomousAction(
            action_type=action_type,
            target=params.get('app', 'system'),
            params=params,
            priority=self._goal_to_priority(goal),
            confidence=goal.confidence * 0.8,
            category=template['category'],
            reasoning=f"Goal-driven action for: {goal.description}"
        )

    def _infer_app_from_goal(self, goal: Goal) -> str:
        """Infer application from goal context"""
        app_mappings = {
            'project_completion': 'VSCode',
            'communication': 'Slack',
            'document_preparation': 'Word',
            'meeting_preparation': 'Calendar',
            'information_gathering': 'Chrome'
        }
        return app_mappings.get(goal.goal_type, 'System')

    def _goal_to_priority(self, goal: Goal) -> ActionPriority:
        """Convert goal level and confidence to action priority"""
        if goal.level == GoalLevel.IMMEDIATE:
            if goal.confidence > 0.9:
                return ActionPriority.HIGH
            else:
                return ActionPriority.MEDIUM
        elif goal.level == GoalLevel.INTERMEDIATE:
            return ActionPriority.MEDIUM
        else:
            return ActionPriority.LOW

    def _deduplicate_actions(self, actions: List[AutonomousAction]) -> List[AutonomousAction]:
        """Remove duplicate actions"""
        seen = set()
        unique_actions = []

        for action in actions:
            # Create unique key for action
            key = f"{action.action_type}_{action.target}_{action.category.value}"

            if key not in seen:
                seen.add(key)
                unique_actions.append(action)
            else:
                # Keep the one with higher confidence
                for i, existing in enumerate(unique_actions):
                    existing_key = f"{existing.action_type}_{existing.target}_{existing.category.value}"
                    if existing_key == key and action.confidence > existing.confidence:
                        unique_actions[i] = action
                        break

        return unique_actions

    async def _update_goal_progress(self, actions: List[AutonomousAction],
                                   inferred_goals: Dict[GoalLevel, List[Goal]]):
        """Update goal progress based on actions taken"""
        for action in actions:
            # Check if action is related to a goal
            if 'goal_id' in action.params:
                goal_id = action.params['goal_id']
                # Update progress (simplified - would track actual execution)
                self.goal_inference.track_goal_progress(goal_id, 0.1)

    def learn_from_feedback(self, action: AutonomousAction, success: bool, user_feedback: Optional[str] = None):
        """Learn from action execution feedback"""
        action_key = f"{action.action_type}:{action.category.value}"
        
        # Update success rates
        if action_key not in self.learned_patterns['action_success_rates']:
            self.learned_patterns['action_success_rates'][action_key] = 0.5
        
        # Moving average update
        current_rate = self.learned_patterns['action_success_rates'][action_key]
        new_rate = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
        self.learned_patterns['action_success_rates'][action_key] = new_rate
        
        # Learn from specific feedback
        if user_feedback:
            self._process_user_feedback(action, user_feedback)
        
        # Save learned patterns
        self._save_learned_patterns()
    
    def _process_user_feedback(self, action: AutonomousAction, feedback: str):
        """Process user feedback to improve future decisions"""
        feedback_lower = feedback.lower()
        
        # Adjust thresholds based on feedback
        if 'too many' in feedback_lower or 'too often' in feedback_lower:
            if action.category == ActionCategory.NOTIFICATION:
                self.thresholds['notification_action'] += 1
        elif 'too late' in feedback_lower or 'earlier' in feedback_lower:
            if action.category == ActionCategory.CALENDAR:
                self.thresholds['meeting_prep_time'] += 2
        elif 'not important' in feedback_lower or 'ignore' in feedback_lower:
            # Lower confidence for similar future actions
            self.learned_patterns['user_preferences'][action.target] = 'low_priority'
    
    def _save_learned_patterns(self):
        """Save learned patterns to persistent storage"""
        patterns_file = Path("backend/data/learned_patterns.json")
        patterns_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(patterns_file, 'w') as f:
                json.dump(self.learned_patterns, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learned patterns: {e}")

async def test_autonomous_decisions():
    """Test the autonomous decision engine"""
    engine = AutonomousDecisionEngine()
    
    # Create test windows
    test_windows = [
        WindowInfo(
            window_id=1,
            app_name="Discord",
            window_title="Discord (5 new messages)",
            bounds={"x": 0, "y": 0, "width": 800, "height": 600},
            is_focused=False,
            layer=0,
            is_visible=True,
            process_id=1001
        ),
        WindowInfo(
            window_id=2,
            app_name="Calendar",
            window_title="Meeting with Team starts in 5 minutes",
            bounds={"x": 800, "y": 0, "width": 600, "height": 400},
            is_focused=True,
            layer=0,
            is_visible=True,
            process_id=1002
        ),
        WindowInfo(
            window_id=3,
            app_name="1Password",
            window_title="1Password - API Keys",
            bounds={"x": 0, "y": 600, "width": 400, "height": 300},
            is_focused=False,
            layer=1,
            is_visible=True,
            process_id=1003
        )
    ]
    
    # Create mock workspace state
    mock_state = WorkspaceAnalysis(
        focused_task="Preparing for team meeting",
        workspace_context="Multiple apps open with notifications",
        important_notifications=["Discord (5)", "Meeting in 5 min"],
        suggestions=["Check Discord messages", "Prepare for meeting"],
        confidence=0.9
    )
    
    # Get autonomous actions
    actions = await engine.analyze_and_decide(mock_state, test_windows)
    
    print("🤖 Autonomous Decision Engine Test Results:")
    print("=" * 50)
    
    for i, action in enumerate(actions):
        print(f"\n{i+1}. {action.action_type}")
        print(f"   Target: {action.target}")
        print(f"   Priority: {action.priority.name}")
        print(f"   Confidence: {action.confidence:.2%}")
        print(f"   Category: {action.category.value}")
        print(f"   Reasoning: {action.reasoning}")
        print(f"   Requires Permission: {action.requires_permission}")
        print(f"   Params: {action.params}")

if __name__ == "__main__":
    asyncio.run(test_autonomous_decisions())