#!/usr/bin/env python3
"""
Context Engine for JARVIS Autonomous System
Understands user context and appropriate timing for actions
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from vision.workspace_analyzer import WorkspaceAnalysis
from vision.window_detector import WindowInfo
from .autonomous_decision_engine import AutonomousAction, ActionPriority

logger = logging.getLogger(__name__)

class UserState(Enum):
    """User's current state"""
    FOCUSED = "focused"          # Deep work, minimal interruptions
    AVAILABLE = "available"      # Normal work, open to interruptions
    IN_MEETING = "in_meeting"    # In a meeting, no interruptions
    IDLE = "idle"               # Away from computer
    TRANSITIONING = "transitioning"  # Between tasks

class ActivityPattern:
    """Tracks user activity patterns"""
    def __init__(self):
        self.last_interaction = datetime.now()
        self.interaction_history = []
        self.window_switches = []
        self.idle_periods = []
        
    def record_interaction(self, interaction_type: str):
        """Record a user interaction"""
        now = datetime.now()
        time_since_last = (now - self.last_interaction).total_seconds()
        
        self.interaction_history.append({
            'timestamp': now,
            'type': interaction_type,
            'gap_seconds': time_since_last
        })
        
        # Detect idle periods
        if time_since_last > 300:  # 5 minutes
            self.idle_periods.append({
                'start': self.last_interaction,
                'end': now,
                'duration': time_since_last
            })
        
        self.last_interaction = now
        
        # Keep only recent history (last hour)
        cutoff = now - timedelta(hours=1)
        self.interaction_history = [
            i for i in self.interaction_history 
            if i['timestamp'] > cutoff
        ]
    
    def get_activity_score(self) -> float:
        """Get current activity score (0-1, higher = more active)"""
        now = datetime.now()
        
        # Time since last interaction
        idle_time = (now - self.last_interaction).total_seconds()
        if idle_time > 600:  # 10 minutes
            return 0.0
        elif idle_time > 300:  # 5 minutes
            return 0.2
        elif idle_time > 60:   # 1 minute
            return 0.5
        
        # Recent interaction frequency
        recent_interactions = len([
            i for i in self.interaction_history
            if (now - i['timestamp']).total_seconds() < 300
        ])
        
        if recent_interactions > 20:
            return 1.0  # Very active
        elif recent_interactions > 10:
            return 0.8
        elif recent_interactions > 5:
            return 0.6
        else:
            return 0.4

@dataclass
class ContextAnalysis:
    """Analysis of current user context"""
    user_state: UserState
    interruption_score: float  # 0-1, lower = less interruptible
    focus_app: Optional[str]
    meeting_probability: float
    activity_score: float
    recommended_delay: Optional[timedelta]
    reasoning: str

class ContextEngine:
    """Understands user context and appropriate timing for actions"""
    
    def __init__(self):
        self.activity_pattern = ActivityPattern()
        self.state_history = []
        self.meeting_patterns = self._load_meeting_patterns()
        
        # Focus indicators (app names that suggest deep work)
        self.focus_indicators = [
            'code', 'visual studio', 'xcode', 'intellij', 'pycharm',
            'terminal', 'console', 'jupyter', 'matlab', 'rstudio',
            'photoshop', 'illustrator', 'figma', 'sketch',
            'final cut', 'premiere', 'logic', 'ableton'
        ]
        
        # Meeting indicators
        self.meeting_indicators = [
            'zoom', 'teams', 'meet', 'webex', 'skype',
            'gotomeeting', 'bluejeans', 'whereby'
        ]
        
        # Context thresholds
        self.thresholds = {
            'focus_interruption': 0.3,    # Max interruption score during focus
            'meeting_interruption': 0.1,   # Max interruption score during meetings
            'idle_threshold': 300,         # Seconds before considering idle
            'transition_duration': 30      # Seconds to detect task transitions
        }
    
    def _load_meeting_patterns(self) -> Dict[str, List[Dict]]:
        """Load learned meeting patterns"""
        # In production, load from persistent storage
        return {
            'recurring': [],  # Recurring meeting times
            'typical_duration': 30,  # Average meeting duration in minutes
            'prep_time': 5    # Minutes before meeting to prepare
        }
    
    async def analyze_context(self, workspace_state: WorkspaceAnalysis,
                            windows: List[WindowInfo]) -> ContextAnalysis:
        """Analyze current user context"""
        # Detect user state
        user_state = await self._detect_user_state(workspace_state, windows)
        
        # Calculate interruption score
        interruption_score = await self._calculate_interruption_score(
            user_state, workspace_state, windows
        )
        
        # Detect focus application
        focus_app = self._get_focus_app(windows)
        
        # Calculate meeting probability
        meeting_prob = await self._calculate_meeting_probability(windows)
        
        # Get activity score
        activity_score = self.activity_pattern.get_activity_score()
        
        # Recommend delay if needed
        recommended_delay = self._recommend_delay(user_state, interruption_score)
        
        # Build reasoning
        reasoning = self._build_reasoning(
            user_state, interruption_score, focus_app, meeting_prob
        )
        
        analysis = ContextAnalysis(
            user_state=user_state,
            interruption_score=interruption_score,
            focus_app=focus_app,
            meeting_probability=meeting_prob,
            activity_score=activity_score,
            recommended_delay=recommended_delay,
            reasoning=reasoning
        )
        
        # Record state for history
        self._record_state(analysis)
        
        return analysis
    
    async def _detect_user_state(self, workspace_state: WorkspaceAnalysis,
                               windows: List[WindowInfo]) -> UserState:
        """Detect current user state"""
        # Check for meeting
        if await self._is_in_meeting(windows):
            return UserState.IN_MEETING
        
        # Check for idle
        if self.activity_pattern.get_activity_score() < 0.1:
            return UserState.IDLE
        
        # Check for focused work
        focused_window = next((w for w in windows if w.is_focused), None)
        if focused_window:
            app_lower = focused_window.app_name.lower()
            if any(indicator in app_lower for indicator in self.focus_indicators):
                # Check for sustained focus (no window switches)
                recent_switches = self._get_recent_window_switches()
                if len(recent_switches) < 3:  # Few switches = focused
                    return UserState.FOCUSED
        
        # Check for transitions
        if self._is_transitioning(windows):
            return UserState.TRANSITIONING
        
        # Default state
        return UserState.AVAILABLE
    
    async def _is_in_meeting(self, windows: List[WindowInfo]) -> bool:
        """Check if user is in a meeting"""
        for window in windows:
            app_lower = window.app_name.lower()
            if any(indicator in app_lower for indicator in self.meeting_indicators):
                # Additional checks for active meeting
                if window.is_focused or 'call' in window.window_title.lower():
                    return True
        
        return False
    
    def _is_transitioning(self, windows: List[WindowInfo]) -> bool:
        """Check if user is transitioning between tasks"""
        # Look for multiple window switches in short time
        recent_switches = self._get_recent_window_switches()
        
        if len(recent_switches) > 5:  # Many switches
            # Check if switches happened recently
            if recent_switches and recent_switches[-1]['timestamp'] > datetime.now() - timedelta(seconds=30):
                return True
        
        return False
    
    def _get_recent_window_switches(self) -> List[Dict]:
        """Get recent window switch events"""
        # In production, track actual window focus changes
        # For now, return mock data based on interaction history
        switches = []
        
        for i in range(1, len(self.activity_pattern.interaction_history)):
            curr = self.activity_pattern.interaction_history[i]
            prev = self.activity_pattern.interaction_history[i-1]
            
            if curr['type'] != prev['type']:
                switches.append({
                    'timestamp': curr['timestamp'],
                    'from': prev['type'],
                    'to': curr['type']
                })
        
        return switches
    
    async def _calculate_interruption_score(self, user_state: UserState,
                                          workspace_state: WorkspaceAnalysis,
                                          windows: List[WindowInfo]) -> float:
        """Calculate how interruptible the user is (0-1, lower = less interruptible)"""
        base_score = 1.0
        
        # State-based scoring
        state_scores = {
            UserState.IN_MEETING: 0.0,      # Never interrupt meetings
            UserState.FOCUSED: 0.2,          # Rarely interrupt focus time
            UserState.TRANSITIONING: 0.8,    # Good time to act
            UserState.IDLE: 1.0,            # Perfect time to act
            UserState.AVAILABLE: 0.6         # Normal interruption level
        }
        
        base_score = state_scores.get(user_state, 0.5)
        
        # Adjust based on activity
        activity_score = self.activity_pattern.get_activity_score()
        if activity_score > 0.8:  # Very active
            base_score *= 0.7
        elif activity_score < 0.2:  # Not very active
            base_score *= 1.2
        
        # Adjust based on focus app
        if self._get_focus_app(windows) in self.focus_indicators:
            base_score *= 0.5
        
        # Time-based adjustments
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 11:  # Morning focus time
            base_score *= 0.8
        elif 13 <= current_hour <= 15:  # Post-lunch
            base_score *= 1.1
        elif current_hour >= 17:  # End of day
            base_score *= 0.9
        
        # Cap between 0 and 1
        return max(0.0, min(1.0, base_score))
    
    def _get_focus_app(self, windows: List[WindowInfo]) -> Optional[str]:
        """Get the currently focused application"""
        focused = next((w for w in windows if w.is_focused), None)
        return focused.app_name if focused else None
    
    async def _calculate_meeting_probability(self, windows: List[WindowInfo]) -> float:
        """Calculate probability of being in or about to be in a meeting"""
        prob = 0.0
        
        # Check for meeting apps
        for window in windows:
            app_lower = window.app_name.lower()
            if any(indicator in app_lower for indicator in self.meeting_indicators):
                prob += 0.5
                
                # Check window title for meeting signs
                title_lower = (window.window_title or "").lower()
                if any(word in title_lower for word in ['call', 'meeting', 'conference']):
                    prob += 0.3
        
        # Check calendar patterns (simplified for now)
        current_time = datetime.now()
        if current_time.minute < 5 or current_time.minute > 55:
            # Near the hour/half-hour when meetings typically start
            prob += 0.2
        
        return min(prob, 1.0)
    
    def _recommend_delay(self, user_state: UserState, 
                        interruption_score: float) -> Optional[timedelta]:
        """Recommend delay before taking action"""
        if user_state == UserState.IN_MEETING:
            # Delay until meeting likely ends (assume 30 min average)
            return timedelta(minutes=30)
        
        elif user_state == UserState.FOCUSED and interruption_score < 0.3:
            # Delay for a pomodoro break
            return timedelta(minutes=25)
        
        elif user_state == UserState.TRANSITIONING:
            # Short delay to let transition complete
            return timedelta(seconds=30)
        
        return None
    
    def _build_reasoning(self, user_state: UserState, interruption_score: float,
                        focus_app: Optional[str], meeting_prob: float) -> str:
        """Build human-readable reasoning for context"""
        reasons = []
        
        if user_state == UserState.IN_MEETING:
            reasons.append("User is in a meeting")
        elif user_state == UserState.FOCUSED:
            reasons.append(f"User is focused on {focus_app or 'work'}")
        elif user_state == UserState.IDLE:
            reasons.append("User appears to be idle")
        
        if interruption_score < 0.3:
            reasons.append("Low interruption window")
        elif interruption_score > 0.7:
            reasons.append("Good time for actions")
        
        if meeting_prob > 0.7:
            reasons.append("Meeting likely soon")
        
        return "; ".join(reasons) if reasons else "Normal working state"
    
    def _record_state(self, analysis: ContextAnalysis):
        """Record state for historical analysis"""
        self.state_history.append({
            'timestamp': datetime.now(),
            'state': analysis.user_state.value,
            'interruption_score': analysis.interruption_score,
            'focus_app': analysis.focus_app
        })
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(hours=24)
        self.state_history = [
            s for s in self.state_history
            if s['timestamp'] > cutoff
        ]
    
    def should_act_now(self, action: AutonomousAction, context: ContextAnalysis) -> Tuple[bool, str]:
        """Determine if action should be taken now based on context"""
        # Critical actions always execute
        if action.priority == ActionPriority.CRITICAL:
            return True, "Critical priority - immediate action required"
        
        # Never interrupt meetings (unless critical)
        if context.user_state == UserState.IN_MEETING:
            return False, "User is in a meeting"
        
        # Check interruption thresholds by priority
        priority_thresholds = {
            ActionPriority.HIGH: 0.4,
            ActionPriority.MEDIUM: 0.6,
            ActionPriority.LOW: 0.8,
            ActionPriority.BACKGROUND: 0.9
        }
        
        threshold = priority_thresholds.get(action.priority, 0.7)
        
        if context.interruption_score < threshold:
            return False, f"Interruption score {context.interruption_score:.1f} below threshold {threshold:.1f}"
        
        # Special handling for focus time
        if context.user_state == UserState.FOCUSED:
            # Only allow if action supports focus
            if action.category.value in ['security', 'urgent_communication']:
                return True, "Action supports current focus"
            else:
                return False, "User is in focused work"
        
        # Good to act
        return True, f"Good timing - interruption score {context.interruption_score:.1f}"
    
    def get_best_action_time(self, action: AutonomousAction) -> Optional[datetime]:
        """Suggest best time to execute action based on patterns"""
        # Analyze historical patterns
        best_times = []
        
        # Look for similar successful actions in history
        for state in self.state_history:
            if state['interruption_score'] > 0.7:
                best_times.append(state['timestamp'].hour)
        
        if best_times:
            # Find most common hour
            from collections import Counter
            most_common_hour = Counter(best_times).most_common(1)[0][0]
            
            # Suggest next occurrence of that hour
            now = datetime.now()
            suggested = now.replace(hour=most_common_hour, minute=0, second=0)
            if suggested <= now:
                suggested += timedelta(days=1)
            
            return suggested
        
        return None

async def test_context_engine():
    """Test the context engine"""
    engine = ContextEngine()
    
    # Create test windows
    test_windows = [
        WindowInfo(
            window_id=1,
            app_name="Visual Studio Code",
            window_title="project.py - MyProject",
            bounds={"x": 0, "y": 0, "width": 1200, "height": 800},
            is_focused=True,
            layer=0,
            is_visible=True,
            process_id=1001
        ),
        WindowInfo(
            window_id=2,
            app_name="Terminal",
            window_title="npm run dev",
            bounds={"x": 1200, "y": 0, "width": 400, "height": 800},
            is_focused=False,
            layer=0,
            is_visible=True,
            process_id=1002
        )
    ]
    
    # Create mock workspace state
    mock_state = WorkspaceAnalysis(
        focused_task="Coding in Visual Studio Code",
        workspace_context="Development environment",
        important_notifications=[],
        suggestions=[],
        confidence=0.9
    )
    
    # Analyze context
    context = await engine.analyze_context(mock_state, test_windows)
    
    print("🧠 Context Engine Analysis:")
    print("=" * 50)
    print(f"User State: {context.user_state.value}")
    print(f"Interruption Score: {context.interruption_score:.1%}")
    print(f"Focus App: {context.focus_app}")
    print(f"Meeting Probability: {context.meeting_probability:.1%}")
    print(f"Activity Score: {context.activity_score:.1%}")
    print(f"Reasoning: {context.reasoning}")
    
    if context.recommended_delay:
        print(f"Recommended Delay: {context.recommended_delay}")
    
    # Test action timing
    from .autonomous_decision_engine import AutonomousAction, ActionCategory, ActionPriority
    
    test_action = AutonomousAction(
        action_type='handle_notifications',
        target='Slack',
        params={'count': 3},
        priority=ActionPriority.MEDIUM,
        confidence=0.8,
        category=ActionCategory.NOTIFICATION,
        reasoning="3 messages in Slack"
    )
    
    should_act, reason = engine.should_act_now(test_action, context)
    print(f"\nShould act now: {should_act}")
    print(f"Reason: {reason}")

if __name__ == "__main__":
    asyncio.run(test_context_engine())