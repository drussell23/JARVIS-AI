#!/usr/bin/env python3
"""
Proactive Vision Assistant for JARVIS
Integrates vision, notifications, and system control for intelligent interactions
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from collections import defaultdict

from .window_detector import WindowDetector, WindowInfo
from .dynamic_vision_engine import get_dynamic_vision_engine
from .dynamic_multi_window_engine import get_dynamic_multi_window_engine
from .workspace_analyzer import WorkspaceAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class NotificationEvent:
    """Detected notification event"""
    app_name: str
    notification_type: str  # message, alert, update, etc.
    content_preview: Optional[str] = None
    priority: str = "normal"  # low, normal, high, urgent
    timestamp: datetime = field(default_factory=datetime.now)
    window_info: Optional[WindowInfo] = None
    visual_cues: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InteractionContext:
    """Context for intelligent interactions"""
    current_task: str
    active_windows: List[str]
    recent_notifications: List[NotificationEvent]
    user_preferences: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    learned_patterns: Dict[str, float]

@dataclass
class ProactiveResponse:
    """Proactive assistant response"""
    primary_description: str
    notifications: List[NotificationEvent]
    available_actions: List[str]
    follow_up_suggestions: List[str]
    context_info: Dict[str, Any]
    confidence: float = 1.0

class ProactiveVisionAssistant:
    """
    Proactive vision assistant that intelligently communicates about
    screen content, notifications, and available actions
    """
    
    def __init__(self):
        # Core components
        self.window_detector = WindowDetector()
        self.vision_engine = get_dynamic_vision_engine()
        self.multi_window_engine = get_dynamic_multi_window_engine()
        self.workspace_analyzer = WorkspaceAnalyzer()
        
        # Notification detection
        self.notification_patterns = self._initialize_notification_patterns()
        self.last_notification_check = {}
        self.notification_history = []
        
        # Learning components
        self.user_preferences = defaultdict(float)
        self.response_patterns = defaultdict(list)
        self.interaction_history = []
        
        # Context tracking
        self.current_context = None
        self.active_notifications = []
        
        # Voice interaction settings
        self.voice_enabled = True
        self.notification_verbosity = "balanced"  # minimal, balanced, detailed
        
        # Load learned data
        self._load_learned_preferences()
        
        logger.info("Proactive Vision Assistant initialized")
    
    def _initialize_notification_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting notifications"""
        # These patterns are learned and expanded over time
        return {
            'message_indicators': [
                r'\((\d+)\)',  # (1), (2) etc
                r'•',  # Bullet indicating unread
                r'new message',
                r'unread',
                r'notification',
                r'\d+ message[s]?',
                r'typing\.{3}',
                r'is online',
                r'sent you'
            ],
            'visual_indicators': [
                'badge', 'dot', 'highlight', 'flash', 'bounce'
            ],
            'app_specific': defaultdict(list)  # Learned per app
        }
    
    async def analyze_screen_proactively(self, query: str = None) -> ProactiveResponse:
        """
        Proactively analyze screen and provide intelligent response
        with notifications and follow-up options
        """
        # Get all windows
        all_windows = self.window_detector.get_all_windows()
        
        # Analyze current workspace
        if query:
            workspace_analysis = await self.workspace_analyzer.analyze_workspace(query)
        else:
            workspace_analysis = await self.workspace_analyzer.analyze_workspace(
                "What am I looking at and what else is happening?"
            )
        
        # Detect notifications across all windows
        notifications = await self._detect_all_notifications(all_windows)
        
        # Build interaction context
        context = self._build_interaction_context(
            workspace_analysis, all_windows, notifications
        )
        
        # Generate proactive response
        response = self._generate_proactive_response(
            query, workspace_analysis, notifications, context
        )
        
        # Learn from this interaction
        self._learn_from_interaction(query, response, context)
        
        return response
    
    async def _detect_all_notifications(self, windows: List[WindowInfo]) -> List[NotificationEvent]:
        """Detect notifications across all windows"""
        notifications = []
        
        for window in windows:
            # Check window title for notification indicators
            if window.window_title:
                notification = self._detect_notification_in_window(window)
                if notification:
                    notifications.append(notification)
            
            # Check if window has visual notification cues
            visual_notification = await self._detect_visual_notifications(window)
            if visual_notification:
                notifications.append(visual_notification)
        
        # Sort by priority and timestamp
        notifications.sort(key=lambda n: (
            {'urgent': 0, 'high': 1, 'normal': 2, 'low': 3}[n.priority],
            n.timestamp
        ))
        
        return notifications
    
    def _detect_notification_in_window(self, window: WindowInfo) -> Optional[NotificationEvent]:
        """Detect notification from window metadata"""
        if not window.window_title:
            return None
        
        title_lower = window.window_title.lower()
        
        # Check for message indicators
        for pattern in self.notification_patterns['message_indicators']:
            match = re.search(pattern, window.window_title, re.IGNORECASE)
            if match:
                # Extract notification details
                notification_type = self._classify_notification_type(window, match)
                priority = self._determine_notification_priority(window, match)
                
                # Try to extract content preview
                content_preview = None
                if hasattr(match, 'group') and match.group(1):
                    content_preview = f"{match.group(1)} new items"
                
                return NotificationEvent(
                    app_name=window.app_name,
                    notification_type=notification_type,
                    content_preview=content_preview,
                    priority=priority,
                    window_info=window,
                    visual_cues={'title_indicator': match.group(0)}
                )
        
        # Check learned app-specific patterns
        app_patterns = self.notification_patterns['app_specific'].get(
            window.app_name.lower(), []
        )
        for pattern in app_patterns:
            if pattern in title_lower:
                return NotificationEvent(
                    app_name=window.app_name,
                    notification_type='app_notification',
                    priority='normal',
                    window_info=window
                )
        
        return None
    
    def _classify_notification_type(self, window: WindowInfo, match) -> str:
        """Classify the type of notification"""
        app_lower = window.app_name.lower()
        title_lower = window.window_title.lower() if window.window_title else ""
        
        # Communication apps
        if any(indicator in app_lower for indicator in 
               ['whatsapp', 'message', 'slack', 'discord', 'telegram', 'signal']):
            return 'message'
        
        # Email
        elif any(indicator in app_lower for indicator in ['mail', 'outlook', 'gmail']):
            return 'email'
        
        # Calendar/Meeting
        elif any(indicator in title_lower for indicator in 
                ['meeting', 'calendar', 'appointment', 'reminder']):
            return 'calendar'
        
        # System alerts
        elif any(indicator in title_lower for indicator in 
                ['error', 'warning', 'alert', 'critical']):
            return 'alert'
        
        # Updates
        elif any(indicator in title_lower for indicator in 
                ['update', 'available', 'download', 'install']):
            return 'update'
        
        else:
            return 'general'
    
    def _determine_notification_priority(self, window: WindowInfo, match) -> str:
        """Determine notification priority"""
        title_lower = window.window_title.lower() if window.window_title else ""
        
        # Urgent indicators
        if any(word in title_lower for word in 
               ['urgent', 'critical', 'emergency', 'important', 'asap']):
            return 'urgent'
        
        # High priority - focused window or recent activity
        elif window.is_focused:
            return 'high'
        
        # Low priority - background apps
        elif window.bounds['width'] < 400 or window.bounds['height'] < 300:
            return 'low'
        
        else:
            return 'normal'
    
    async def _detect_visual_notifications(self, window: WindowInfo) -> Optional[NotificationEvent]:
        """Detect visual notification indicators using vision analysis"""
        # This would use Claude Vision to detect visual notification badges
        # For now, we'll use heuristics
        
        # Check if window recently changed (would need window state tracking)
        # Check for typical notification positions (top-right badges, etc.)
        
        return None  # Placeholder for visual detection
    
    def _build_interaction_context(self, workspace_analysis, 
                                 windows: List[WindowInfo], 
                                 notifications: List[NotificationEvent]) -> InteractionContext:
        """Build context for intelligent interaction"""
        # Get active window names
        active_windows = [w.app_name for w in windows if w.bounds['width'] > 100]
        
        # Build conversation history from recent interactions
        recent_history = self.interaction_history[-10:] if self.interaction_history else []
        
        return InteractionContext(
            current_task=workspace_analysis.focused_task,
            active_windows=active_windows[:10],  # Top 10
            recent_notifications=notifications,
            user_preferences=dict(self.user_preferences),
            conversation_history=recent_history,
            learned_patterns=dict(self.response_patterns)
        )
    
    def _generate_proactive_response(self, query: Optional[str],
                                   workspace_analysis,
                                   notifications: List[NotificationEvent],
                                   context: InteractionContext) -> ProactiveResponse:
        """Generate intelligent proactive response"""
        # Build primary description
        description_parts = []
        
        # 1. Describe what's on screen
        description_parts.append(workspace_analysis.focused_task)
        
        # 2. Mention other available windows/apps
        if context.active_windows:
            other_apps = [app for app in context.active_windows 
                         if app not in workspace_analysis.focused_task][:3]
            if other_apps:
                description_parts.append(
                    f"\n\nI can also see you have {', '.join(other_apps)} open."
                )
        
        # 3. Highlight notifications
        if notifications:
            urgent_notifs = [n for n in notifications if n.priority in ['urgent', 'high']]
            if urgent_notifs:
                notif = urgent_notifs[0]
                description_parts.append(
                    f"\n\n📬 You have a new {notif.notification_type} "
                    f"from {notif.app_name}"
                )
                if notif.content_preview:
                    description_parts.append(f": {notif.content_preview}")
        
        # 4. Build available actions
        available_actions = self._determine_available_actions(
            notifications, context, workspace_analysis
        )
        
        # 5. Generate follow-up suggestions
        follow_up_suggestions = self._generate_follow_up_suggestions(
            query, notifications, context
        )
        
        # 6. Add context-aware information
        context_info = {
            'notification_count': len(notifications),
            'focused_app': context.active_windows[0] if context.active_windows else None,
            'workspace_confidence': workspace_analysis.confidence,
            'has_urgent': any(n.priority == 'urgent' for n in notifications)
        }
        
        return ProactiveResponse(
            primary_description='\n'.join(description_parts),
            notifications=notifications,
            available_actions=available_actions,
            follow_up_suggestions=follow_up_suggestions,
            context_info=context_info,
            confidence=workspace_analysis.confidence
        )
    
    def _determine_available_actions(self, notifications: List[NotificationEvent],
                                   context: InteractionContext,
                                   workspace_analysis) -> List[str]:
        """Determine available actions based on context"""
        actions = []
        
        # Notification actions
        if notifications:
            for notif in notifications[:3]:  # Top 3
                if notif.notification_type == 'message':
                    actions.append(f"Read {notif.app_name} message")
                    actions.append(f"Reply to {notif.app_name}")
                elif notif.notification_type == 'email':
                    actions.append(f"Open {notif.app_name} email")
                elif notif.notification_type == 'calendar':
                    actions.append("Check calendar event")
        
        # Window actions
        if len(context.active_windows) > 1:
            actions.append("Switch to another window")
            actions.append("Show all windows")
        
        # Context-specific actions
        if 'error' in workspace_analysis.workspace_context.lower():
            actions.append("Debug the error")
        
        if 'code' in workspace_analysis.focused_task.lower():
            actions.append("Run the code")
            actions.append("Save your work")
        
        # General actions
        actions.extend([
            "Describe a specific part of the screen",
            "Analyze a different window",
            "Check for updates"
        ])
        
        return actions[:8]  # Limit to 8 actions
    
    def _generate_follow_up_suggestions(self, query: Optional[str],
                                      notifications: List[NotificationEvent],
                                      context: InteractionContext) -> List[str]:
        """Generate intelligent follow-up suggestions"""
        suggestions = []
        
        # Based on notifications
        if notifications:
            notif = notifications[0]
            if notif.notification_type == 'message':
                suggestions.append(
                    f"Would you like me to read the {notif.app_name} message?"
                )
                suggestions.append(
                    f"Should I open {notif.app_name} so you can reply?"
                )
            elif notif.notification_type == 'calendar':
                suggestions.append("Should I check your calendar?")
        
        # Based on workspace
        if 'multiple' in str(context.active_windows):
            suggestions.append(
                "Would you like me to describe what's in your other windows?"
            )
        
        # Based on learned preferences
        if self.user_preferences.get('prefers_summaries', 0) > 0.7:
            suggestions.append(
                "Would you like a summary of all your open applications?"
            )
        
        # Time-based suggestions
        current_hour = datetime.now().hour
        if 17 <= current_hour <= 19:  # End of day
            suggestions.append(
                "Would you like me to summarize what you worked on today?"
            )
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    async def handle_notification_interaction(self, 
                                            notification: NotificationEvent,
                                            action: str) -> str:
        """Handle user's response to notification"""
        response_parts = []
        
        if action == "read":
            # Read notification content
            content = await self._get_notification_content(notification)
            response_parts.append(f"The message says: {content}")
            response_parts.append("\nWould you like to reply?")
            
        elif action == "reply":
            # Prepare reply context
            reply_context = await self._prepare_reply_context(notification)
            response_parts.append("I'm ready to help you reply.")
            response_parts.append(f"\nContext: {reply_context}")
            response_parts.append("\nWhat would you like to say?")
            
        elif action == "dismiss":
            response_parts.append(f"I've noted that you dismissed the {notification.app_name} notification.")
            # Learn from dismissal
            self._learn_notification_preference(notification, 'dismissed')
            
        elif action == "later":
            response_parts.append(f"I'll remind you about the {notification.app_name} notification later.")
            # Schedule reminder
            
        return '\n'.join(response_parts)
    
    async def _get_notification_content(self, notification: NotificationEvent) -> str:
        """Get full notification content using vision or system APIs"""
        # This would integrate with system APIs or use vision to read content
        # For now, return placeholder
        return f"Content from {notification.app_name}"
    
    async def _prepare_reply_context(self, notification: NotificationEvent) -> str:
        """Prepare context for replying to notification"""
        # Analyze conversation history, contact info, etc.
        context_parts = []
        
        # Check if it's a known contact
        if notification.app_name == "WhatsApp":
            context_parts.append("Personal message")
        elif notification.app_name in ["Slack", "Teams"]:
            context_parts.append("Work message")
        
        # Suggest reply tone based on context
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour > 18:
            context_parts.append("Outside work hours")
        
        return ", ".join(context_parts) if context_parts else "General message"
    
    def _learn_notification_preference(self, notification: NotificationEvent, action: str):
        """Learn user's notification preferences"""
        key = f"{notification.app_name}_{notification.notification_type}"
        
        if action == 'dismissed':
            self.user_preferences[f"{key}_dismiss_rate"] += 0.1
        elif action == 'read':
            self.user_preferences[f"{key}_read_rate"] += 0.1
        elif action == 'replied':
            self.user_preferences[f"{key}_reply_rate"] += 0.1
        
        # Adjust notification priority based on patterns
        if self.user_preferences.get(f"{key}_dismiss_rate", 0) > 0.7:
            # User often dismisses these, lower priority
            self.notification_patterns['app_specific'][notification.app_name].append(
                {'pattern': notification.window_info.window_title, 'priority_adj': -1}
            )
    
    def _learn_from_interaction(self, query: Optional[str], 
                              response: ProactiveResponse,
                              context: InteractionContext):
        """Learn from the interaction"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'notification_count': len(response.notifications),
            'actions_offered': len(response.available_actions),
            'context': {
                'active_windows': len(context.active_windows),
                'current_task': context.current_task
            }
        }
        
        self.interaction_history.append(interaction)
        
        # Save learned data periodically
        if len(self.interaction_history) % 10 == 0:
            self._save_learned_preferences()
    
    def generate_contextual_message_options(self, 
                                          notification: NotificationEvent,
                                          context: InteractionContext) -> List[str]:
        """Generate contextual message reply options"""
        options = []
        
        # Quick replies based on notification type
        if notification.notification_type == 'message':
            # Personal message options
            if 'whatsapp' in notification.app_name.lower():
                options.extend([
                    "I'll get back to you soon",
                    "Thanks for the message!",
                    "Can we talk later?",
                    "On my way",
                    "Give me 5 minutes"
                ])
            
            # Work message options
            elif any(app in notification.app_name.lower() 
                    for app in ['slack', 'teams', 'discord']):
                options.extend([
                    "Looking into it now",
                    "I'll check and update you",
                    "In a meeting, will respond soon",
                    "Thanks for the update",
                    "Can you provide more details?"
                ])
        
        # Time-based options
        current_hour = datetime.now().hour
        if 12 <= current_hour <= 13:  # Lunch time
            options.append("At lunch, will respond after")
        elif current_hour >= 18:  # Evening
            options.append("Will look at this tomorrow")
        
        # Context-based options
        if 'meeting' in context.current_task.lower():
            options.append("In a meeting, will follow up later")
        elif 'code' in context.current_task.lower():
            options.append("Deep in code, will check in a bit")
        
        # Learned preferences
        learned_replies = self.response_patterns.get(
            f"{notification.app_name}_replies", []
        )
        options.extend(learned_replies[:3])
        
        # Remove duplicates and limit
        seen = set()
        unique_options = []
        for opt in options:
            if opt not in seen:
                seen.add(opt)
                unique_options.append(opt)
        
        return unique_options[:8]  # Max 8 options
    
    def _save_learned_preferences(self):
        """Save learned preferences and patterns"""
        data = {
            'user_preferences': dict(self.user_preferences),
            'response_patterns': dict(self.response_patterns),
            'notification_patterns': {
                'app_specific': dict(self.notification_patterns['app_specific'])
            },
            'interaction_stats': {
                'total_interactions': len(self.interaction_history),
                'last_interaction': self.interaction_history[-1] if self.interaction_history else None
            }
        }
        
        save_path = Path("backend/data/proactive_assistant_learning.json")
        save_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Saved proactive assistant learning data")
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
    
    def _load_learned_preferences(self):
        """Load previously learned preferences"""
        save_path = Path("backend/data/proactive_assistant_learning.json")
        
        if not save_path.exists():
            return
        
        try:
            with open(save_path, 'r') as f:
                data = json.load(f)
            
            self.user_preferences = defaultdict(float, data.get('user_preferences', {}))
            self.response_patterns = defaultdict(list, data.get('response_patterns', {}))
            
            # Load app-specific patterns
            app_patterns = data.get('notification_patterns', {}).get('app_specific', {})
            for app, patterns in app_patterns.items():
                self.notification_patterns['app_specific'][app] = patterns
            
            logger.info("Loaded proactive assistant learning data")
        except Exception as e:
            logger.error(f"Error loading learning data: {e}")

async def test_proactive_assistant():
    """Test the proactive vision assistant"""
    print("🤖 Testing Proactive Vision Assistant")
    print("=" * 50)
    
    assistant = ProactiveVisionAssistant()
    
    # Test proactive analysis
    print("\n1️⃣ Testing proactive screen analysis...")
    response = await assistant.analyze_screen_proactively()
    
    print(f"\n📺 Screen Description:")
    print(response.primary_description)
    
    if response.notifications:
        print(f"\n🔔 Notifications ({len(response.notifications)}):")
        for notif in response.notifications:
            print(f"  • {notif.app_name}: {notif.notification_type} "
                  f"[{notif.priority}]")
            if notif.content_preview:
                print(f"    Preview: {notif.content_preview}")
    
    if response.available_actions:
        print(f"\n⚡ Available Actions:")
        for i, action in enumerate(response.available_actions, 1):
            print(f"  {i}. {action}")
    
    if response.follow_up_suggestions:
        print(f"\n💡 Suggestions:")
        for suggestion in response.follow_up_suggestions:
            print(f"  • {suggestion}")
    
    # Test notification interaction
    if response.notifications:
        print(f"\n2️⃣ Testing notification interaction...")
        notif = response.notifications[0]
        
        # Simulate reading notification
        read_response = await assistant.handle_notification_interaction(notif, "read")
        print(f"\n📖 Read Response:")
        print(read_response)
        
        # Get contextual reply options
        context = assistant._build_interaction_context(
            None, [], response.notifications
        )
        reply_options = assistant.generate_contextual_message_options(notif, context)
        
        print(f"\n💬 Suggested Replies:")
        for i, option in enumerate(reply_options, 1):
            print(f"  {i}. {option}")
    
    print("\n✅ Proactive assistant test complete!")

if __name__ == "__main__":
    asyncio.run(test_proactive_assistant())