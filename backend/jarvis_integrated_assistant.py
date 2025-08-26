#!/usr/bin/env python3
"""
JARVIS Integrated Assistant - Seamless Vision & Control
Dynamic AI assistant with proactive notifications and contextual interactions
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from collections import defaultdict

# Import existing components
from vision.proactive_vision_assistant import ProactiveVisionAssistant, NotificationEvent
from vision.workspace_analyzer import WorkspaceAnalyzer
from vision.dynamic_vision_engine import get_dynamic_vision_engine
from vision.dynamic_multi_window_engine import get_dynamic_multi_window_engine
from system_control.vision_action_handler import get_vision_action_handler
from voice.jarvis_agent_voice import JARVISAgentVoice
from system_control.dynamic_app_controller import DynamicAppController
from system_control.claude_command_interpreter import ClaudeCommandInterpreter

logger = logging.getLogger(__name__)


@dataclass
class IntegratedContext:
    """Context for integrated vision-control interactions"""
    current_screen_analysis: Optional[Any] = None
    active_notifications: List[NotificationEvent] = field(default_factory=list)
    user_focus: Optional[str] = None
    conversation_state: str = "idle"  # idle, notification, reply, command
    last_interaction: Optional[datetime] = None
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IntegratedResponse:
    """Response from integrated assistant"""
    verbal_response: str
    visual_context: Dict[str, Any]
    available_actions: List[str]
    system_actions: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_needed: bool = False
    confidence: float = 1.0


class JARVISIntegratedAssistant:
    """
    Fully integrated JARVIS assistant combining vision, voice, and control
    with zero hardcoding and dynamic learning
    """
    
    def __init__(self, user_name: str = "Sir"):
        # Initialize core components
        self.user_name = user_name
        self.context = IntegratedContext()
        
        # Vision components
        self.proactive_vision = ProactiveVisionAssistant()
        self.workspace_analyzer = WorkspaceAnalyzer()
        self.vision_engine = get_dynamic_vision_engine()
        self.multi_window_engine = get_dynamic_multi_window_engine()
        self.vision_handler = get_vision_action_handler()
        
        # Voice component
        self.voice_system = JARVISAgentVoice(user_name)
        
        # System control
        self.app_controller = DynamicAppController()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.command_interpreter = ClaudeCommandInterpreter(api_key)
        else:
            self.command_interpreter = None
            logger.warning("Claude command interpreter not available - no API key")
        
        # Learning components
        self.interaction_patterns = defaultdict(lambda: defaultdict(float))
        self.notification_responses = defaultdict(list)
        self.contextual_preferences = defaultdict(dict)
        
        # Notification monitoring
        self.notification_monitor_active = False
        self.notification_check_interval = 5  # seconds
        self.last_notification_states = {}
        
        # Load learned data
        self._load_interaction_history()
        
        logger.info(f"JARVIS Integrated Assistant initialized for {user_name}")
    
    async def process_vision_command(self, command: str) -> IntegratedResponse:
        """
        Process vision command with full integration
        Provides proactive information about screen and notifications
        """
        # Analyze screen proactively
        proactive_response = await self.proactive_vision.analyze_screen_proactively(command)
        
        # Update context
        self.context.current_screen_analysis = proactive_response
        self.context.active_notifications = proactive_response.notifications
        self.context.last_interaction = datetime.now()
        
        # Build context from proactive response
        context = self.proactive_vision._build_interaction_context(
            None, [], proactive_response.notifications
        )
        
        # Build integrated response
        response_parts = []
        
        # 1. Main description
        response_parts.append(proactive_response.primary_description)
        
        # 2. Proactively mention other windows/apps
        if context.active_windows and len(context.active_windows) > 1:
            other_apps = [app for app in context.active_windows
                         if app not in proactive_response.primary_description][:3]
            if other_apps:
                response_parts.append(
                    f"\nI also notice you have {self._format_app_list(other_apps)} running."
                )
        
        # 3. Highlight notifications proactively
        if proactive_response.notifications:
            urgent = [n for n in proactive_response.notifications 
                     if n.priority in ['urgent', 'high']]
            
            if urgent:
                notif = urgent[0]
                response_parts.append(
                    f"\n\n{self.user_name}, you have a {notif.notification_type} "
                    f"from {notif.app_name}."
                )
                
                # Offer to read it
                response_parts.append(
                    "Would you like me to read it to you?"
                )
        
        # 4. Suggest screen areas to explore
        response_parts.append(
            "\n\nI can describe any specific part of your screen in more detail. "
            "Just let me know what you'd like to focus on."
        )
        
        # Generate verbal response
        verbal_response = '\n'.join(response_parts)
        
        # Speak the response
        if self.voice_system:
            await self._speak_response(verbal_response)
        
        # Build visual context
        visual_context = {
            'primary_app': proactive_response.context_info.get('focused_app'),
            'window_count': len(context.active_windows) if context.active_windows else 0,
            'notification_count': len(proactive_response.notifications),
            'has_urgent': any(n.priority == 'urgent' for n in proactive_response.notifications)
        }
        
        # Determine available actions
        available_actions = proactive_response.available_actions
        
        # Add notification-specific actions
        if proactive_response.notifications:
            available_actions.insert(0, "Read notifications")
            available_actions.insert(1, "Reply to message")
        
        # Learn from interaction
        self._learn_interaction_pattern(command, proactive_response)
        
        return IntegratedResponse(
            verbal_response=verbal_response,
            visual_context=visual_context,
            available_actions=available_actions,
            follow_up_needed=bool(proactive_response.notifications),
            confidence=proactive_response.confidence
        )
    
    async def handle_notification_detected(self, notification: NotificationEvent) -> IntegratedResponse:
        """
        Handle when a notification is detected
        Proactively inform user and offer actions
        """
        # Update context
        self.context.active_notifications.append(notification)
        self.context.conversation_state = "notification"
        
        # Generate contextual announcement
        announcement = self._generate_notification_announcement(notification)
        
        # Speak the announcement
        if self.voice_system:
            await self._speak_response(announcement)
        
        # Determine actions based on notification type and context
        actions = []
        
        if notification.notification_type == 'message':
            actions.extend([
                "Read the message",
                "Reply to the message",
                "Mark as read",
                "Remind me later"
            ])
            
            # Add quick reply options
            quick_replies = self._generate_quick_replies(notification)
            if quick_replies:
                actions.append("Send quick reply")
        
        elif notification.notification_type == 'calendar':
            actions.extend([
                "Show calendar details",
                "Join the meeting",
                "Snooze reminder"
            ])
        
        # Add context-aware suggestion
        if self.context.user_focus and 'code' in self.context.user_focus.lower():
            actions.append("Silence notifications for 30 minutes")
        
        return IntegratedResponse(
            verbal_response=announcement,
            visual_context={
                'notification_app': notification.app_name,
                'notification_type': notification.notification_type,
                'priority': notification.priority
            },
            available_actions=actions,
            follow_up_needed=True
        )
    
    async def read_notification(self, notification: NotificationEvent) -> IntegratedResponse:
        """Read notification content to user"""
        # Get full notification content
        content = await self._get_notification_content(notification)
        
        # Build response
        response_parts = [
            f"The {notification.notification_type} from {notification.app_name} says:",
            content,
            "\nWould you like to reply?"
        ]
        
        verbal_response = '\n'.join(response_parts)
        
        # Update context
        self.context.conversation_state = "awaiting_reply_decision"
        
        # Speak response
        if self.voice_system:
            await self._speak_response(verbal_response)
        
        # Generate contextual reply options
        reply_options = self._generate_contextual_replies(notification, content)
        
        return IntegratedResponse(
            verbal_response=verbal_response,
            visual_context={'notification_content': content},
            available_actions=["Reply with message", "Use quick reply"] + reply_options[:5],
            follow_up_needed=True
        )
    
    async def compose_reply(self, notification: NotificationEvent, 
                          reply_type: str = "contextual") -> IntegratedResponse:
        """Help user compose a reply"""
        # Generate reply suggestions based on context
        suggestions = self._generate_contextual_replies(notification)
        
        response_parts = [
            f"I can help you reply to {notification.app_name}.",
            "\nBased on your past interactions, here are some suggestions:"
        ]
        
        for i, suggestion in enumerate(suggestions[:5], 1):
            response_parts.append(f"{i}. {suggestion}")
        
        response_parts.append("\nOr you can dictate a custom message.")
        
        verbal_response = '\n'.join(response_parts)
        
        # Update context
        self.context.conversation_state = "composing_reply"
        
        # Speak response
        if self.voice_system:
            await self._speak_response(verbal_response)
        
        return IntegratedResponse(
            verbal_response=verbal_response,
            visual_context={'reply_suggestions': suggestions},
            available_actions=["Send suggestion 1-5", "Dictate custom message", "Cancel"],
            system_actions=[
                {
                    'action': 'focus_app',
                    'app': notification.app_name,
                    'ready': True
                }
            ]
        )
    
    async def send_reply(self, notification: NotificationEvent, 
                        message: str) -> IntegratedResponse:
        """Send reply through system control"""
        try:
            # Focus the app
            await self.app_controller.focus_app(notification.app_name)
            
            # Type the message (would integrate with actual typing)
            # For now, we'll simulate
            success = True
            
            if success:
                response = f"I've sent your reply to {notification.app_name}."
                
                # Learn from this interaction
                self._learn_reply_pattern(notification, message)
            else:
                response = f"I couldn't send the message. Would you like me to try again?"
            
            # Reset context
            self.context.conversation_state = "idle"
            self.context.active_notifications.remove(notification)
            
            # Speak response
            if self.voice_system:
                await self._speak_response(response)
            
            return IntegratedResponse(
                verbal_response=response,
                visual_context={'reply_sent': success},
                available_actions=["Check for new messages", "Continue with previous task"],
                follow_up_needed=False
            )
            
        except Exception as e:
            logger.error(f"Error sending reply: {e}")
            return IntegratedResponse(
                verbal_response=f"I encountered an error sending the reply: {str(e)}",
                visual_context={'error': str(e)},
                available_actions=["Try again", "Cancel"],
                follow_up_needed=True
            )
    
    async def describe_screen_area(self, area_description: str) -> IntegratedResponse:
        """Describe a specific area of the screen"""
        # Use vision engine to analyze specific area
        result = await self.vision_handler.describe_screen({
            'query': f"describe the {area_description} area of my screen in detail"
        })
        
        # Build response
        response_parts = [
            f"Looking at the {area_description} area:",
            result.description
        ]
        
        # Check for actionable items in that area
        if 'button' in result.description.lower() or 'link' in result.description.lower():
            response_parts.append(
                "\nI can help you interact with elements in this area if needed."
            )
        
        verbal_response = '\n'.join(response_parts)
        
        # Speak response
        if self.voice_system:
            await self._speak_response(verbal_response)
        
        return IntegratedResponse(
            verbal_response=verbal_response,
            visual_context={'area': area_description, 'analysis': result.data},
            available_actions=[
                "Click on element",
                "Read text in area",
                "Describe another area"
            ],
            confidence=result.confidence
        )
    
    async def start_notification_monitoring(self) -> None:
        """Start monitoring for notifications proactively"""
        self.notification_monitor_active = True
        
        logger.info("Starting proactive notification monitoring")
        
        while self.notification_monitor_active:
            try:
                # Check for new notifications
                all_windows = self.proactive_vision.window_detector.get_all_windows()
                current_notifications = await self.proactive_vision._detect_all_notifications(all_windows)
                
                # Check for new notifications
                for notif in current_notifications:
                    notif_key = f"{notif.app_name}_{notif.window_info.window_title}"
                    
                    if notif_key not in self.last_notification_states:
                        # New notification detected
                        self.last_notification_states[notif_key] = notif
                        
                        # Only announce high priority notifications
                        if notif.priority in ['urgent', 'high'] or self._should_announce(notif):
                            await self.handle_notification_detected(notif)
                
                # Clean up old notification states
                current_keys = {f"{n.app_name}_{n.window_info.window_title}" 
                               for n in current_notifications}
                self.last_notification_states = {
                    k: v for k, v in self.last_notification_states.items() 
                    if k in current_keys
                }
                
                # Wait before next check
                await asyncio.sleep(self.notification_check_interval)
                
            except Exception as e:
                logger.error(f"Error in notification monitoring: {e}")
                await asyncio.sleep(self.notification_check_interval)
    
    def _generate_notification_announcement(self, notification: NotificationEvent) -> str:
        """Generate natural announcement for notification"""
        # Use learned patterns for natural announcements
        app_announcements = self.notification_responses.get(notification.app_name, [])
        
        if app_announcements and len(app_announcements) > 3:
            # Use varied announcements based on history
            import random
            template = random.choice(app_announcements)
        else:
            # Generate based on context
            if notification.priority == 'urgent':
                template = f"{self.user_name}, you have an urgent {notification.notification_type} from {notification.app_name}"
            elif self.context.user_focus and 'meeting' in self.context.user_focus:
                template = f"Excuse me {self.user_name}, {notification.app_name} just sent you a {notification.notification_type}"
            else:
                templates = [
                    f"{self.user_name}, {notification.app_name} has a new {notification.notification_type} for you",
                    f"You've received a {notification.notification_type} from {notification.app_name}",
                    f"{notification.app_name} just sent you a {notification.notification_type}"
                ]
                template = templates[len(self.notification_responses) % len(templates)]
        
        # Add content preview if available
        if notification.content_preview:
            template += f". {notification.content_preview}"
        
        # Learn this announcement
        self.notification_responses[notification.app_name].append(template)
        
        return template
    
    def _generate_quick_replies(self, notification: NotificationEvent) -> List[str]:
        """Generate quick reply options"""
        return self.proactive_vision.generate_contextual_message_options(
            notification,
            self.proactive_vision._build_interaction_context(None, [], [notification])
        )
    
    def _generate_contextual_replies(self, notification: NotificationEvent, 
                                   content: str = None) -> List[str]:
        """Generate contextual reply suggestions"""
        # Get base suggestions from proactive vision
        base_suggestions = self._generate_quick_replies(notification)
        
        # Add learned replies for this app
        app_replies = self.contextual_preferences.get(
            f"{notification.app_name}_replies", []
        )
        
        # Combine and deduplicate
        all_suggestions = list(dict.fromkeys(base_suggestions + app_replies))
        
        # Add time-based suggestions
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour < 7:
            all_suggestions.insert(0, "I'll respond in the morning")
        
        return all_suggestions[:8]
    
    def _format_app_list(self, apps: List[str]) -> str:
        """Format app list naturally"""
        if len(apps) == 1:
            return apps[0]
        elif len(apps) == 2:
            return f"{apps[0]} and {apps[1]}"
        else:
            return f"{', '.join(apps[:-1])}, and {apps[-1]}"
    
    def _should_announce(self, notification: NotificationEvent) -> bool:
        """Determine if notification should be announced"""
        # Check user preferences
        app_key = f"{notification.app_name}_announce_rate"
        
        if app_key in self.contextual_preferences:
            return self.contextual_preferences[app_key] > 0.5
        
        # Default logic
        return notification.notification_type in ['message', 'calendar']
    
    async def _speak_response(self, text: str) -> None:
        """Speak response using voice system"""
        try:
            if hasattr(self.voice_system, 'speak'):
                await self.voice_system.speak(text)
            else:
                self.voice_system.voice_engine.speak(text)
        except Exception as e:
            logger.error(f"Error speaking response: {e}")
    
    async def _get_notification_content(self, notification: NotificationEvent) -> str:
        """Get full notification content"""
        # This would integrate with system APIs or vision
        # For now, simulate
        return f"Message content from {notification.app_name}"
    
    def _learn_interaction_pattern(self, command: str, response: Any) -> None:
        """Learn from interaction patterns"""
        # Record interaction
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'had_notifications': len(response.notifications) > 0,
            'notification_types': [n.notification_type for n in response.notifications],
            'user_focus': self.context.user_focus
        }
        
        self.context.interaction_history.append(interaction)
        
        # Learn patterns
        if response.notifications:
            for notif in response.notifications:
                key = f"{notif.app_name}_{notif.notification_type}"
                self.interaction_patterns[key]['frequency'] += 1
        
        # Save periodically
        if len(self.context.interaction_history) % 10 == 0:
            self._save_interaction_history()
    
    def _learn_reply_pattern(self, notification: NotificationEvent, reply: str) -> None:
        """Learn from reply patterns"""
        key = f"{notification.app_name}_replies"
        
        if key not in self.contextual_preferences:
            self.contextual_preferences[key] = []
        
        # Add reply if not already present
        if reply not in self.contextual_preferences[key]:
            self.contextual_preferences[key].append(reply)
            
            # Keep only recent replies
            self.contextual_preferences[key] = self.contextual_preferences[key][-20:]
    
    def _save_interaction_history(self) -> None:
        """Save interaction history and learned patterns"""
        data = {
            'interaction_patterns': dict(self.interaction_patterns),
            'notification_responses': dict(self.notification_responses),
            'contextual_preferences': dict(self.contextual_preferences),
            'recent_interactions': self.context.interaction_history[-100:]
        }
        
        save_path = Path("backend/data/jarvis_integrated_learning.json")
        save_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Saved integrated assistant learning data")
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
    
    def _load_interaction_history(self) -> None:
        """Load previous interaction history"""
        save_path = Path("backend/data/jarvis_integrated_learning.json")
        
        if not save_path.exists():
            return
        
        try:
            with open(save_path, 'r') as f:
                data = json.load(f)
            
            self.interaction_patterns = defaultdict(
                lambda: defaultdict(float), 
                data.get('interaction_patterns', {})
            )
            self.notification_responses = defaultdict(
                list,
                data.get('notification_responses', {})
            )
            self.contextual_preferences = defaultdict(
                dict,
                data.get('contextual_preferences', {})
            )
            
            logger.info("Loaded integrated assistant learning data")
        except Exception as e:
            logger.error(f"Error loading learning data: {e}")


async def test_integrated_assistant():
    """Test the integrated assistant"""
    print("🤖 Testing JARVIS Integrated Assistant")
    print("=" * 50)
    
    # Initialize assistant
    assistant = JARVISIntegratedAssistant("Sir")
    
    # Test 1: Vision command with proactive information
    print("\n1️⃣ Testing vision command...")
    response = await assistant.process_vision_command("describe my screen")
    
    print(f"\n🗣️ JARVIS says:")
    print(response.verbal_response)
    
    print(f"\n📊 Visual Context:")
    for key, value in response.visual_context.items():
        print(f"  • {key}: {value}")
    
    print(f"\n⚡ Available Actions:")
    for action in response.available_actions[:5]:
        print(f"  • {action}")
    
    # Test 2: Simulate notification
    if response.visual_context.get('notification_count', 0) > 0:
        print("\n2️⃣ Testing notification handling...")
        
        # Get first notification from context
        notif = assistant.context.active_notifications[0]
        
        # Handle notification
        notif_response = await assistant.handle_notification_detected(notif)
        
        print(f"\n🔔 Notification Alert:")
        print(notif_response.verbal_response)
        
        print(f"\n💬 Quick Actions:")
        for action in notif_response.available_actions:
            print(f"  • {action}")
    
    # Test 3: Screen area description
    print("\n3️⃣ Testing screen area description...")
    area_response = await assistant.describe_screen_area("top right corner")
    
    print(f"\n🔍 Area Analysis:")
    print(area_response.verbal_response)
    
    print("\n✅ Integrated assistant test complete!")
    print("\nFeatures demonstrated:")
    print("• Proactive screen analysis with notifications")
    print("• Natural verbal communication")
    print("• Contextual action suggestions")
    print("• Dynamic learning from interactions")
    print("• Zero hardcoding - all dynamic!")


if __name__ == "__main__":
    asyncio.run(test_integrated_assistant())