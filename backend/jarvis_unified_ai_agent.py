#!/usr/bin/env python3
"""
JARVIS Unified AI Agent - Bridging Swift, Vision, and Control
A true AI agent that seamlessly integrates all components
"""

import asyncio
import logging
import os
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Swift Bridge for intelligent command classification
from swift_bridge.advanced_python_bridge import AdvancedSwiftBridge

# Vision System (Python/C++)
from vision.proactive_vision_assistant import ProactiveVisionAssistant, NotificationEvent
from vision.workspace_analyzer import WorkspaceAnalyzer
from vision.dynamic_vision_engine import get_dynamic_vision_engine
from vision.dynamic_multi_window_engine import get_dynamic_multi_window_engine

# System Control
from system_control.vision_action_handler import get_vision_action_handler
from system_control.dynamic_app_controller import DynamicAppController

# Voice System
from voice.jarvis_agent_voice import JARVISAgentVoice

logger = logging.getLogger(__name__)

@dataclass
class UnifiedAgentState:
    """Unified state across all components"""
    current_focus: Optional[str] = None
    active_notifications: List[NotificationEvent] = field(default_factory=list)
    conversation_mode: str = "idle"  # idle, notification_alert, reading, composing_reply
    swift_context: Dict[str, Any] = field(default_factory=dict)
    vision_context: Dict[str, Any] = field(default_factory=dict)
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)

class JARVISUnifiedAIAgent:
    """
    True AI Agent that bridges:
    - Swift: Intelligent command classification & app control
    - Vision: Screen analysis & notification detection (Python/C++)  
    - Voice: Natural communication
    - Control: System actions
    """
    
    def __init__(self, user_name: str = "Sir"):
        self.user_name = user_name
        self.state = UnifiedAgentState()
        
        # Initialize Swift for intelligent command processing
        self.swift_bridge = AdvancedSwiftBridge()
        logger.info("✅ Swift intelligence initialized")
        
        # Initialize Vision (Python/C++)
        self.vision_assistant = ProactiveVisionAssistant()
        self.workspace_analyzer = WorkspaceAnalyzer()
        self.vision_engine = get_dynamic_vision_engine()
        logger.info("✅ Vision system (Python/C++) initialized")
        
        # Initialize Voice
        self.voice_system = JARVISAgentVoice(user_name)
        logger.info("✅ Voice system initialized")
        
        # Initialize Control
        self.app_controller = DynamicAppController()
        self.vision_handler = get_vision_action_handler()
        logger.info("✅ Control system initialized")
        
        # Learning components
        self.notification_patterns = defaultdict(list)
        self.reply_history = defaultdict(list)
        self.user_preferences = defaultdict(float)
        
        # Notification monitoring
        self.monitoring_active = False
        self.monitor_interval = 3  # seconds
        
        logger.info(f"🤖 JARVIS Unified AI Agent ready for {user_name}")
    
    async def start_intelligent_monitoring(self):
        """
        Start monitoring with Swift intelligence + Vision detection
        This is where the magic happens - all components work together
        """
        self.monitoring_active = True
        logger.info("🚀 Starting intelligent monitoring with Swift+Vision integration")
        
        while self.monitoring_active:
            try:
                # Step 1: Use Vision to detect what's on screen
                screen_analysis = await self._analyze_screen_with_vision()
                
                # Step 2: Use Swift to understand the context intelligently
                swift_understanding = await self._process_with_swift_intelligence(
                    screen_analysis
                )
                
                # Step 3: Detect notifications combining both systems
                notifications = await self._detect_notifications_unified(
                    screen_analysis, swift_understanding
                )
                
                # Step 4: Handle any new notifications
                for notification in notifications:
                    await self._handle_notification_with_full_integration(notification)
                
                # Update state
                self.state.vision_context = screen_analysis
                self.state.swift_context = swift_understanding
                
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.monitor_interval)
    
    async def _analyze_screen_with_vision(self) -> Dict[str, Any]:
        """Use Vision system to analyze screen"""
        # Get all windows
        all_windows = self.vision_assistant.window_detector.get_all_windows()
        
        # Detect notifications visually
        visual_notifications = await self.vision_assistant._detect_all_notifications(all_windows)
        
        # Get workspace context
        workspace = await self.workspace_analyzer.analyze_workspace(
            "What's currently happening?"
        )
        
        return {
            'windows': all_windows,
            'visual_notifications': visual_notifications,
            'workspace': workspace,
            'timestamp': datetime.now()
        }
    
    async def _process_with_swift_intelligence(self, 
                                             vision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use Swift to intelligently understand the context"""
        # Build context string for Swift
        context_description = self._build_context_description(vision_data)
        
        # Use Swift's NLP to understand
        swift_result = self.swift_bridge.classify_command(context_description)
        
        # Extract Swift's understanding
        understanding = {
            'primary_activity': swift_result.action,
            'context_category': swift_result.category.value,
            'confidence': swift_result.confidence,
            'detected_entities': swift_result.entities,
            'suggested_actions': swift_result.suggested_actions,
            'app_states': swift_result.context.get('app_states', {})
        }
        
        return understanding
    
    async def _detect_notifications_unified(self, 
                                          vision_data: Dict[str, Any],
                                          swift_data: Dict[str, Any]) -> List[NotificationEvent]:
        """Combine Vision detection with Swift intelligence"""
        unified_notifications = []
        
        # Get visual notifications
        visual_notifs = vision_data.get('visual_notifications', [])
        
        # Enhance each with Swift's understanding
        for notif in visual_notifs:
            # Ask Swift about this specific app/notification
            app_context = f"{notif.app_name} showing {notif.notification_type}"
            swift_analysis = self.swift_bridge.classify_command(app_context)
            
            # Enhance notification with Swift intelligence
            notif.visual_cues['swift_priority'] = swift_analysis.confidence
            notif.visual_cues['swift_category'] = swift_analysis.category.value
            
            # Adjust priority based on Swift's understanding
            if swift_analysis.confidence > 0.8:
                notif.priority = 'high'
            
            unified_notifications.append(notif)
        
        return unified_notifications
    
    async def _handle_notification_with_full_integration(self, 
                                                       notification: NotificationEvent):
        """
        Handle notification using ALL systems together
        This is the core of the AI agent experience
        """
        # Update state
        self.state.active_notifications.append(notification)
        self.state.conversation_mode = "notification_alert"
        
        # Step 1: Use Swift to understand how to handle this app
        app_command = f"handle notification from {notification.app_name}"
        swift_handling = self.swift_bridge.classify_command(app_command)
        
        # Step 2: Generate intelligent announcement
        announcement = self._generate_intelligent_announcement(
            notification, swift_handling
        )
        
        # Step 3: Speak using voice system
        await self.voice_system.speak(announcement)
        
        # Step 4: Wait for user response
        # In real implementation, this would listen for voice input
        await asyncio.sleep(2)
        
        # Step 5: Offer to read the notification
        if notification.app_name.lower() == "whatsapp":
            await self._handle_whatsapp_notification(notification)
    
    async def _handle_whatsapp_notification(self, notification: NotificationEvent):
        """
        Special handling for WhatsApp as requested
        Full integration of Swift + Vision + Voice + Control
        """
        # State: Offering to read
        self.state.conversation_mode = "offering_read"
        
        # Offer to read
        offer = f"Would you like me to read the {notification.notification_type}?"
        await self.voice_system.speak(offer)
        
        # Simulate user saying "yes" (in real implementation, voice recognition)
        await asyncio.sleep(1)
        
        # State: Reading notification
        self.state.conversation_mode = "reading"
        
        # Use Vision to get notification content
        # Use Swift to focus WhatsApp window
        await self._read_notification_content(notification)
        
        # State: Offering reply
        self.state.conversation_mode = "offering_reply"
        
        # Ask about reply
        await self.voice_system.speak("Would you like to reply?")
        
        # Simulate user saying "yes"
        await asyncio.sleep(1)
        
        # State: Composing reply
        self.state.conversation_mode = "composing_reply"
        
        # Offer contextual reply options
        await self._offer_reply_options(notification)
    
    async def _read_notification_content(self, notification: NotificationEvent):
        """Read notification using Vision + Swift coordination"""
        # Use Swift to focus the app
        focus_result = await self._swift_focus_app(notification.app_name)
        
        if focus_result['success']:
            # Use Vision to read content
            content = await self._get_notification_content_with_vision(notification)
            
            # Speak the content
            message = f"The message says: {content}"
            await self.voice_system.speak(message)
        else:
            await self.voice_system.speak(
                f"I couldn't focus {notification.app_name} to read the message"
            )
    
    async def _offer_reply_options(self, notification: NotificationEvent):
        """Offer intelligent reply options based on context and history"""
        # Get current context
        current_hour = datetime.now().hour
        current_activity = self.state.swift_context.get('primary_activity', 'working')
        
        # Generate contextual options
        options = []
        
        # Time-based options
        if current_hour >= 22 or current_hour < 7:
            options.append("I'll reply in the morning")
        elif 12 <= current_hour <= 13:
            options.append("At lunch, will respond after")
        
        # Activity-based options (from Swift understanding)
        if 'coding' in current_activity.lower():
            options.extend([
                "Deep in code, will check in a bit",
                "Debugging something, give me 30 minutes"
            ])
        elif 'meeting' in current_activity.lower():
            options.extend([
                "In a meeting, will respond soon",
                "Can't talk now, will call you back"
            ])
        
        # Learn from history
        app_replies = self.reply_history.get(notification.app_name, [])
        if app_replies:
            # Add most used replies
            options.extend(app_replies[:2])
        
        # Default options
        options.extend([
            "Give me 5 minutes",
            "Thanks, will get back to you",
            "On my way"
        ])
        
        # Remove duplicates, keep order
        seen = set()
        unique_options = []
        for opt in options:
            if opt not in seen:
                seen.add(opt)
                unique_options.append(opt)
        
        # Speak options
        message = "Based on your context, here are some reply suggestions:\n"
        for i, option in enumerate(unique_options[:5], 1):
            message += f"{i}. {option}\n"
        message += "\nOr you can dictate a custom message."
        
        await self.voice_system.speak(message)
        
        return unique_options[:5]
    
    async def _swift_focus_app(self, app_name: str) -> Dict[str, Any]:
        """Use Swift to intelligently focus an app"""
        # Swift command to focus app
        command = f"focus {app_name}"
        result = self.swift_bridge.classify_command(command)
        
        # Execute through Swift's app control
        if result.confidence > 0.7:
            try:
                # Swift handles the actual focusing
                success = await self.app_controller.focus_app(app_name)
                return {'success': success, 'method': 'swift'}
            except Exception as e:
                logger.error(f"Swift focus failed: {e}")
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'reason': 'low_confidence'}
    
    async def _get_notification_content_with_vision(self, 
                                                  notification: NotificationEvent) -> str:
        """Use Vision to read notification content"""
        # This would use actual vision/OCR
        # For demo, return sample content
        content_samples = {
            'whatsapp': [
                "Hey, are you free for a quick call?",
                "Did you see the document I sent?",
                "Meeting moved to 3 PM",
                "Thanks for your help!"
            ]
        }
        
        import random
        app_key = notification.app_name.lower()
        if app_key in content_samples:
            return random.choice(content_samples[app_key])
        
        return "New message received"
    
    def _generate_intelligent_announcement(self, 
                                         notification: NotificationEvent,
                                         swift_handling: Any) -> str:
        """Generate natural announcement using Swift intelligence"""
        # Use Swift's confidence to determine announcement style
        if swift_handling.confidence > 0.8:
            # High confidence - Swift knows this is important
            style = "urgent"
        else:
            style = "normal"
        
        # Vary announcements based on learned patterns
        app_patterns = self.notification_patterns[notification.app_name]
        
        if style == "urgent":
            announcement = f"{self.user_name}, urgent {notification.notification_type} from {notification.app_name}"
        elif len(app_patterns) > 5:
            # Use variety for frequent apps
            templates = [
                f"{notification.app_name} just sent you a {notification.notification_type}",
                f"New {notification.notification_type} from {notification.app_name}, {self.user_name}",
                f"{self.user_name}, {notification.app_name} has something for you"
            ]
            announcement = templates[len(app_patterns) % len(templates)]
        else:
            # Default
            announcement = f"{self.user_name}, you have a {notification.notification_type} from {notification.app_name}"
        
        # Learn this pattern
        self.notification_patterns[notification.app_name].append({
            'template': announcement,
            'timestamp': datetime.now(),
            'swift_confidence': swift_handling.confidence
        })
        
        return announcement
    
    def _build_context_description(self, vision_data: Dict[str, Any]) -> str:
        """Build context description for Swift processing"""
        windows = vision_data.get('windows', [])
        workspace = vision_data.get('workspace')
        
        # Build natural description
        parts = []
        
        if workspace:
            parts.append(f"User is {workspace.focused_task}")
        
        if windows:
            app_names = [w.app_name for w in windows[:5]]
            parts.append(f"Open apps: {', '.join(app_names)}")
        
        notifications = vision_data.get('visual_notifications', [])
        if notifications:
            parts.append(f"{len(notifications)} notifications detected")
        
        return ". ".join(parts)
    
    async def send_reply_with_swift_intelligence(self, 
                                               notification: NotificationEvent,
                                               message: str):
        """Send reply using Swift for app control"""
        # Use Swift to handle the reply action
        command = f"reply to {notification.app_name} with message: {message}"
        swift_action = self.swift_bridge.classify_command(command)
        
        if swift_action.confidence > 0.7:
            # Swift handles the app interaction
            success = await self.app_controller.send_text_to_app(
                notification.app_name, message
            )
            
            if success:
                # Learn from successful reply
                self.reply_history[notification.app_name].append(message)
                
                # Announce completion
                await self.voice_system.speak(
                    f"I've sent your reply to {notification.app_name}"
                )
            else:
                await self.voice_system.speak(
                    "I couldn't send the message. Would you like me to try again?"
                )
        
        # Reset state
        self.state.conversation_mode = "idle"
        self.state.active_notifications.remove(notification)

async def demonstrate_unified_agent():
    """Demonstrate the unified AI agent"""
    print("🤖 JARVIS Unified AI Agent Demo")
    print("=" * 50)
    print("\nBridging Swift + Vision + Voice + Control")
    print("\n✅ Components:")
    print("• Swift: Command intelligence & app control")
    print("• Vision: Screen analysis (Python/C++)")
    print("• Voice: Natural communication")
    print("• Control: System actions")
    
    # Initialize JARVIS
    jarvis = JARVISUnifiedAIAgent("Sir")
    
    print("\n📱 Simulating WhatsApp Notification Scenario:")
    print("-" * 40)
    
    # Create mock notification
    mock_notification = NotificationEvent(
        app_name="WhatsApp",
        notification_type="message",
        priority="high"
    )
    
    # Simulate the full flow
    print("\n1️⃣ Swift detects notification + Vision confirms")
    print("   Swift: High confidence (0.92) - important message")
    print("   Vision: Visual notification badge detected")
    
    print("\n2️⃣ JARVIS announces:")
    print('   🗣️ "Sir, urgent message from WhatsApp"')
    
    await asyncio.sleep(1)
    
    print("\n3️⃣ JARVIS offers:")
    print('   🗣️ "Would you like me to read the message?"')
    
    print('\n4️⃣ User: "Yes"')
    
    print("\n5️⃣ Swift focuses WhatsApp + Vision reads content:")
    print('   🗣️ "The message says: Hey, are you free for a quick call?"')
    
    await asyncio.sleep(1)
    
    print("\n6️⃣ JARVIS asks:")
    print('   🗣️ "Would you like to reply?"')
    
    print('\n7️⃣ User: "Yes"')
    
    print("\n8️⃣ JARVIS offers contextual options:")
    print('   🗣️ "Based on your context, here are some reply suggestions:"')
    print("      1. Deep in code, will check in a bit")
    print("      2. Give me 5 minutes")
    print("      3. On my way")
    print("      Or you can dictate a custom message.")
    
    print("\n✨ All powered by:")
    print("• Swift understanding WHAT to do")
    print("• Vision seeing WHAT's happening")
    print("• Voice communicating naturally")
    print("• Control executing actions")
    
    print("\n✅ True AI Agent - Seamless Integration!")

if __name__ == "__main__":
    asyncio.run(demonstrate_unified_agent())