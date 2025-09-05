#!/usr/bin/env python3
"""
Swift-Vision Bridge for JARVIS
Connects Swift's intelligent command classification with Vision's screen analysis
"""

import subprocess
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from swift_bridge.advanced_python_bridge import AdvancedSwiftBridge
from vision.proactive_vision_assistant import ProactiveVisionAssistant, NotificationEvent
from system_control.dynamic_app_controller import DynamicAppController

logger = logging.getLogger(__name__)

class SwiftVisionBridge:
    """
    Bridge between Swift intelligence and Vision capabilities
    Enables seamless collaboration between command understanding and visual analysis
    """
    
    def __init__(self):
        # Swift for intelligent command processing
        self.swift_bridge = AdvancedSwiftBridge()
        
        # Vision for screen analysis
        self.vision_assistant = ProactiveVisionAssistant()
        
        # App control
        self.app_controller = DynamicAppController()
        
        # Context mapping between Swift and Vision
        self.context_mapping = {
            'swift_to_vision': {},
            'vision_to_swift': {}
        }
        
        logger.info("Swift-Vision Bridge initialized")
    
    async def process_screen_with_intelligence(self, user_query: str) -> Dict[str, Any]:
        """
        Process screen analysis request using both Swift and Vision
        Swift understands intent, Vision executes analysis
        """
        # Step 1: Swift classifies the intent
        swift_intent = self.swift_bridge.classify_command(user_query)
        
        # Step 2: Vision analyzes based on Swift's understanding
        if swift_intent.category.value == 'vision':
            # Swift identified this as a vision command
            vision_params = self._extract_vision_params_from_swift(swift_intent)
            
            # Execute vision analysis with Swift-enhanced parameters
            vision_result = await self.vision_assistant.analyze_screen_proactively(
                vision_params.get('enhanced_query', user_query)
            )
            
            # Step 3: Combine results
            unified_result = {
                'swift_understanding': {
                    'intent': swift_intent.action,
                    'confidence': swift_intent.confidence,
                    'entities': swift_intent.entities,
                    'category': swift_intent.category.value
                },
                'vision_analysis': {
                    'description': vision_result.primary_description,
                    'notifications': [n.__dict__ for n in vision_result.notifications],
                    'available_actions': vision_result.available_actions
                },
                'unified_actions': self._generate_unified_actions(
                    swift_intent, vision_result
                )
            }
            
            return unified_result
        
        return {
            'swift_understanding': swift_intent.__dict__,
            'vision_analysis': None,
            'message': 'Not a vision-related command'
        }
    
    async def handle_app_notification_intelligently(self, 
                                                  app_name: str,
                                                  notification: NotificationEvent) -> Dict[str, Any]:
        """
        Handle app notifications using Swift's app expertise and Vision's detection
        """
        # Swift understands how to handle this app
        swift_command = f"handle {app_name} notification"
        swift_handling = self.swift_bridge.classify_command(swift_command)
        
        # Vision provides the visual context
        vision_context = {
            'notification_type': notification.notification_type,
            'visual_indicators': notification.visual_cues,
            'priority': notification.priority
        }
        
        # Determine best action based on both
        if swift_handling.confidence > 0.8:
            # Swift knows this app well
            if app_name.lower() == 'whatsapp':
                return await self._handle_whatsapp_with_swift_vision(
                    notification, swift_handling, vision_context
                )
            elif app_name.lower() in ['slack', 'discord', 'teams']:
                return await self._handle_work_chat_with_swift_vision(
                    notification, swift_handling, vision_context
                )
        
        # Generic handling
        return {
            'action': 'announce',
            'swift_confidence': swift_handling.confidence,
            'vision_priority': notification.priority,
            'suggested_response': 'Check notification'
        }
    
    async def _handle_whatsapp_with_swift_vision(self,
                                               notification: NotificationEvent,
                                               swift_handling: Any,
                                               vision_context: Dict) -> Dict[str, Any]:
        """
        Special handling for WhatsApp combining Swift and Vision
        """
        # Swift knows WhatsApp's behavior
        swift_actions = {
            'can_focus': True,
            'supports_quick_reply': True,
            'notification_style': 'personal',
            'typical_actions': ['read', 'reply', 'mark_read', 'call_back']
        }
        
        # Vision provides visual state
        vision_state = {
            'has_unread_indicator': 'badge' in str(vision_context.get('visual_indicators', {})),
            'notification_count': self._extract_notification_count(notification),
            'is_urgent': vision_context['priority'] == 'urgent'
        }
        
        # Combine for intelligent response
        response = {
            'immediate_action': 'announce_personally',
            'announcement': self._generate_personal_announcement(notification),
            'follow_up_actions': [
                'offer_to_read',
                'prepare_quick_replies',
                'monitor_for_typing'
            ],
            'quick_reply_suggestions': self._generate_whatsapp_replies(vision_state),
            'swift_confidence': swift_handling.confidence
        }
        
        return response
    
    def _extract_vision_params_from_swift(self, swift_intent: Any) -> Dict[str, Any]:
        """Extract vision-specific parameters from Swift's understanding"""
        params = {}
        
        # Swift might have identified specific areas or targets
        if swift_intent.entities:
            if 'screen_area' in swift_intent.entities:
                params['focus_area'] = swift_intent.entities['screen_area']
            
            if 'app_name' in swift_intent.entities:
                params['target_app'] = swift_intent.entities['app_name']
            
            if 'notification' in swift_intent.entities:
                params['check_notifications'] = True
        
        # Enhance the query based on Swift's understanding
        if swift_intent.action == 'describe':
            params['enhanced_query'] = f"Describe in detail what you see, focusing on {swift_intent.entities.get('target', 'the screen')}"
        elif swift_intent.action == 'check':
            params['enhanced_query'] = f"Check for {swift_intent.entities.get('target', 'updates and notifications')}"
        else:
            params['enhanced_query'] = swift_intent.raw_command
        
        return params
    
    def _generate_unified_actions(self, 
                                swift_intent: Any,
                                vision_result: Any) -> List[Dict[str, Any]]:
        """Generate actions that leverage both Swift and Vision capabilities"""
        actions = []
        
        # If notifications detected by Vision
        if vision_result.notifications:
            for notif in vision_result.notifications:
                # Swift determines how to handle each app
                app_action = self.swift_bridge.classify_command(
                    f"interact with {notif.app_name}"
                )
                
                if app_action.confidence > 0.7:
                    actions.append({
                        'type': 'app_interaction',
                        'app': notif.app_name,
                        'swift_action': app_action.action,
                        'vision_context': notif.notification_type,
                        'priority': notif.priority,
                        'executable': True
                    })
        
        # Add Swift's suggested actions
        if hasattr(swift_intent, 'suggested_actions'):
            for action in swift_intent.suggested_actions:
                actions.append({
                    'type': 'swift_suggestion',
                    'action': action,
                    'confidence': swift_intent.confidence
                })
        
        # Add Vision's available actions
        for action in vision_result.available_actions[:3]:
            actions.append({
                'type': 'vision_capability',
                'action': action,
                'requires_visual': True
            })
        
        return actions
    
    def _generate_personal_announcement(self, notification: NotificationEvent) -> str:
        """Generate personal announcement style for messaging apps"""
        templates = [
            f"You have a personal message on {notification.app_name}",
            f"{notification.app_name} message just arrived",
            f"Someone messaged you on {notification.app_name}"
        ]
        
        import random
        return random.choice(templates)
    
    def _extract_notification_count(self, notification: NotificationEvent) -> int:
        """Extract notification count from visual cues"""
        import re
        
        # Look for numbers in parentheses like (3)
        if notification.window_info and notification.window_info.window_title:
            match = re.search(r'\((\d+)\)', notification.window_info.window_title)
            if match:
                return int(match.group(1))
        
        return 1
    
    def _generate_whatsapp_replies(self, vision_state: Dict) -> List[str]:
        """Generate WhatsApp-specific quick replies based on context"""
        replies = []
        
        # Urgent vs non-urgent
        if vision_state.get('is_urgent'):
            replies.extend([
                "On it right now",
                "Give me 2 minutes",
                "Calling you back"
            ])
        else:
            replies.extend([
                "Hey! Just saw this",
                "Thanks for the message",
                "Will check and get back"
            ])
        
        # Multiple messages
        if vision_state.get('notification_count', 1) > 3:
            replies.append("Just saw all your messages, reading now")
        
        return replies
    
    async def execute_unified_action(self, action: Dict[str, Any]) -> Dict[str, bool]:
        """Execute an action using the appropriate system"""
        result = {'success': False, 'method': 'none'}
        
        if action['type'] == 'app_interaction':
            # Use Swift bridge for app control
            app_name = action['app']
            if action.get('swift_action') == 'focus':
                success = await self.app_controller.focus_app(app_name)
                result = {'success': success, 'method': 'swift_control'}
                
        elif action['type'] == 'vision_capability':
            # Use Vision for visual tasks
            if 'describe' in action['action'].lower():
                # Would trigger vision analysis
                result = {'success': True, 'method': 'vision_analysis'}
                
        elif action['type'] == 'swift_suggestion':
            # Execute Swift's suggested action
            result = {'success': True, 'method': 'swift_intelligence'}
        
        return result

async def test_bridge():
    """Test the Swift-Vision bridge"""
    print("🌉 Testing Swift-Vision Bridge")
    print("=" * 50)
    
    bridge = SwiftVisionBridge()
    
    # Test 1: Screen analysis with Swift intelligence
    print("\n1️⃣ Testing intelligent screen analysis:")
    result = await bridge.process_screen_with_intelligence(
        "What's on my screen and do I have any WhatsApp messages?"
    )
    
    print(f"\n🧠 Swift Understanding:")
    swift_data = result['swift_understanding']
    print(f"  Intent: {swift_data.get('intent')}")
    print(f"  Confidence: {swift_data.get('confidence', 0):.0%}")
    print(f"  Category: {swift_data.get('category')}")
    
    if result.get('vision_analysis'):
        print(f"\n👁️ Vision Analysis:")
        vision_data = result['vision_analysis']
        print(f"  Description: {vision_data.get('description', '')[:100]}...")
        print(f"  Notifications: {len(vision_data.get('notifications', []))}")
    
    # Test 2: WhatsApp notification handling
    print("\n2️⃣ Testing WhatsApp notification with Swift+Vision:")
    
    mock_notif = NotificationEvent(
        app_name="WhatsApp",
        notification_type="message",
        priority="high"
    )
    
    handling = await bridge.handle_app_notification_intelligently(
        "WhatsApp", mock_notif
    )
    
    print(f"\n🤝 Unified Handling:")
    print(f"  Action: {handling.get('immediate_action')}")
    print(f"  Swift Confidence: {handling.get('swift_confidence', 0):.0%}")
    print(f"  Quick Replies: {handling.get('quick_reply_suggestions', [])[:3]}")
    
    print("\n✅ Bridge test complete!")
    print("\n🎯 Key Integration Points:")
    print("• Swift classifies intent → Vision executes analysis")
    print("• Vision detects notifications → Swift handles app control")
    print("• Both systems enhance each other's capabilities")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_bridge())