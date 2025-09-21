#!/usr/bin/env python3
"""
Proactive Communication Module for JARVIS
Handles natural, context-aware communication with progressive disclosure
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from collections import deque
from enum import Enum
import random

logger = logging.getLogger(__name__)


class CommunicationStyle(Enum):
    """Communication style preferences"""
    MINIMAL = "minimal"       # Just the facts
    BALANCED = "balanced"     # Natural but concise
    DETAILED = "detailed"     # Comprehensive information
    CONVERSATIONAL = "conversational"  # More personality


class ConversationState(Enum):
    """State of ongoing conversation"""
    INITIAL = "initial"
    FOLLOWUP = "followup"
    CLARIFYING = "clarifying"
    RESOLVED = "resolved"


class ProactiveCommunicator:
    """
    Manages natural, context-aware communication between JARVIS and the user
    Handles progressive disclosure, follow-ups, and conversation flow
    """
    
    def __init__(self, voice_callback=None, text_callback=None):
        """
        Initialize communicator with output callbacks
        
        Args:
            voice_callback: Async function to speak messages
            text_callback: Async function to display text messages
        """
        self.voice_callback = voice_callback
        self.text_callback = text_callback
        
        # Communication preferences
        self.preferences = {
            'style': CommunicationStyle.BALANCED,
            'voice_enabled': True,
            'visual_overlay': False,
            'progressive_disclosure': True,
            'personality_level': 0.7,  # 0-1, how much personality to inject
            'interruption_sensitivity': 0.5,  # 0-1, how careful about interrupting
            'detail_preference': 0.5  # 0-1, how much detail to include initially
        }
        
        # Conversation tracking
        self.active_conversations = {}  # topic -> ConversationState
        self.conversation_history = deque(maxlen=50)
        self.pending_followups = deque()
        self.last_message_time = None
        
        # Context awareness
        self.current_context = {
            'user_activity': None,
            'focus_level': 0.5,
            'time_of_day': None,
            'recent_topics': deque(maxlen=10),
            'user_mood': 'neutral'  # inferred from interactions
        }
        
        # Message templates with variations
        self.message_templates = self._initialize_message_templates()
        
    def _initialize_message_templates(self) -> Dict[str, List[str]]:
        """Initialize natural message templates with variations"""
        return {
            # Update notifications
            'update_available': [
                "I notice {app} has a new update available.",
                "There's a new update for {app} ready to install.",
                "{app} just released an update.",
                "I see {app} has an update waiting for you."
            ],
            
            # Error notifications
            'error_detected': [
                "I see an error in your {location}: {error}",
                "There's an error that popped up in {location}: {error}",
                "I noticed an error message in {location}: {error}",
                "An error just appeared in {location}: {error}"
            ],
            
            # Completion notifications
            'process_complete': [
                "Your {process} just finished successfully.",
                "Good news - {process} is complete.",
                "{process} has finished running.",
                "All done with {process}."
            ],
            
            # Contextual interjections
            'while_coding': [
                "While you're coding, ",
                "I see you're working on code - ",
                "Quick interruption from your coding - ",
                "Pardon the interruption - "
            ],
            
            # Follow-up prompts
            'offer_help': [
                "Would you like me to help with that?",
                "I can assist if you'd like.",
                "Need any help with this?",
                "Should I take care of that for you?"
            ],
            
            # Acknowledgments
            'acknowledged': [
                "Got it.",
                "Understood.",
                "I'll remember that.",
                "Noted."
            ],
            
            # Timing suggestions
            'suggest_later': [
                "I'll remind you about this when you're less busy.",
                "I'll bring this up again at a better time.",
                "Let's handle this after you finish what you're doing.",
                "I'll wait for a better moment to address this."
            ]
        }
        
    async def send_proactive_message(self, 
                                   change: 'ScreenChange',
                                   context: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a proactive message about a screen change
        
        Args:
            change: The detected change to communicate
            context: Additional context for message generation
            
        Returns:
            The message that was sent
        """
        # Update context
        if context:
            self.current_context.update(context)
            
        # Generate appropriate message
        message = await self._generate_message(change)
        
        # Apply personality and style
        message = self._apply_communication_style(message, change)
        
        # Determine delivery method
        await self._deliver_message(message, change.importance.value)
        
        # Track conversation
        self._track_conversation(change, message)
        
        # Schedule follow-up if needed
        if self.preferences['progressive_disclosure'] and change.importance.value == 'high':
            await self._schedule_followup(change)
            
        return message
        
    async def _generate_message(self, change: 'ScreenChange') -> str:
        """Generate natural message for the change"""
        # Start with suggested message from Claude
        base_message = change.suggested_message
        
        # Enhance with templates if available
        if change.category.value == 'update' and 'app' in base_message.lower():
            # Extract app name and use template
            import re
            app_match = re.search(r'(\w+)\s+has', base_message)
            if app_match:
                app_name = app_match.group(1)
                template = random.choice(self.message_templates['update_available'])
                base_message = template.format(app=app_name)
                
        # Add contextual prefix if appropriate
        if self._should_add_context_prefix(change):
            prefix = self._get_context_prefix()
            base_message = prefix + base_message.lower()
            
        return base_message
        
    def _apply_communication_style(self, message: str, change: 'ScreenChange') -> str:
        """Apply user's preferred communication style"""
        style = self.preferences['style']
        
        if style == CommunicationStyle.MINIMAL:
            # Strip to essentials
            # "I notice Cursor has a new update available." -> "Cursor update available"
            import re
            message = re.sub(r"I (notice|see|observed) ", "", message)
            message = re.sub(r"has a new ", "", message)
            message = re.sub(r"\.$", "", message)
            
        elif style == CommunicationStyle.DETAILED:
            # Add more context
            if change.confidence < 0.9:
                message += f" (confidence: {change.confidence:.0%})"
            if change.location:
                message += f" Location: {change.location}"
                
        elif style == CommunicationStyle.CONVERSATIONAL:
            # Add personality
            if self.preferences['personality_level'] > 0.7:
                if change.category.value == 'update':
                    additions = [
                        " Looks like it might have some nice improvements.",
                        " The changelog might be worth checking out.",
                        " Could be some useful fixes in there."
                    ]
                    message += random.choice(additions)
                    
        return message
        
    def _should_add_context_prefix(self, change: 'ScreenChange') -> bool:
        """Determine if we should add contextual prefix"""
        # Add prefix if interrupting focused work
        if self.current_context['focus_level'] > 0.7:
            return True
            
        # Add prefix if it's been a while since last message
        if self.last_message_time:
            time_since = (datetime.now() - self.last_message_time).seconds
            if time_since > 300:  # 5 minutes
                return True
                
        return False
        
    def _get_context_prefix(self) -> str:
        """Get appropriate contextual prefix"""
        activity = self.current_context.get('user_activity', '').lower()
        
        if 'coding' in activity:
            return random.choice(self.message_templates['while_coding'])
        elif 'writing' in activity:
            return "Quick note while you're writing - "
        elif self.current_context['focus_level'] > 0.8:
            return "Sorry to interrupt - "
        else:
            return ""
            
    async def _deliver_message(self, message: str, priority: str):
        """Deliver message through appropriate channel"""
        # Voice delivery
        if self.preferences['voice_enabled'] and self.voice_callback:
            # Adjust voice parameters based on context
            voice_params = self._get_voice_parameters(priority)
            await self.voice_callback(message, **voice_params)
            
        # Text delivery
        if self.text_callback:
            await self.text_callback(message, priority=priority)
            
        # Visual overlay if enabled
        if self.preferences['visual_overlay']:
            await self._show_visual_notification(message, priority)
            
        self.last_message_time = datetime.now()
        
    def _get_voice_parameters(self, priority: str) -> Dict[str, Any]:
        """Get voice parameters based on priority and context"""
        params = {
            'speed': 1.0,
            'volume': 1.0,
            'tone': 'normal'
        }
        
        # Adjust for priority
        if priority == 'high':
            params['tone'] = 'alert'
            params['volume'] = 1.1
        elif priority == 'low':
            params['volume'] = 0.9
            params['speed'] = 0.95
            
        # Adjust for context
        if self.current_context['focus_level'] > 0.8:
            params['volume'] *= 0.9  # Quieter when focused
            
        return params
        
    def _track_conversation(self, change: 'ScreenChange', message: str):
        """Track conversation for continuity"""
        topic = f"{change.category.value}:{change.location}"
        
        # Add to history
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'change': change,
            'message': message,
            'topic': topic,
            'response': None  # To be filled if user responds
        })
        
        # Update active conversations
        if topic not in self.active_conversations:
            self.active_conversations[topic] = ConversationState.INITIAL
            
        # Track topics
        self.current_context['recent_topics'].append(topic)
        
    async def _schedule_followup(self, change: 'ScreenChange'):
        """Schedule follow-up for progressive disclosure"""
        followup_delay = 5.0  # seconds
        
        async def send_followup():
            await asyncio.sleep(followup_delay)
            
            # Check if user engaged with initial message
            if self._user_showed_interest(change):
                followup_msg = self._generate_followup(change)
                await self._deliver_message(followup_msg, 'medium')
                
        asyncio.create_task(send_followup())
        
    def _user_showed_interest(self, change: 'ScreenChange') -> bool:
        """Determine if user showed interest in the notification"""
        # This would check for user actions like:
        # - Clicking on the notification
        # - Asking a follow-up question
        # - Taking related action
        # For now, return True for high priority
        return change.importance.value == 'high'
        
    def _generate_followup(self, change: 'ScreenChange') -> str:
        """Generate follow-up message with more details"""
        followups = {
            'update': "The update includes {details}. Would you like me to install it when you're done with your current task?",
            'error': "This error might be related to {cause}. I can help troubleshoot if you'd like.",
            'notification': "This seems important. Should I open it for you?"
        }
        
        base_followup = followups.get(change.category.value, 
                                     "Would you like more details about this?")
        
        # Add offer to help
        if random.random() > 0.5:
            help_offer = random.choice(self.message_templates['offer_help'])
            base_followup = f"{base_followup} {help_offer}"
            
        return base_followup
        
    async def _show_visual_notification(self, message: str, priority: str):
        """Show visual overlay notification"""
        # This would integrate with a visual notification system
        # For now, just log
        logger.info(f"Visual notification ({priority}): {message}")
        
    async def handle_user_response(self, response: str) -> str:
        """
        Handle user response to continue conversation
        
        Args:
            response: User's response text
            
        Returns:
            JARVIS's reply
        """
        # Find most recent relevant conversation
        recent_topic = None
        for conv in reversed(self.conversation_history):
            if conv['response'] is None and \
               (datetime.now() - conv['timestamp']).seconds < 60:
                recent_topic = conv['topic']
                conv['response'] = response
                break
                
        if not recent_topic:
            return "I'm not sure what you're referring to. Could you clarify?"
            
        # Generate contextual response
        return await self._generate_contextual_response(response, recent_topic)
        
    async def _generate_contextual_response(self, 
                                          user_response: str, 
                                          topic: str) -> str:
        """Generate response based on conversation context"""
        response_lower = user_response.lower()
        
        # Handle common responses
        if any(word in response_lower for word in ['yes', 'sure', 'okay', 'do it']):
            return await self._handle_affirmative_response(topic)
        elif any(word in response_lower for word in ['no', 'not now', 'later']):
            return await self._handle_negative_response(topic)
        elif any(word in response_lower for word in ['what', 'why', 'how', 'tell me']):
            return await self._handle_information_request(topic)
        else:
            return "I understand. Let me know if you need anything else regarding this."
            
    async def _handle_affirmative_response(self, topic: str) -> str:
        """Handle affirmative user response"""
        category = topic.split(':')[0]
        
        responses = {
            'update': "I'll help you with the update. Let me know when you're ready to proceed.",
            'error': "I'll help you resolve this error. Let me analyze it further...",
            'notification': "Opening that for you now."
        }
        
        base_response = responses.get(category, "I'll take care of that for you.")
        
        # Update conversation state
        self.active_conversations[topic] = ConversationState.FOLLOWUP
        
        return base_response
        
    async def _handle_negative_response(self, topic: str) -> str:
        """Handle negative user response"""
        responses = [
            "No problem. I'll keep monitoring.",
            "Understood. Let me know if you change your mind.",
            "Got it. I'll remind you later if it's still relevant."
        ]
        
        # Mark conversation as resolved
        self.active_conversations[topic] = ConversationState.RESOLVED
        
        return random.choice(responses)
        
    async def _handle_information_request(self, topic: str) -> str:
        """Handle request for more information"""
        # This would fetch more details about the topic
        # For now, return a placeholder
        return f"Let me get more details about {topic} for you..."
        
    def update_preferences(self, preferences: Dict[str, Any]):
        """Update communication preferences"""
        self.preferences.update(preferences)
        logger.info(f"Communication preferences updated: {preferences}")
        
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return {
            'total_messages': len(self.conversation_history),
            'active_conversations': len(self.active_conversations),
            'average_response_rate': self._calculate_response_rate(),
            'preferred_style': self.preferences['style'].value,
            'recent_topics': list(self.current_context['recent_topics'])
        }
        
    def _calculate_response_rate(self) -> float:
        """Calculate user response rate to notifications"""
        if not self.conversation_history:
            return 0.0
            
        responded = sum(1 for conv in self.conversation_history if conv['response'])
        return responded / len(self.conversation_history)