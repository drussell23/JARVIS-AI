#!/usr/bin/env python3
"""
Comprehensive Voice Integration System for JARVIS

This module provides intelligent voice announcements, natural conversation capabilities,
and voice-based approval systems for the JARVIS AI assistant. It integrates with
Claude API for dynamic natural language generation and supports multiple voice
interaction types including announcements, conversations, and approval requests.

The system features:
- Dynamic announcement generation with context awareness
- Natural conversational AI with memory and context
- Voice-based approval workflows for autonomous actions
- Intelligent model selection for optimal response generation
- Integration with JARVIS core systems (notifications, decisions, monitoring)

Example:
    >>> voice_system = VoiceIntegrationSystem(api_key="your_api_key")
    >>> await voice_system.start_voice_integration()
    >>> response = await voice_system.process_voice_command("What's my schedule?")
    >>> await voice_system.stop_voice_integration()
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import re
import random
from collections import deque, defaultdict
import anthropic
import threading
import queue
import time

# Import existing JARVIS components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Voice system imports
from engines.voice_engine import VoiceAssistant, VoiceConfig, TTSEngine, VoiceCommand
from voice.jarvis_voice import EnhancedJARVISVoiceAssistant, EnhancedJARVISPersonality
from voice.macos_voice import MacOSVoice

# Autonomy system imports
from autonomy.autonomous_decision_engine import AutonomousDecisionEngine, AutonomousAction, ActionPriority
from autonomy.notification_intelligence import NotificationIntelligence, IntelligentNotification, NotificationContext
from autonomy.contextual_understanding import ContextualUnderstandingEngine, EmotionalState, CognitiveLoad

# Vision system imports
from vision.enhanced_monitoring import EnhancedWorkspaceMonitor

logger = logging.getLogger(__name__)

class VoiceInteractionType(Enum):
    """Types of voice interactions supported by the system.
    
    Attributes:
        ANNOUNCEMENT: Simple one-way announcements
        CONVERSATION: Interactive conversational exchanges
        APPROVAL_REQUEST: Requests requiring user approval
        SYSTEM_STATUS: System status updates and reports
        NOTIFICATION_ALERT: Notification-based alerts
        PROACTIVE_SUGGESTION: AI-generated suggestions
        EMERGENCY_ALERT: High-priority emergency notifications
    """
    ANNOUNCEMENT = "announcement"
    CONVERSATION = "conversation"
    APPROVAL_REQUEST = "approval_request"
    SYSTEM_STATUS = "system_status"
    NOTIFICATION_ALERT = "notification_alert"
    PROACTIVE_SUGGESTION = "proactive_suggestion"
    EMERGENCY_ALERT = "emergency_alert"

class VoicePersonality(Enum):
    """Voice personality modes for different interaction styles.
    
    Attributes:
        PROFESSIONAL: Formal, business-like tone
        FRIENDLY: Warm, casual interaction style
        CONCISE: Brief, to-the-point responses
        DETAILED: Comprehensive, explanatory responses
        CONTEXTUAL: Adaptive personality based on context
    """
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CONCISE = "concise"
    DETAILED = "detailed"
    CONTEXTUAL = "contextual"

class ApprovalResponse(Enum):
    """User approval responses for voice-based requests.
    
    Attributes:
        APPROVED: User approved the request
        DENIED: User denied the request
        CLARIFICATION_NEEDED: User needs more information
        DEFER: User wants to decide later
        CANCEL: User wants to cancel the request
    """
    APPROVED = "approved"
    DENIED = "denied"
    CLARIFICATION_NEEDED = "clarification_needed"
    DEFER = "defer"
    CANCEL = "cancel"

@dataclass
class VoiceContext:
    """Context information for voice interactions.
    
    Attributes:
        user_emotional_state: Current emotional state of the user
        cognitive_load: User's current cognitive load level
        time_of_day: Current time period (morning, afternoon, evening)
        current_activity: User's current activity or focus
        recent_interactions: List of recent voice interactions
        environment_noise_level: Ambient noise level (0.0-1.0)
        user_availability: Whether user is available for interactions
        urgency_threshold: Minimum urgency level for announcements
    """
    user_emotional_state: EmotionalState = EmotionalState.NEUTRAL
    cognitive_load: CognitiveLoad = CognitiveLoad.MODERATE
    time_of_day: Optional[str] = None
    current_activity: Optional[str] = None
    recent_interactions: List[str] = field(default_factory=list)
    environment_noise_level: float = 0.5
    user_availability: bool = True
    urgency_threshold: float = 0.7

@dataclass
class VoiceAnnouncement:
    """Structured voice announcement with metadata and lifecycle management.
    
    Attributes:
        content: The announcement text content
        urgency: Urgency level from 0.0 (low) to 1.0 (critical)
        context: Context category for the announcement
        requires_approval: Whether user approval is needed
        related_action: Associated autonomous action if any
        expiry_time: When the announcement expires
        retry_count: Number of delivery attempts made
        max_retries: Maximum number of retry attempts
    """
    content: str
    urgency: float
    context: str
    requires_approval: bool = False
    related_action: Optional[AutonomousAction] = None
    expiry_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def is_expired(self) -> bool:
        """Check if announcement has expired.
        
        Returns:
            bool: True if announcement has passed its expiry time
        """
        if self.expiry_time:
            return datetime.now() > self.expiry_time
        return False

@dataclass
class ConversationState:
    """Current conversation state and context tracking.
    
    Attributes:
        active: Whether a conversation is currently active
        topic: Current conversation topic
        context_history: History of conversation exchanges
        last_interaction: Timestamp of last interaction
        awaiting_response: Whether system is waiting for user response
        conversation_id: Unique identifier for the conversation
        personality_mode: Current personality mode being used
    """
    active: bool = False
    topic: Optional[str] = None
    context_history: List[Dict[str, str]] = field(default_factory=list)
    last_interaction: Optional[datetime] = None
    awaiting_response: bool = False
    conversation_id: str = ""
    personality_mode: VoicePersonality = VoicePersonality.CONTEXTUAL

class VoiceAnnouncementSystem:
    """
    Intelligent voice announcement system with dynamic content generation.
    
    Handles notifications, system alerts, and proactive suggestions with
    context-aware delivery timing and natural language generation.
    
    Attributes:
        claude: Anthropic Claude API client
        voice_engine: Voice synthesis engine
        use_intelligent_selection: Whether to use intelligent model selection
        announcement_queue: Queue for pending announcements
        pending_announcements: Deque of announcements awaiting processing
        announcement_history: History of delivered announcements
        context_cache: Cache for context information
        user_preferences: User preference settings
        response_patterns: Learned response patterns
        effectiveness_scores: Effectiveness tracking for announcements
        is_active: Whether the system is currently active
        processing_task: Background processing task
    """

    def __init__(self, claude_api_key: str, voice_engine: VoiceAssistant, use_intelligent_selection: bool = True):
        """Initialize the voice announcement system.
        
        Args:
            claude_api_key: API key for Claude AI service
            voice_engine: Voice synthesis engine instance
            use_intelligent_selection: Whether to use intelligent model selection
            
        Raises:
            ValueError: If claude_api_key is invalid
        """
        self.claude = anthropic.Anthropic(api_key=claude_api_key)
        self.voice_engine = voice_engine
        self.use_intelligent_selection = use_intelligent_selection
        
        # Announcement queue and management
        self.announcement_queue = asyncio.Queue()
        self.pending_announcements = deque(maxlen=100)
        self.announcement_history = deque(maxlen=1000)
        
        # Dynamic content generation
        self.context_cache = {}
        self.user_preferences = {
            'announcement_style': 'concise',
            'urgency_threshold': 0.7,
            'quiet_hours': (22, 7),  # 10 PM to 7 AM
            'preferred_voice_personality': VoicePersonality.CONTEXTUAL
        }
        
        # Learning system
        self.response_patterns = defaultdict(list)
        self.effectiveness_scores = defaultdict(float)
        
        # State management
        self.is_active = False
        self.processing_task = None
        
    async def start_announcement_system(self) -> None:
        """Start the voice announcement system.
        
        Initializes the background processing task and begins monitoring
        the announcement queue for new items to process.
        """
        if self.is_active:
            return
            
        self.is_active = True
        self.processing_task = asyncio.create_task(self._process_announcements())
        logger.info("🔊 Voice Announcement System activated")
        
    async def stop_announcement_system(self) -> None:
        """Stop the announcement system.
        
        Cancels the background processing task and cleans up resources.
        """
        self.is_active = False
        if self.processing_task:
            self.processing_task.cancel()
        logger.info("🔊 Voice Announcement System deactivated")
        
    async def queue_announcement(self, 
                                content: str, 
                                urgency: float = 0.5,
                                context: str = "general",
                                requires_approval: bool = False,
                                related_action: Optional[AutonomousAction] = None) -> str:
        """Queue an announcement for processing.
        
        Args:
            content: The announcement text content
            urgency: Urgency level from 0.0 to 1.0
            context: Context category for the announcement
            requires_approval: Whether user approval is needed
            related_action: Associated autonomous action if any
            
        Returns:
            str: Unique announcement ID for tracking
            
        Example:
            >>> announcement_id = await system.queue_announcement(
            ...     "New message received", 
            ...     urgency=0.6, 
            ...     context="notification"
            ... )
        """
        
        # Generate unique ID
        announcement_id = f"announce_{int(time.time() * 1000)}"
        
        # Create announcement
        announcement = VoiceAnnouncement(
            content=content,
            urgency=urgency,
            context=context,
            requires_approval=requires_approval,
            related_action=related_action,
            expiry_time=datetime.now() + timedelta(minutes=30)  # 30-minute expiry
        )
        
        # Add to queue
        await self.announcement_queue.put((announcement_id, announcement))
        logger.info(f"📢 Queued announcement: {content[:50]}... (urgency: {urgency})")
        
        return announcement_id
        
    async def _process_announcements(self) -> None:
        """Main announcement processing loop.
        
        Continuously processes announcements from the queue, handling
        expiry, context checking, and delivery.
        """
        while self.is_active:
            try:
                # Get next announcement with timeout
                try:
                    announcement_id, announcement = await asyncio.wait_for(
                        self.announcement_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if expired
                if announcement.is_expired():
                    logger.debug(f"Skipping expired announcement: {announcement_id}")
                    continue
                
                # Process the announcement
                await self._process_single_announcement(announcement_id, announcement)
                
            except Exception as e:
                logger.error(f"Error processing announcements: {e}")
                await asyncio.sleep(5)
                
    async def _process_single_announcement(self, announcement_id: str, announcement: VoiceAnnouncement) -> None:
        """Process a single announcement.
        
        Args:
            announcement_id: Unique identifier for the announcement
            announcement: The announcement object to process
            
        Handles context checking, content generation, delivery, and retry logic.
        """
        try:
            # Check if should announce based on context
            if not await self._should_announce(announcement):
                logger.debug(f"Skipping announcement due to context: {announcement_id}")
                return
                
            # Generate dynamic content
            dynamic_content = await self._generate_dynamic_announcement(announcement)
            
            # Deliver announcement
            if announcement.requires_approval:
                response = await self._deliver_approval_announcement(dynamic_content, announcement)
                await self._handle_approval_response(response, announcement)
            else:
                await self._deliver_simple_announcement(dynamic_content)
                
            # Record for learning
            self._record_announcement_effectiveness(announcement_id, announcement, True)
            
        except Exception as e:
            logger.error(f"Error processing announcement {announcement_id}: {e}")
            
            # Retry logic
            if announcement.retry_count < announcement.max_retries:
                announcement.retry_count += 1
                await asyncio.sleep(5)  # Brief delay before retry
                await self.announcement_queue.put((announcement_id, announcement))
                
    async def _should_announce(self, announcement: VoiceAnnouncement) -> bool:
        """Intelligent decision on whether to announce.
        
        Args:
            announcement: The announcement to evaluate
            
        Returns:
            bool: True if the announcement should be delivered
            
        Considers quiet hours, urgency thresholds, and recent announcement history
        to avoid spam and respect user preferences.
        """
        current_time = datetime.now()
        
        # Check quiet hours
        quiet_start, quiet_end = self.user_preferences['quiet_hours']
        current_hour = current_time.hour
        
        if quiet_start <= current_hour or current_hour < quiet_end:
            # Only announce high urgency during quiet hours
            return announcement.urgency > 0.8
            
        # Check urgency threshold
        if announcement.urgency < self.user_preferences['urgency_threshold']:
            return False
            
        # Check for recent similar announcements (avoid spam)
        recent_threshold = timedelta(minutes=5)
        for hist_announcement in self.announcement_history:
            if (current_time - hist_announcement['timestamp']) < recent_threshold:
                if self._is_similar_announcement(announcement, hist_announcement['announcement']):
                    return False
                    
        return True
        
    async def _generate_dynamic_announcement_with_intelligent_selection(self, announcement: VoiceAnnouncement) -> str:
        """Generate announcement using intelligent model selection.
        
        Args:
            announcement: The announcement to generate content for
            
        Returns:
            str: Generated announcement text
            
        Raises:
            ImportError: If hybrid orchestrator is not available
            Exception: If generation fails
        """
        try:
            from backend.core.hybrid_orchestrator import HybridOrchestrator

            orchestrator = HybridOrchestrator()
            if not orchestrator.is_running:
                await orchestrator.start()

            # Build context for announcement
            context_info = await self._build_announcement_context(announcement)

            # Build rich context for model selection
            context = {
                "task_type": "voice_announcement",
                "urgency": announcement.urgency,
                "announcement_context": announcement.context,
                "voice_confidence": 1.0,
                "time_of_day": context_info,
                "requires_natural_language": True,
            }

            # Create prompt
            prompt = f"""You are JARVIS, Tony Stark's AI assistant. Generate a natural voice announcement.

Context: {announcement.context}
Original content: {announcement.content}
Urgency level: {announcement.urgency}
Current context: {context_info}

Requirements:
1. Be concise and natural (1-2 sentences)
2. Match the urgency level in tone
3. Use "Sir" appropriately but not excessively
4. Make it sound conversational, not robotic
5. Include relevant context if it helps

Examples:
- High urgency: "Sir, urgent notification from Slack. The deployment is failing and requires immediate attention."
- Medium urgency: "You have a meeting reminder. The team sync starts in 10 minutes."
- Low urgency: "New message in the general channel when you have a moment."

Generate the announcement:"""

            # Execute with intelligent model selection
            result = await orchestrator.execute_with_intelligent_model_selection(
                query=prompt,
                intent="voice_processing",
                required_capabilities={"nlp_analysis", "voice_understanding", "conversational_ai"},
                context=context,
                max_tokens=150,
                temperature=0.3,
            )

            if not result.get("success"):
                raise Exception(result.get("error", "Unknown error"))

            response = result.get("text", "").strip()
            model_used = result.get("model_used", "intelligent_selection")

            logger.info(f"✨ Announcement generated using {model_used}")
            return response

        except ImportError:
            logger.warning("Hybrid orchestrator not available, falling back to direct API")
            raise
        except Exception as e:
            logger.error(f"Error in intelligent selection: {e}")
            raise

    async def _generate_dynamic_announcement(self, announcement: VoiceAnnouncement) -> str:
        """Generate dynamic announcement content using Claude.
        
        Args:
            announcement: The announcement to generate content for
            
        Returns:
            str: Generated announcement text with natural language
            
        Uses intelligent model selection when available, falls back to direct
        Claude API calls. Includes context awareness and JARVIS personality.
        """

        # Try intelligent selection first
        if self.use_intelligent_selection:
            try:
                return await self._generate_dynamic_announcement_with_intelligent_selection(announcement)
            except Exception as e:
                logger.warning(f"Intelligent selection failed, falling back to direct API: {e}")

        # Fallback to direct API
        # Build context for Claude
        context_info = await self._build_announcement_context(announcement)

        # Create prompt for natural announcement generation
        prompt = f"""You are JARVIS, Tony Stark's AI assistant. Generate a natural voice announcement.

Context: {announcement.context}
Original content: {announcement.content}
Urgency level: {announcement.urgency}
Current context: {context_info}

Requirements:
1. Be concise and natural (1-2 sentences)
2. Match the urgency level in tone
3. Use "Sir" appropriately but not excessively
4. Make it sound conversational, not robotic
5. Include relevant context if it helps

Examples:
- High urgency: "Sir, urgent notification from Slack. The deployment is failing and requires immediate attention."
- Medium urgency: "You have a meeting reminder. The team sync starts in 10 minutes."
- Low urgency: "New message in the general channel when you have a moment."

Generate the announcement:"""

        try:
            message = await asyncio.to_thread(
                self.claude.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=150,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            return message.content[0].text.strip()

        except Exception as e:
            logger.error(f"Error generating dynamic announcement: {e}")
            # Fallback to original content
            return announcement.content
            
    async def _build_announcement_context(self, announcement: VoiceAnnouncement) -> str:
        """Build context information for announcement generation.
        
        Args:
            announcement: The announcement to build context for
            
        Returns:
            str: Formatted context string for use in prompts
            
        Gathers time, user state, and recent activity information to provide
        rich context for natural language generation.
        """
        context_parts = []
        
        # Time context
        current_time = datetime.now()
        if current_time.hour < 12:
            context_parts.append("morning")
        elif current_time.hour < 17:
            context_parts.append("afternoon")
        else:
            context_parts.append("evening")
            
        # User state context (would integrate with contextual understanding)
        context_parts.append("user appears focused")
        
        # Recent activity context
        if self.announcement_history:
            recent_count = len([a for a in self.announcement_history 
                              if (datetime.now() - a['timestamp']).seconds < 300])
            if recent_count > 2:
                context_parts.append("multiple recent notifications")
                
        return ", ".join(context_parts)
        
    async def _deliver_simple_announcement(self, content: str) -> None:
        """Deliver a simple announcement via voice.
        
        Args:
            content: The announcement text to speak
            
        Raises:
            Exception: If voice delivery fails
        """
        try:
            await self.voice_engine.speak(content)
            logger.info(f"🔊 Delivered announcement: {content}")
        except Exception as e:
            logger.error(f"Error delivering announcement: {e}")
            
    async def _deliver_approval_announcement(self, content: str, announcement: VoiceAnnouncement) -> ApprovalResponse:
        """Deliver announcement requiring approval.
        
        Args:
            content: The announcement text to speak
            announcement: The announcement object requiring approval
            
        Returns:
            ApprovalResponse: The user's approval response
            
        Raises:
            Exception: If voice delivery fails
        """
        try:
            # Add approval request to content
            approval_content = f"{content} Would you like me to proceed?"
            await self.voice_engine.speak(approval_content)
            
            # Listen for response
            # This would integrate with the voice recognition system
            # For now, return a placeholder
            return ApprovalResponse.APPROVED
            
        except Exception as e:
            logger.error(f"Error delivering approval announcement: {e}")
            return ApprovalResponse.DENIED

    def _is_similar_announcement(self, announcement1: VoiceAnnouncement, announcement2: VoiceAnnouncement) -> bool:
        """Check if two announcements are similar to avoid spam.
        
        Args:
            announcement1: First announcement to compare
            announcement2: Second announcement to compare
            
        Returns:
            bool: True if announcements are considered similar
        """
        # Simple similarity check based on content and context
        return (announcement1.context == announcement2.context and 
                announcement1.content[:50] == announcement2.content[:50])

    def _record_announcement_effectiveness(self, announcement_id: str, announcement: VoiceAnnouncement, success: bool) -> None:
        """Record announcement effectiveness for learning.
        
        Args:
            announcement_id: Unique identifier for the announcement
            announcement: The announcement object
            success: Whether the announcement was delivered successfully
        """
        self.announcement_history.append({
            'id': announcement_id,
            'announcement': announcement,
            'success': success,
            'timestamp': datetime.now()
        })

    async def _handle_approval_response(self, response: ApprovalResponse, announcement: VoiceAnnouncement) -> None:
        """Handle user response to approval request.
        
        Args:
            response: The user's approval response
            announcement: The announcement that required approval
        """
        if response == ApprovalResponse.APPROVED and announcement.related_action:
            # Execute the related action
            logger.info(f"User approved action: {announcement.related_action.action_type}")
        elif response == ApprovalResponse.DENIED:
            logger.info("User denied the approval request")

class NaturalVoiceCommunication:
    """
    Natural conversational voice system with context awareness.
    
    Handles ongoing voice conversations and voice-based approvals with
    memory, context tracking, and natural language understanding.
    
    Attributes:
        claude: Anthropic Claude API client
        voice_engine: Voice synthesis engine
        use_intelligent_selection: Whether to use intelligent model selection
        conversation_state: Current conversation state and context
        context_engine: Contextual understanding engine
        intent_patterns: Patterns for intent recognition
        response_templates: Templates for response generation
        pending_approvals: Dictionary of pending approval requests
        approval_timeout: Timeout for approval responses in seconds
        command_queue: Queue for processing voice commands
        is_processing: Whether the system is actively processing
    """
    
    def __init__(self, claude_api_key: str, voice_engine: VoiceAssistant, use_intelligent_selection: bool = True):
        """Initialize the natural voice communication system.
        
        Args:
            claude_api_key: API key for Claude AI service
            voice_engine: Voice synthesis engine instance
            use_intelligent_selection: Whether to use intelligent model selection
            
        Raises:
            ValueError: If claude_api_key is invalid
        """
        self.claude = anthropic.Anthropic(api_key=claude_api_key)
        self.voice_engine = voice_engine
        self.use_intelligent_selection = use_intelligent_selection

        # Conversation management
        self.conversation_state = ConversationState()
        self.context_engine = ContextualUnderstandingEngine(claude_api_key)
        
        # Natural language understanding
        self.intent_patterns = {}
        self.response_templates = {}
        
        # Approval system
        self.pending_approvals = {}
        self.approval_timeout = 30  # seconds
        
        # Voice command processing
        self.command_queue = asyncio.Queue()
        self.is_processing = False
        
    async def start_voice_communication(self) -> None:
        """Start the natural voice communication system.
        
        Initializes the conversation system and prepares for voice interactions.
        """
        if self.is_processing:
            return
            
        self.is_processing = True
        logger.info("🎤 Natural Voice Communication System activated")
        
    async def process_voice_command(self, command: str, confidence: float = 1.0) -> str:
        """Process a voice command naturally.
        
        Args:
            command: The voice command text to process
            confidence: Confidence level of the speech recognition (0.0-1.0)
            
        Returns:
            str: Natural language response to the command
            
        Raises:
            Exception: If command processing fails
            
        Example:
            >>> response = await comm.process_voice_command("What's the weather?", 0.9)
            >>> print(response)  # "Let me check the current weather for you, sir."
        """
        try:
            # Update conversation state
            self.conversation_state.last_interaction = datetime.now()
            
            # Add to context history
            self.conversation_state.context_history.append({
                "role": "user",
                "content": command,
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence
            })
            
            # Generate natural response
            response = await self._generate_natural_response(command, confidence)
            
            # Add response to context
            self.conversation_state.context_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Limit context history
            if len(self.conversation_state.context_history) > 20:
                self.conversation_state.context_history = self.conversation_state.context_history[-20:]
                
            return response
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            return "I apologize, sir. I encountered an error processing your request."
            
    async def _generate_natural_response_with_intelligent_selection(self, command: str, confidence: float) -> str:
        """Generate natural response using intelligent model selection.
        
        Args:
            command: The user's voice command
            confidence: Speech recognition confidence level
            
        Returns:
            str: Generated natural language response
            
        Raises:
            ImportError: If hybrid orchestrator is not available
            Exception: If response generation fails
        """
        try:
            from backend.core.hybrid_orchestrator import HybridOrchestrator

            orchestrator = HybridOrchestrator()
            if not orchestrator.is_running:
                await orchestrator.start()

            # Build context
            context_info = await self._build_conversation_context()
            recent_history = self.conversation_state.context_history[-5:]

            # Rich context for model selection
            context = {
                "task_type": "conversational_response",
                "voice_confidence": confidence,
                "conversation_active": self.conversation_state.active,
                "conversation_length": len(self.conversation_state.context_history),
                "user_patterns": recent_history,
                "requires_natural_language": True,
            }

            # Create prompt
            prompt = f"""You are JARVIS, Tony Stark's AI assistant, engaged in natural conversation.

Current context: {context_info}
Voice confidence: {confidence:.2f}
User command: "{command}"

Recent conversation:
{self._format_conversation_history(recent_history)}

Guidelines:
1. Respond naturally and conversationally
2. Be helpful and proactive
3. Ask clarifying questions if needed (especially if confidence is low)
4. Vary your responses - don't be repetitive
5. Use appropriate formality level
6. If the user seems frustrated or confused, be extra patient
7. Offer relevant suggestions when appropriate

Generate a natural response:"""

            # Execute
            result = await orchestrator.execute_with_intelligent_model_selection(
                query=prompt,
                intent="voice_processing",
                required_capabilities={"nlp_analysis", "voice_understanding", "conversational_ai"},
                context=context,
                max_tokens=300,
                temperature=0.7,
            )

            if not result.get("success"):
                raise Exception(result.get("error", "Unknown error"))

            response = result.get("text", "").strip()
            model_used = result.get("model_used", "intelligent_selection")

            logger.info(f"✨ Response generated using {model_used}")
            return response

        except ImportError:
            logger.warning("Hybrid orchestrator not available, falling back to direct API")
            raise
        except Exception as e:
            logger.error(f"Error in intelligent selection: {e}")
            raise

    async def _generate_natural_response(self, command: str, confidence: float) -> str:
        """Generate natural conversational response.
        
        Args:
            command: The user's voice command
            confidence: Speech recognition confidence level
            
        Returns:
            str: Generated natural language response
            
        Uses intelligent model selection when available, falls back to direct
        Claude API. Considers conversation history and context for coherent responses.
        """

        # Try intelligent selection first
        if self.use_intelligent_selection:
            try:
                return await self._generate_natural_response_with_intelligent_selection(command, confidence)
            except Exception as e:
                logger.warning(f"Intelligent selection failed, falling back to direct API: {e}")

        # Fall