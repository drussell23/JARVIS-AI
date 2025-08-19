#!/usr/bin/env python3
"""
JARVIS Agent Voice System - AI Agent with System Control
Enhanced version with macOS control capabilities
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

from voice.ml_enhanced_voice_system import MLEnhancedVoiceSystem
from voice.jarvis_personality_adapter import PersonalityAdapter
from system_control import ClaudeCommandInterpreter, CommandCategory, SafetyLevel
from chatbots.claude_chatbot import ClaudeChatbot

logger = logging.getLogger(__name__)


class JARVISAgentVoice(MLEnhancedVoiceSystem):
    """JARVIS AI Agent with system control capabilities"""
    
    def __init__(self, user_name: str = "Sir"):
        super().__init__(user_name)
        self.user_name = user_name
        self.wake_words = ["jarvis", "hey jarvis", "okay jarvis", "yo jarvis"]
        self.wake_word_variations = ["jar vis", "hey jar vis", "jarv", "j.a.r.v.i.s"]
        self.urgent_wake_words = ["jarvis emergency", "jarvis urgent"]
        
        # Initialize system control
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.api_key:
            self.command_interpreter = ClaudeCommandInterpreter(self.api_key)
            self.claude_chatbot = ClaudeChatbot(self.api_key)
            self.system_control_enabled = True
        else:
            self.system_control_enabled = False
            logger.warning("System control disabled - no API key")
            
        # Add personality adapter for compatibility
        self.personality = PersonalityAdapter(self)
        
        # Add special commands compatibility
        self.special_commands = {
            "system control": "Switch to system control mode",
            "conversation mode": "Switch to conversation mode",
            "morning routine": "Start morning routine",
            "development setup": "Start development setup",
            "check my screen": "Analyze what's on screen",
            "check for updates": "Check for software updates",
            "monitor updates": "Start monitoring for updates",
            "vision mode": "Enable screen comprehension",
            "meeting prep": "Prepare for meeting"
        }
        
        # Add voice_engine compatibility (references parent's if exists)
        if hasattr(self, 'voice_engine'):
            self.voice_engine = self.voice_engine
        else:
            # Create a mock voice engine for compatibility
            class MockVoiceEngine:
                def speak(self, text): 
                    logger.info(f"[Voice]: {text}")
            self.voice_engine = MockVoiceEngine()
            
        # Initialize vision integration if available
        self.vision_enabled = False
        self.intelligent_vision_enabled = False
        try:
            # Try to use intelligent vision first
            from vision.intelligent_vision_integration import IntelligentJARVISVision
            self.vision_integration = IntelligentJARVISVision()
            self.vision_enabled = True
            self.intelligent_vision_enabled = True
            logger.info("Intelligent vision system initialized successfully")
        except ImportError:
            # Fallback to basic vision
            try:
                from vision.screen_vision import ScreenVisionSystem, JARVISVisionIntegration
                self.vision_system = ScreenVisionSystem()
                self.vision_integration = JARVISVisionIntegration(self.vision_system)
                self.vision_enabled = True
                logger.info("Basic vision system initialized")
            except ImportError:
                logger.info("Vision system not available - install vision dependencies")
            
        # Command modes
        self.command_mode = "conversation"  # conversation, system_control, workflow
        self.pending_confirmations = {}
        
        # System control keywords
        self.system_keywords = {
            "open", "close", "launch", "quit", "switch", "show",
            "volume", "mute", "screenshot", "sleep", "wifi",
            "search", "google", "browse", "website",
            "create", "delete", "file", "folder",
            "routine", "workflow", "setup",
            "screen", "update", "monitor", "vision", "see", "check"
        }
        
        # Add agent-specific responses
        self.agent_responses = {
            "app_opened": "I've opened {app} for you, {user}.",
            "app_closed": "{app} has been closed, {user}.",
            "volume_set": "Volume adjusted to {level}%, {user}.",
            "screenshot_taken": "Screenshot captured and saved, {user}.",
            "workflow_started": "Initiating {workflow} routine, {user}.",
            "confirmation_needed": "This action requires your confirmation, {user}. Say 'confirm' to proceed or 'cancel' to abort.",
            "action_completed": "Task completed successfully, {user}.",
            "action_failed": "I apologize, {user}, but I couldn't complete that action.",
            "system_control_mode": "Switching to system control mode. I can now help you control your Mac.",
            "conversation_mode": "Returning to conversation mode, {user}."
        }
        
    async def process_voice_input(self, text: str) -> str:
        """Process voice input with system control capabilities"""
        logger.info(f"JARVISAgentVoice received: '{text}'")
        
        # Check if we need to detect wake word in text
        if not self.running:
            if not self.detect_wake_word_in_text(text):
                logger.info("No wake word detected, ignoring")
                return ""
            else:
                # Wake word detected, activate
                logger.info("Wake word detected, activating JARVIS")
                self.running = True
            
        # Check for mode switches
        if "system control" in text.lower() or "control my mac" in text.lower():
            self.command_mode = "system_control"
            return self._format_response("system_control_mode")
            
        if "conversation mode" in text.lower() or "normal mode" in text.lower():
            self.command_mode = "conversation"
            return self._format_response("conversation_mode")
            
        # Check for pending confirmations
        if self.pending_confirmations:
            return await self._handle_confirmation(text)
            
        # Detect if this is a system command
        if self._is_system_command(text):
            logger.info(f"Detected system command: {text}")
            return await self._handle_system_command(text)
            
        # Otherwise, use normal conversation processing
        logger.info(f"Processing as conversation: {text}")
        
        # Since parent doesn't have process_voice_input, handle conversation here
        if self.claude_chatbot:
            try:
                response = await self.claude_chatbot.generate_response(text)
                logger.info(f"Claude response: {response}")
                return response
            except Exception as e:
                logger.error(f"Error getting Claude response: {e}")
                return f"I apologize, {self.user_name}, but I encountered an error processing your request."
        else:
            return f"I'm sorry, {self.user_name}, but I need my API key to answer that question."
        
    def detect_wake_word_in_text(self, text: str) -> bool:
        """Detect wake word in text input"""
        text_lower = text.lower()
        
        # Check primary wake words
        for wake_word in self.wake_words:
            if wake_word in text_lower:
                return True
                
        # Check variations
        for variation in self.wake_word_variations:
            if variation in text_lower:
                return True
                
        # Check urgent wake words
        for urgent in self.urgent_wake_words:
            if urgent in text_lower:
                return True
                
        return False
        
    def _is_system_command(self, text: str) -> bool:
        """Detect if input is a system command"""
        text_lower = text.lower()
        
        # Check if in system control mode
        if self.command_mode == "system_control":
            return True
            
        # Check for vision-related phrases first (more specific)
        vision_phrases = [
            "what am i working", "what i'm working", "working on",
            "can you see", "do you see", "what's on my screen",
            "what do you see", "describe what you see",
            "analyze my screen", "look at my screen",
            "tell me what", "show me what", "what are you seeing"
        ]
        if any(phrase in text_lower for phrase in vision_phrases):
            return True
            
        # Check for system keywords
        return any(keyword in text_lower for keyword in self.system_keywords)
        
    async def _handle_system_command(self, text: str) -> str:
        """Handle system control commands"""
        if not self.system_control_enabled:
            return "System control is not available. Please configure your API key."
            
        # Check for vision commands with expanded patterns
        text_lower = text.lower()
        vision_triggers = [
            "screen", "update", "monitor", "vision", "see",
            "what am i", "what i'm", "working on", "cursor",
            "analyze", "look at", "show me", "tell me about",
            "describe", "can you see", "do you see"
        ]
        if any(trigger in text_lower for trigger in vision_triggers):
            return await self._handle_vision_command(text)
            
        try:
            # Get system context
            context = {
                "mode": self.command_mode,
                "open_apps": self.command_interpreter.controller.list_open_applications(),
                "time": datetime.now().strftime("%H:%M"),
                "user": self.user_name
            }
            
            # Interpret command using Claude
            intent = await self.command_interpreter.interpret_command(text, context)
            
            # Log the interpreted command
            logger.info(f"Command interpreted: {intent.action} on {intent.target} "
                       f"(confidence: {intent.confidence:.2f})")
            
            # Check confidence threshold
            if intent.confidence < 0.6:
                return f"I'm not sure what you want me to do, {self.user_name}. Could you please rephrase?"
                
            # Execute the command
            result = await self.command_interpreter.execute_intent(intent)
            
            # Handle results
            if result.follow_up_needed:
                # Store pending confirmation
                confirmation_id = f"confirm_{datetime.now().timestamp()}"
                self.pending_confirmations[confirmation_id] = {
                    "intent": intent,
                    "data": result.data,
                    "timestamp": datetime.now()
                }
                return self._format_response("confirmation_needed") + " " + result.message
                
            elif result.success:
                # Format success response
                if intent.category == CommandCategory.APPLICATION:
                    if intent.action == "open_app":
                        return self._format_response("app_opened", app=intent.target)
                    elif intent.action == "close_app":
                        return self._format_response("app_closed", app=intent.target)
                elif intent.category == CommandCategory.SYSTEM:
                    if intent.action == "set_volume":
                        return self._format_response("volume_set", 
                                                   level=intent.parameters.get("level", 50))
                    elif intent.action == "screenshot":
                        return self._format_response("screenshot_taken")
                elif intent.category == CommandCategory.WORKFLOW:
                    return self._format_response("workflow_started", workflow=intent.target)
                    
                # Generic success
                return self._format_response("action_completed") + " " + result.message
                
            else:
                # Handle failure
                return self._format_response("action_failed") + " " + result.message
                
        except Exception as e:
            logger.error(f"System command error: {e}")
            return f"I encountered an error, {self.user_name}. Please try again."
            
    async def _handle_confirmation(self, text: str) -> str:
        """Handle confirmation responses"""
        text_lower = text.lower()
        
        if "confirm" in text_lower or "yes" in text_lower or "proceed" in text_lower:
            # Execute the pending action
            if self.pending_confirmations:
                # Get the most recent confirmation
                conf_id = list(self.pending_confirmations.keys())[-1]
                confirmation = self.pending_confirmations[conf_id]
                
                # Check if it's a file deletion
                if confirmation["data"].get("action") == "delete":
                    path = confirmation["data"]["path"]
                    success, message = self.command_interpreter.controller.delete_file(
                        path, confirm=False
                    )
                    
                    # Clear confirmation
                    del self.pending_confirmations[conf_id]
                    
                    if success:
                        return self._format_response("action_completed") + " " + message
                    else:
                        return self._format_response("action_failed") + " " + message
                        
        elif "cancel" in text_lower or "no" in text_lower or "abort" in text_lower:
            # Cancel pending actions
            self.pending_confirmations.clear()
            return f"Action cancelled, {self.user_name}."
            
        return "Please say 'confirm' to proceed or 'cancel' to abort."
        
    async def _handle_vision_command(self, text: str) -> str:
        """Handle vision-related commands"""
        if not self.vision_enabled:
            return f"Vision capabilities are not available, {self.user_name}. Please install the required dependencies."
            
        try:
            # Use intelligent vision if available
            if self.intelligent_vision_enabled:
                # Map common queries to intelligent analysis
                text_lower = text.lower()
                
                # Handle "what am I working on" type queries
                if any(phrase in text_lower for phrase in ["what am i working", "what i'm working", "what are you seeing"]):
                    # Use Claude to analyze the screen contextually
                    from vision.screen_capture_fallback import capture_with_intelligence
                    result = capture_with_intelligence(
                        query="Analyze what the user is currently working on based on the open applications and visible content. Be specific about the applications, files, and tasks visible.",
                        use_claude=True
                    )
                    
                    if result.get("intelligence_used") and result.get("analysis"):
                        return f"Sir, {result['analysis']}"
                    elif result.get("success"):
                        return "I can see your screen, but I need the Claude API to provide intelligent analysis of what you're working on."
                    else:
                        return "I can't see your screen right now. Please ensure screen recording permission is granted."
                
                # Use the intelligent handler for other commands
                response = await self.vision_integration.handle_intelligent_command(text)
            else:
                # Use basic vision handler
                response = await self.vision_integration.handle_vision_command(text)
                
            return response
        except Exception as e:
            logger.error(f"Vision command error: {e}")
            return f"I encountered an error with the vision system, {self.user_name}."
        
    def _format_response(self, response_type: str, **kwargs) -> str:
        """Format agent responses"""
        # Try agent_responses first, then fall back to default
        template = self.agent_responses.get(response_type, "")
        
        if not template:
            # Provide a default response if not found
            template = f"Command {response_type} completed, {{user}}."
        
        # Add default user name
        kwargs["user"] = kwargs.get("user", self.user_name)
        
        return template.format(**kwargs)
        
    async def execute_workflow(self, workflow_name: str) -> str:
        """Execute predefined workflows with voice feedback"""
        
        voice_feedback = {
            "morning_routine": [
                "Starting your morning routine, {user}.",
                "Opening your email...",
                "Checking your calendar...",
                "Getting today's weather...",
                "Morning routine complete, {user}. Have a productive day!"
            ],
            "development_setup": [
                "Setting up your development environment, {user}.",
                "Launching Visual Studio Code...",
                "Opening terminal...",
                "Starting Docker...",
                "Development environment ready, {user}!"
            ],
            "meeting_prep": [
                "Preparing for your meeting, {user}.",
                "Adjusting volume...",
                "Closing distractions...",
                "Opening Zoom...",
                "You're all set for your meeting, {user}."
            ]
        }
        
        if workflow_name not in voice_feedback:
            yield f"Unknown workflow: {workflow_name}"
            return
            
        # Execute workflow with voice feedback
        feedback_messages = voice_feedback[workflow_name]
        
        for i, message in enumerate(feedback_messages):
            # Speak the message
            formatted_message = message.format(user=self.user_name)
            if i == 0:
                yield formatted_message
                
            # Execute corresponding action
            if i < len(feedback_messages) - 1:
                await asyncio.sleep(1)  # Brief pause between actions
                
        # Execute actual workflow
        success, result = await self.command_interpreter.controller.execute_workflow(
            workflow_name
        )
        
        if success:
            yield feedback_messages[-1].format(user=self.user_name)
        else:
            yield f"There was an issue with the workflow, {self.user_name}: {result}"
            
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get current agent capabilities"""
        capabilities = super().get_capabilities()
        
        if self.system_control_enabled:
            capabilities["system_control"] = {
                "applications": [
                    "Open any macOS application",
                    "Close applications",
                    "Switch between apps",
                    "List open applications"
                ],
                "files": [
                    "Create files and folders",
                    "Open documents",
                    "Search for files",
                    "Delete files (with confirmation)"
                ],
                "system": [
                    "Control volume",
                    "Take screenshots",
                    "Toggle WiFi",
                    "Sleep display"
                ],
                "web": [
                    "Open websites",
                    "Perform web searches",
                    "Research information"
                ],
                "workflows": [
                    "Morning routine",
                    "Development setup",
                    "Meeting preparation"
                ]
            }
            
        return capabilities
        
    def get_help_commands(self) -> str:
        """Get help on available commands"""
        help_text = super().get_help_commands()
        
        if self.system_control_enabled:
            help_text += """
            
System Control Commands:
- "Open [application]" - Launch any application
- "Close [application]" - Quit an application  
- "Set volume to [X]%" - Adjust system volume
- "Take a screenshot" - Capture screen
- "Search for [query]" - Web search
- "Start my morning routine" - Execute workflow
- "Switch to system control mode" - Focus on system commands
            """
            
        return help_text