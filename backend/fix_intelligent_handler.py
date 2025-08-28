#!/usr/bin/env python3
"""
Fix the intelligent command handler file
"""

import os
from pathlib import Path

backend_dir = Path(__file__).parent

# Restore the proper intelligent command handler
proper_content = '''#!/usr/bin/env python3
"""
Intelligent Command Handler for JARVIS
Uses Swift classifier for intelligent command routing without hardcoding
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Import Swift bridge
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'swift_bridge'))
from python_bridge import IntelligentCommandRouter

# Import existing components
from system_control import ClaudeCommandInterpreter, CommandCategory
from chatbots.claude_chatbot import ClaudeChatbot

logger = logging.getLogger(__name__)


class IntelligentCommandHandler:
    """
    Handles commands using Swift-based intelligent classification
    No hardcoding - learns and adapts dynamically
    """
    
    def __init__(self, user_name: str = "Sir"):
        self.user_name = user_name
        self.router = IntelligentCommandRouter()
        
        # Initialize handlers
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.api_key:
            self.command_interpreter = ClaudeCommandInterpreter(self.api_key)
            self.claude_chatbot = ClaudeChatbot(self.api_key)
            self.enabled = True
        else:
            self.enabled = False
            logger.warning("Intelligent command handling disabled - no API key")
        
        # Track command history for learning
        self.command_history = []
        self.max_history = 100
        
    async def handle_command(self, text: str, context: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Intelligently handle command using Swift classification
        
        Returns:
            Tuple of (response, handler_used)
        """
        if not self.enabled:
            return "I need my API key to handle commands intelligently.", "error"
        
        try:
            # Get intelligent classification from Swift
            try:
                result = await self.router.route_command(text, context)
                if isinstance(result, tuple) and len(result) == 2:
                    handler_type, classification = result
                else:
                    # Fallback for incorrect return format
                    logger.warning(f"Unexpected router response format: {type(result)}")
                    handler_type = 'conversation'
                    classification = {'confidence': 0.5, 'type': 'conversation'}
            except ValueError as e:
                # Handle unpacking errors
                logger.warning(f"Router unpacking error: {e}")
                handler_type = 'conversation'
                classification = {'confidence': 0.5, 'type': 'conversation'}
            
            logger.info(f"Intelligent routing: '{text}' → {handler_type} "
                       f"(confidence: {classification.get('confidence', 0):.2f})")
            
            # Route to appropriate handler
            # Check for weather-related queries - route to system for Weather app workflow
            if any(word in text.lower() for word in ['weather', 'temperature', 'forecast', 'rain', 'snow', 'sunny', 'cloudy', 'hot', 'cold']):
                logger.info(f"Detected weather query, routing to system handler for Weather app workflow")
                # Override classification to ensure it goes to system handler
                classification['type'] = 'system'
                classification['confidence'] = 0.9
                return await self._handle_system_command(text, classification)
            
            
            if handler_type == 'system':
                response = await self._handle_system_command(text, classification)
            elif handler_type == 'vision':
                response = await self._handle_vision_command(text, classification)
            elif handler_type == 'conversation':
                response = await self._handle_conversation(text, classification)
            else:
                # Fallback
                response = await self._handle_fallback(text, classification)
                handler_type = 'fallback'
            
            # Record for learning
            self._record_command(text, handler_type, classification, response)
            
            return response, handler_type
            
        except Exception as e:
            logger.error(f"Error in intelligent command handling: {e}")
            return f"I encountered an error processing your command, {self.user_name}.", "error"
    
    async def _handle_system_command(self, text: str, classification: Dict[str, Any]) -> str:
        """Handle system control commands"""
        try:
            # Build context with classification insights
            context = {
                "classification": classification,
                "entities": classification.get('entities', {}),
                "intent": classification.get('intent', 'unknown'),
                "user": self.user_name,
                "timestamp": datetime.now().isoformat()
            }
            
            # Use command interpreter
            intent = await self.command_interpreter.interpret_command(text, context)
            
            # Execute if confident
            if intent.confidence >= 0.6:
                result = await self.command_interpreter.execute_intent(intent)
                
                if result.success:
                    # Learn from successful execution
                    await self.router.provide_feedback(text, 'system', True)
                    return self._format_success_response(intent, result)
                else:
                    return f"I couldn't complete that action: {result.message}"
            else:
                return f"I'm not confident about that command, {self.user_name}. Could you rephrase?"
                
        except Exception as e:
            logger.error(f"System command error: {e}")
            return f"I encountered an error with that system command, {self.user_name}."
    
    async def _handle_vision_command(self, text: str, classification: Dict[str, Any]) -> str:
        """Handle vision analysis commands"""
        try:
            # Build context with classification insights
            context = {
                "classification": classification,
                "entities": classification.get('entities', {}),
                "intent": classification.get('intent', 'unknown'),
                "user": self.user_name,
                "timestamp": datetime.now().isoformat()
            }
            
            # Use command interpreter for vision commands too
            intent = await self.command_interpreter.interpret_command(text, context)
            
            # For "can you see my screen?" type questions, ensure we get a proper response
            if any(phrase in text.lower() for phrase in ['can you see', 'do you see', 'are you able to see']):
                # This is a yes/no question about vision capability
                # Add confirmation flag to context
                context['is_vision_confirmation'] = True
                
                # Re-interpret with the updated context
                intent = await self.command_interpreter.interpret_command(text, context)
                
                # Execute the vision command to get screen content with timeout
                try:
                    result = await asyncio.wait_for(
                        self.command_interpreter.execute_intent(intent),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error("Vision command timed out after 30 seconds")
                    return f"I'm sorry {self.user_name}, the vision analysis is taking too long. Please try again."
                
                if result.success and result.message:
                    # Check if we got the unwanted "options" response
                    if "I'm not quite sure what you'd like me to do" in result.message:
                        # Force a direct screen analysis instead
                        direct_intent = await self.command_interpreter.interpret_command("describe my screen", context)
                        direct_result = await self.command_interpreter.execute_intent(direct_intent)
                        if direct_result.success and direct_result.message:
                            return f"Yes {self.user_name}, I can see your screen. {direct_result.message}"
                    else:
                        # Format as a confirmation with description
                        return f"Yes {self.user_name}, I can see your screen. {result.message}"
                else:
                    return f"I'm having trouble accessing the screen right now, {self.user_name}."
            
            # For other vision commands, execute normally
            if intent.confidence >= 0.6:
                try:
                    result = await asyncio.wait_for(
                        self.command_interpreter.execute_intent(intent),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error("Vision command timed out after 30 seconds")
                    return f"I'm sorry {self.user_name}, the vision analysis is taking too long. Please try again."
                
                if result.success:
                    # Learn from successful execution
                    await self.router.provide_feedback(text, 'vision', True)
                    return result.message  # Return the actual vision analysis
                else:
                    return f"I couldn't analyze the screen: {result.message}"
            else:
                return f"I'm not confident about that vision command, {self.user_name}. Could you rephrase?"
                
        except Exception as e:
            logger.error(f"Vision command error: {e}")
            error_msg = str(e).lower()
            
            # Provide specific error messages based on the error type
            if "503" in error_msg or "service unavailable" in error_msg:
                return f"The vision system is temporarily unavailable, {self.user_name}. Please try again in a moment."
            elif "timeout" in error_msg:
                return f"The vision analysis is taking too long, {self.user_name}. Please try again."
            elif "connection" in error_msg or "connect" in error_msg:
                return f"I'm having trouble connecting to the vision system, {self.user_name}. Please check if the service is running."
            elif "permission" in error_msg or "denied" in error_msg:
                return f"I don't have permission to access the screen, {self.user_name}. Please check the system permissions."
            else:
                return f"I encountered an error analyzing the screen: {str(e)[:100]}. Please try again, {self.user_name}."
    
    async def _handle_conversation(self, text: str, classification: Dict[str, Any]) -> str:
        """Handle conversational queries"""
        if self.claude_chatbot:
            try:
                response = await self.claude_chatbot.generate_response(text)
                return response
            except Exception as e:
                logger.error(f"Conversation error: {e}")
                return f"I'm having trouble with that conversation, {self.user_name}."
        else:
            return f"I need my Claude API to have conversations, {self.user_name}."
    
    async def _handle_fallback(self, text: str, classification: Dict[str, Any]) -> str:
        """Fallback handler for uncertain classifications"""
        confidence = classification.get('confidence', 0)
        
        if confidence < 0.3:
            return f"I'm not sure how to interpret that command, {self.user_name}. Could you rephrase?"
        else:
            # Try conversation as fallback
            return await self._handle_conversation(text, classification)
    
    def _format_success_response(self, intent: Any, result: Any) -> str:
        """Format successful command execution response"""
        if intent.action == "close_app":
            return f"{intent.target} has been closed, {self.user_name}."
        elif intent.action == "open_app":
            return f"I've opened {intent.target} for you, {self.user_name}."
        elif intent.action == "switch_app":
            return f"Switched to {intent.target}, {self.user_name}."
        elif intent.action in ["describe_screen", "analyze_window", "check_screen"]:
            # For vision commands, return the actual analysis result
            if hasattr(result, 'message') and result.message:
                return result.message
            else:
                return f"I've analyzed your screen, {self.user_name}."
        else:
            return f"Command executed successfully, {self.user_name}."
    
    def _record_command(self, text: str, handler: str, 
                       classification: Dict[str, Any], response: str):
        """Record command for learning and analysis"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'command': text,
            'handler': handler,
            'classification': classification,
            'response_preview': response[:100] + '...' if len(response) > 100 else response
        }
        
        self.command_history.append(record)
        
        # Limit history size
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)
    
    async def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about command classification"""
        stats = await self.router.classifier.get_stats()
        
        # Add local stats
        handler_counts = {}
        for record in self.command_history:
            handler = record['handler']
            handler_counts[handler] = handler_counts.get(handler, 0) + 1
        
        stats['recent_commands'] = len(self.command_history)
        stats['handler_distribution'] = handler_counts
        
        return stats
    
    async def improve_from_feedback(self, command: str, 
                                  correct_handler: str, 
                                  was_successful: bool):
        """Improve classification based on user feedback"""
        await self.router.provide_feedback(command, correct_handler, was_successful)
        logger.info(f"Learned: '{command}' should use {correct_handler} handler")
'''

# Write the proper content
handler_path = backend_dir / "voice" / "intelligent_command_handler.py"
with open(handler_path, 'w') as f:
    f.write(proper_content)

print("✅ Restored intelligent command handler")