"""
Enhanced Context Wrapper - Drop-in Replacement
=============================================

Provides a drop-in replacement for EnhancedSimpleContextHandler
that uses the new Context Intelligence System.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import our new context intelligence components
from ..core.context_manager import get_context_manager, ContextManager
from ..core.feedback_manager import get_feedback_manager, FeedbackType, FeedbackManager
from ..core.command_queue import CommandPriority
from ..integrations.jarvis_integration import get_jarvis_integration

logger = logging.getLogger(__name__)


class EnhancedContextIntelligenceHandler:
    """
    Drop-in replacement for EnhancedSimpleContextHandler that uses
    the new Context Intelligence System
    """
    
    def __init__(self, command_processor):
        """Initialize with compatibility for existing code"""
        self.command_processor = command_processor
        self.context_manager: ContextManager = get_context_manager()
        self.feedback_manager: FeedbackManager = get_feedback_manager()
        self.jarvis_integration = get_jarvis_integration()
        
        # Track execution steps for compatibility
        self.execution_steps = []
        
        # Initialize the system
        self._initialized = False
        
        # Register feedback handlers
        self._register_feedback_handlers()
        
    async def _ensure_initialized(self):
        """Ensure the context intelligence system is initialized"""
        if not self._initialized:
            await self.jarvis_integration.initialize()
            self._initialized = True
            
    def _register_feedback_handlers(self):
        """Register handlers for voice and visual feedback"""
        from ..core.feedback_manager import FeedbackChannel
        
        # Voice handler - will be sent back via WebSocket
        async def voice_handler(feedback):
            if hasattr(self, '_current_websocket') and self._current_websocket:
                await self._current_websocket.send_json({
                    "type": "voice_feedback",
                    "text": feedback.content,
                    "emotion": self._get_emotion(feedback.type),
                    "timestamp": datetime.now().isoformat()
                })
                
        self.feedback_manager.register_channel_handler(
            FeedbackChannel.VOICE, 
            voice_handler
        )
        
    def _get_emotion(self, feedback_type: FeedbackType) -> str:
        """Map feedback type to JARVIS emotion"""
        emotion_map = {
            FeedbackType.INFO: "informative",
            FeedbackType.PROGRESS: "focused",
            FeedbackType.SUCCESS: "satisfied",
            FeedbackType.WARNING: "concerned",
            FeedbackType.ERROR: "apologetic",
            FeedbackType.QUESTION: "curious"
        }
        return emotion_map.get(feedback_type, "neutral")
        
    def _add_step(self, step: str, details: Dict[str, Any] = None):
        """Add execution step for compatibility"""
        self.execution_steps.append({
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        })
        logger.info(f"[CONTEXT STEP] {step}")
        
    async def process_with_context(self, command: str, 
                                 websocket=None) -> Dict[str, Any]:
        """
        Process command with full context intelligence
        
        This method maintains the same interface as EnhancedSimpleContextHandler
        but uses the new Context Intelligence System
        """
        try:
            # Store websocket for feedback
            self._current_websocket = websocket
            
            # Ensure initialized
            await self._ensure_initialized()
            
            # Reset steps
            self.execution_steps = []
            
            logger.info(f"[CONTEXT INTELLIGENCE] ========= START PROCESSING =========")
            logger.info(f"[CONTEXT INTELLIGENCE] Command: '{command}'")
            
            # Step 1: Send initial acknowledgment
            if websocket:
                await websocket.send_json({
                    "type": "processing",
                    "message": "Processing your command...",
                    "timestamp": datetime.now().isoformat()
                })
                
            # Step 2: Process through context intelligence system
            try:
                result = await self.jarvis_integration.process_voice_command(
                    command=command,
                    voice_context={
                        "source": "voice",
                        "urgency": "normal"
                    },
                    websocket=websocket
                )
            except Exception as e:
                logger.error(f"Error in jarvis_integration.process_voice_command: {e}")
                # Fall back to direct processing
                if hasattr(self.command_processor, 'process_command'):
                    self._add_step("Falling back to direct command processing", {
                        "error": str(e)
                    })
                    result = await self.command_processor.process_command(command)
                    return {
                        "success": True,
                        "response": result.get("response", "I processed your command."),
                        "result": result,
                        "execution_steps": self.execution_steps
                    }
                else:
                    raise
            
            # Step 3: Handle the result based on status
            if result.get("status") == "queued":
                # Command was queued due to locked screen
                self._add_step("Command queued - screen locked", {
                    "command_id": result.get("command_id"),
                    "requires_unlock": True
                })
                
                # Send immediate feedback about screen being locked
                feedback_message = result.get("message", "Your screen is locked, unlocking now.")
                
                # Return immediately with the queued status message
                return {
                    "success": True,
                    "response": feedback_message,
                    "status": "queued",
                    "command_id": result.get("command_id"),
                    "execution_steps": self.execution_steps,
                    "requires_unlock": True
                }
                    
            else:
                # Command is being processed immediately  
                self._add_step("Processing command immediately", {
                    "requires_unlock": False
                })
                
                # Check if screen lock was needed but we're proceeding anyway
                if result.get("requires_unlock"):
                    # This means we detected lock, handled unlock, and are now executing
                    # Don't call the original processor - use our result
                    feedback_message = result.get("message", "I processed your command.")
                    return {
                        "success": True,
                        "response": feedback_message,
                        "result": result,
                        "execution_steps": self.execution_steps
                    }
                
            # Step 4: Execute through original processor ONLY if not queued
            if hasattr(self.command_processor, 'process_command') and result.get("status") != "queued":
                try:
                    # Call the original processor
                    processor_result = await self.command_processor.process_command(command)
                    
                    # Merge results
                    if isinstance(processor_result, dict):
                        result.update(processor_result)
                        
                except Exception as e:
                    logger.error(f"Error in command processor: {e}")
                    # Continue with context intelligence result
                    
            # Step 5: Send final response
            intent = result.get("intent", {})
            success_message = self._build_success_message(command, intent)
            
            await self.feedback_manager.send_contextual_feedback(
                "command_complete",
                success_message
            )
            
            logger.info(f"[CONTEXT INTELLIGENCE] ========= COMPLETED =========")
            
            return {
                "success": True,
                "response": result.get("response", result.get("message", success_message)),
                "result": result,
                "execution_steps": self.execution_steps,
                "message": result.get("message", success_message)
            }
            
        except Exception as e:
            logger.error(f"Error in context intelligence processing: {e}", exc_info=True)
            
            # Send error feedback
            await self.feedback_manager.send_feedback(
                f"I encountered an error processing your command: {str(e)}",
                FeedbackType.ERROR
            )
            
            return {
                "success": False,
                "error": str(e),
                "execution_steps": self.execution_steps
            }
            
        finally:
            # Clear websocket reference
            self._current_websocket = None
            
    def _extract_action(self, command: str) -> str:
        """Extract main action from command"""
        command_lower = command.lower()
        
        if "search for" in command_lower:
            return command_lower.split("search for")[1].strip()
        elif "open" in command_lower:
            parts = command_lower.split("open")
            if len(parts) > 1:
                target = parts[1].strip()
                # Handle "open X and do Y" commands
                if " and " in target:
                    return f"open {target}"
                return f"open {target}"
        elif "go to" in command_lower:
            return f"navigate to {command_lower.split('go to')[1].strip()}"
        else:
            return command
            
    def _get_command_type(self, intent: Dict[str, Any]) -> str:
        """Determine command type from intent"""
        action = intent.get("action", "").lower()
        
        if action in ["open", "launch"]:
            return "open_app"
        elif action in ["search", "find", "google"]:
            return "search"
        elif action in ["navigate", "go"]:
            return "browse"
        else:
            return "system_command"
            
    def _build_success_message(self, command: str, intent: Dict[str, Any]) -> str:
        """Build success message from command and intent"""
        action = intent.get("action", "completed")
        target = intent.get("target", "your request")
        
        if action == "open" and target:
            return f"opened {target}"
        elif action == "search" and target:
            return f"searched for {target}"
        else:
            return "completed your request"
            
    def _requires_screen(self, command: str) -> bool:
        """Check if command requires screen access"""
        command_lower = command.lower()
        
        # Same patterns as original for compatibility
        screen_patterns = [
            "open", "search", "browse", "launch", "start",
            "click", "type", "show", "display"
        ]
        
        return any(pattern in command_lower for pattern in screen_patterns)


def wrap_with_enhanced_context(processor):
    """
    Drop-in replacement for the original wrap_with_enhanced_context
    
    This function maintains the same interface but returns our new
    context intelligence handler
    """
    return EnhancedContextIntelligenceHandler(processor)


# Alias for compatibility
EnhancedSimpleContextHandler = EnhancedContextIntelligenceHandler