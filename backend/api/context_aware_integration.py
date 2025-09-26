"""
Context-Aware Command Processing Integration
===========================================

Integrates context awareness into JARVIS command processing
"""

import logging
from typing import Dict, Any, Optional, Callable

from context_intelligence.handlers.context_aware_handler import get_context_aware_handler
from context_intelligence.core.system_state_monitor import get_system_monitor

logger = logging.getLogger(__name__)


class ContextAwareCommandProcessor:
    """
    Wrapper for command processing with context awareness
    """
    
    def __init__(self, original_processor):
        """
        Initialize with the original command processor
        
        Args:
            original_processor: The existing command processor instance
        """
        self.original_processor = original_processor
        self.context_handler = get_context_aware_handler()
        self.system_monitor = get_system_monitor()
        self.context_enabled = True
        
        # Start system monitoring
        self._start_monitoring()
        
    def _start_monitoring(self):
        """Start background system monitoring"""
        import asyncio
        try:
            # Start monitoring in background
            loop = asyncio.get_event_loop()
            loop.create_task(self.system_monitor.start_monitoring())
            logger.info("Started system state monitoring for context awareness")
        except Exception as e:
            logger.error(f"Failed to start system monitoring: {e}")
            
    async def process_command_with_context(self, command_text: str, websocket=None) -> Dict[str, Any]:
        """
        Process command with context awareness
        
        Args:
            command_text: The command to process
            websocket: Optional websocket connection
            
        Returns:
            Command result with context handling
        """
        if not self.context_enabled:
            # Fall back to original processing
            return await self.original_processor.process_command(command_text, websocket)
            
        try:
            # Define execution callback that uses the original processor
            async def execute_original_command(cmd: str, context: Dict[str, Any] = None):
                # Call the original processor
                result = await self.original_processor.process_command(cmd, websocket)
                return result
                
            # Process with context awareness
            context_result = await self.context_handler.handle_command_with_context(
                command_text,
                execute_callback=execute_original_command
            )
            
            # Format the response
            if context_result["success"]:
                # Combine messages for response
                response_text = context_result.get("summary", "Command completed")
                
                # If we have the original command result, merge it
                if "result" in context_result:
                    original_result = context_result["result"]
                    # Preserve the original structure but enhance the response
                    if isinstance(original_result, dict):
                        original_result["response"] = response_text
                        original_result["context_aware"] = True
                        original_result["steps_taken"] = context_result.get("steps_taken", [])
                        return original_result
                        
                # Otherwise create a new response
                return {
                    "success": True,
                    "response": response_text,
                    "context_aware": True,
                    "steps_taken": context_result.get("steps_taken", []),
                    "context": context_result.get("context", {})
                }
            else:
                # Handle failure
                return {
                    "success": False,
                    "response": context_result.get("summary", "Command failed"),
                    "context_aware": True,
                    "steps_taken": context_result.get("steps_taken", []),
                    "error": context_result.get("messages", ["Unknown error"])
                }
                
        except Exception as e:
            logger.error(f"Error in context-aware processing: {e}")
            # Fall back to original processing
            return await self.original_processor.process_command(command_text, websocket)
            
    def toggle_context_awareness(self, enabled: bool):
        """Enable or disable context awareness"""
        self.context_enabled = enabled
        logger.info(f"Context awareness {'enabled' if enabled else 'disabled'}")


def wrap_command_processor_with_context(original_processor) -> ContextAwareCommandProcessor:
    """
    Wrap an existing command processor with context awareness
    
    Args:
        original_processor: The original command processor
        
    Returns:
        Context-aware wrapper
    """
    return ContextAwareCommandProcessor(original_processor)


# For backwards compatibility - can be imported and used directly
async def process_command_with_context(processor, command_text: str, websocket=None) -> Dict[str, Any]:
    """
    Process a command with context awareness
    
    Args:
        processor: The command processor instance
        command_text: The command to process
        websocket: Optional websocket
        
    Returns:
        Command result
    """
    # Check if processor is already context-aware
    if hasattr(processor, 'process_command_with_context'):
        return await processor.process_command_with_context(command_text, websocket)
    else:
        # Wrap it temporarily
        context_processor = wrap_command_processor_with_context(processor)
        return await context_processor.process_command_with_context(command_text, websocket)