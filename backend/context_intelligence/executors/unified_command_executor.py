"""
Unified Command Executor
=======================

Integrates the Context Intelligence System with the existing
JARVIS unified command processor for actual command execution.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class UnifiedCommandExecutor:
    """
    Executes commands using the existing JARVIS unified command processor
    while providing progress updates to the context intelligence system.
    """
    
    def __init__(self):
        """Initialize the command executor"""
        self._processor = None
        self._initialized = False
        
    async def _ensure_processor(self):
        """Ensure we have access to the unified command processor"""
        if not self._initialized:
            try:
                from api.unified_command_processor import get_unified_processor
                self._processor = get_unified_processor()
                self._initialized = True
                logger.info("Unified command processor initialized")
            except ImportError as e:
                logger.error(f"Failed to import unified processor: {e}")
                # Try alternative import
                try:
                    import sys
                    import os
                    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    if backend_dir not in sys.path:
                        sys.path.insert(0, backend_dir)
                    
                    from api.unified_command_processor import get_unified_processor
                    self._processor = get_unified_processor()
                    self._initialized = True
                    logger.info("Unified command processor initialized (alternative import)")
                except Exception as e2:
                    logger.error(f"Failed alternative import: {e2}")
                    raise
                    
    async def execute_command(self, command_text: str, 
                            intent: Dict[str, Any],
                            context: Dict[str, Any],
                            progress_callback: Optional[callable] = None,
                            websocket = None) -> Dict[str, Any]:
        """
        Execute a command using the unified processor
        
        Args:
            command_text: The command to execute
            intent: Parsed intent information
            context: Current system context
            progress_callback: Optional callback for progress updates
            websocket: WebSocket connection for real-time updates
            
        Returns:
            Execution result dictionary
        """
        # Check if this is a lock screen command we should handle directly
        if self._is_lock_screen_command(command_text, intent):
            return await self._handle_lock_screen(command_text, progress_callback)
        
        await self._ensure_processor()
        
        if not self._processor:
            return {
                "success": False,
                "error": "Command processor not available"
            }
            
        try:
            # Notify start of execution
            if progress_callback:
                await progress_callback("Starting command execution...")
                
            # Build enhanced command if we have compound actions
            enhanced_command = self._build_enhanced_command(command_text, intent)
            
            # Execute through unified processor
            logger.info(f"Executing command: {enhanced_command}")
            
            result = await self._processor.process_command(
                enhanced_command,
                websocket=websocket
            )
            
            # Process result
            if result.get('success', False):
                if progress_callback:
                    await progress_callback("Command executed successfully")
                    
                return {
                    "success": True,
                    "result": result,
                    "executed_command": enhanced_command
                }
            else:
                error = result.get('error', 'Unknown error')
                if progress_callback:
                    await progress_callback(f"Execution failed: {error}")
                    
                return {
                    "success": False,
                    "error": error,
                    "result": result
                }
                
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            
            if progress_callback:
                await progress_callback(f"Error: {str(e)}")
                
            return {
                "success": False,
                "error": str(e)
            }
            
    def _build_enhanced_command(self, command_text: str, 
                               intent: Dict[str, Any]) -> str:
        """
        Build an enhanced command from intent information
        
        This helps ensure the command is properly formatted for
        the unified processor.
        """
        # If we have clear intent information, we can enhance the command
        action = intent.get("action", "").lower()
        target = intent.get("target", "")
        
        # For compound commands like "open X and search for Y"
        if " and " in command_text.lower():
            # The command is already compound, use as-is
            return command_text
            
        # For simple commands, ensure they're clear
        if action and target:
            if action == "open":
                return f"open {target}"
            elif action == "search":
                # Ensure search commands include browser context
                if "safari" not in command_text.lower() and "chrome" not in command_text.lower():
                    return f"search for {target} in Safari"
                    
        return command_text
    
    def _is_lock_screen_command(self, command_text: str, intent: Dict[str, Any]) -> bool:
        """Check if this is a lock screen command"""
        command_lower = command_text.lower()
        
        # Check for explicit lock phrases
        lock_phrases = [
            "lock my screen", "lock screen", "lock the screen",
            "lock my mac", "lock mac", "lock the mac",
            "lock computer", "lock the computer", "lock my computer"
        ]
        
        return any(phrase in command_lower for phrase in lock_phrases)
    
    async def _handle_lock_screen(self, command_text: str, 
                                 progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Handle screen lock command directly"""
        try:
            # Get unlock manager (which also handles locking)
            from ..core.unlock_manager import get_unlock_manager
            unlock_manager = get_unlock_manager()
            
            if progress_callback:
                await progress_callback("Locking your screen...")
            
            success, message = await unlock_manager.lock_screen("User command")
            
            if success:
                return {
                    "success": True,
                    "result": {
                        "response": "Screen locked successfully",
                        "message": message,
                        "action": "lock_screen"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": message,
                    "result": {
                        "response": f"Failed to lock screen: {message}",
                        "action": "lock_screen"
                    }
                }
        except Exception as e:
            logger.error(f"Error handling lock screen command: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": {
                    "response": f"Error locking screen: {str(e)}",
                    "action": "lock_screen"
                }
            }
        
    async def test_execution(self) -> bool:
        """Test if the executor can successfully connect to the processor"""
        try:
            await self._ensure_processor()
            return self._processor is not None
        except Exception as e:
            logger.error(f"Executor test failed: {e}")
            return False


# Global executor instance
_executor: Optional[UnifiedCommandExecutor] = None


def get_command_executor() -> UnifiedCommandExecutor:
    """Get or create global command executor"""
    global _executor
    if _executor is None:
        _executor = UnifiedCommandExecutor()
    return _executor