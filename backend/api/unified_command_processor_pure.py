"""
Unified Command Processor - Pure Intelligence Version
Simplified to use Claude's natural understanding instead of pattern matching

The old way: Complex routing logic, pattern matching, multiple handlers
The new way: Claude understands everything naturally
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PureUnifiedContext:
    """Simplified unified context - Claude maintains the real context"""
    last_command: Optional[str] = None
    last_response: Optional[str] = None
    last_command_time: Optional[datetime] = None
    session_start: datetime = None
    
    def __post_init__(self):
        if not self.session_start:
            self.session_start = datetime.now()


class PureUnifiedCommandProcessor:
    """
    Unified processor using pure Claude intelligence.
    No routing tables, no pattern matching - Claude understands intent naturally.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.context = PureUnifiedContext()
        self.vision_handler = None
        self._initialized = False
        
    async def _ensure_initialized(self):
        """Lazy initialization of handlers"""
        if not self._initialized:
            try:
                # Initialize pure vision handler
                from .vision_command_handler import vision_command_handler
                self.vision_handler = vision_command_handler
                
                # Initialize with API key if available
                if self.api_key and hasattr(self.vision_handler, 'initialize_intelligence'):
                    await self.vision_handler.initialize_intelligence(self.api_key)
                    
                self._initialized = True
                logger.info("[PURE PROCESSOR] Initialized with pure intelligence")
                
            except Exception as e:
                logger.error(f"Failed to initialize pure processor: {e}")
                
    async def process_command(self, command_text: str, websocket=None) -> Dict[str, Any]:
        """
        Process any command using pure Claude intelligence.
        Claude figures out what the user wants - no pattern matching needed.
        """
        logger.info(f"[PURE] Processing: '{command_text}'")
        
        # Ensure initialized
        await self._ensure_initialized()
        
        # Update context
        self.context.last_command = command_text
        self.context.last_command_time = datetime.now()
        
        try:
            # For vision-related queries, use vision intelligence
            # But we don't need to detect this - Claude will understand
            if self._might_be_vision_related(command_text) and self.vision_handler:
                result = await self.vision_handler.handle_command(command_text)
                
                # Update context
                self.context.last_response = result.get('response', '')
                
                return {
                    'success': True,
                    'response': result.get('response', ''),
                    'command_type': 'vision_intelligence',
                    'pure_intelligence': True,
                    **result
                }
            else:
                # For non-vision commands, we would use other pure intelligence handlers
                # For now, indicate it needs implementation
                return {
                    'success': False,
                    'response': await self._get_natural_fallback_response(command_text),
                    'command_type': 'not_implemented',
                    'pure_intelligence': True
                }
                
        except Exception as e:
            logger.error(f"Pure processor error: {e}", exc_info=True)
            return {
                'success': False,
                'response': await self._get_natural_error_response(command_text, str(e)),
                'command_type': 'error',
                'error': str(e),
                'pure_intelligence': True
            }
            
    def _might_be_vision_related(self, command: str) -> bool:
        """
        Simple heuristic to decide if we should try vision handler.
        In a fully pure system, we wouldn't even need this - Claude would route.
        """
        # Very broad check - let Claude figure out the specifics
        vision_indicators = [
            'see', 'look', 'screen', 'monitor', 'show', 'what', 
            'window', 'open', 'error', 'terminal', 'battery',
            'can you', 'do you', 'analyze', 'tell me'
        ]
        
        command_lower = command.lower()
        return any(indicator in command_lower for indicator in vision_indicators)
        
    async def _get_natural_fallback_response(self, command: str) -> str:
        """
        Even fallback responses are natural and varied.
        In production, this would use Claude to generate a natural response.
        """
        # This would be replaced with actual Claude call
        responses = [
            f"I understand you're asking about '{command}', but I don't have that capability enabled yet.",
            f"That's an interesting request about '{command}'. This feature is still being developed.",
            f"I heard '{command}', but I'm not equipped to handle that type of request yet.",
        ]
        
        # In production: return await claude.generate_natural_response(command, context="capability_not_available")
        import random
        return random.choice(responses)
        
    async def _get_natural_error_response(self, command: str, error: str) -> str:
        """
        Natural error responses - no templates.
        In production, Claude would generate these based on the error context.
        """
        # This would be replaced with actual Claude call
        return f"I encountered an issue while processing your request about '{command}'. Let me help you troubleshoot this."
        
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        session_duration = (datetime.now() - self.context.session_start).total_seconds()
        
        stats = {
            'session_duration_seconds': session_duration,
            'last_command': self.context.last_command,
            'last_command_age': (
                (datetime.now() - self.context.last_command_time).total_seconds()
                if self.context.last_command_time else None
            ),
            'pure_intelligence': True
        }
        
        # Add vision intelligence stats if available
        if self.vision_handler and hasattr(self.vision_handler, 'get_intelligence_stats'):
            stats['vision_intelligence'] = self.vision_handler.get_intelligence_stats()
            
        return stats


# Singleton instance
_pure_unified_processor = None

def get_pure_unified_processor(api_key: Optional[str] = None) -> PureUnifiedCommandProcessor:
    """Get or create the pure unified command processor"""
    global _pure_unified_processor
    if _pure_unified_processor is None:
        _pure_unified_processor = PureUnifiedCommandProcessor(api_key)
    return _pure_unified_processor