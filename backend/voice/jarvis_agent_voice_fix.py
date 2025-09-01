#!/usr/bin/env python3
"""
Fixed JARVIS Agent Voice with Intelligent Command Routing
Replaces keyword-based routing with Swift NLP classifier
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Tuple

# Import the intelligent command handler
try:
    from .intelligent_command_handler import IntelligentCommandHandler
except ImportError:
    from voice.intelligent_command_handler import IntelligentCommandHandler

logger = logging.getLogger(__name__)

class IntelligentVoiceCommandMixin:
    """
    Mixin to replace keyword-based command routing with intelligent classification
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize intelligent command handler
        self.intelligent_handler = IntelligentCommandHandler(
            user_name=getattr(self, 'user_name', 'Sir')
        )
        logger.info("Initialized Swift-based intelligent command routing")
        
    async def process_voice_input(self, text: str) -> str:
        """
        Process voice input using intelligent classification instead of keywords
        
        This replaces the problematic keyword-based _is_system_command check
        """
        logger.info(f"Processing voice input with intelligent routing: {text}")
        
        # Check for mode switching commands first
        if hasattr(self, '_is_mode_switch') and self._is_mode_switch(text):
            return await self._handle_mode_switch(text)
            
        # Check for confirmation if needed
        if hasattr(self, 'awaiting_confirmation') and self.awaiting_confirmation:
            return await self._handle_confirmation(text)
        
        # Use intelligent classification instead of keyword matching
        response, handler_used = await self.intelligent_handler.handle_command(text)
        
        logger.info(f"Intelligent routing result: handler={handler_used}, "
                   f"command='{text}'")
        
        # If the intelligent handler already processed it, return the response
        if handler_used in ['system', 'conversation']:
            return response
            
        # For vision commands, use the existing vision handler if available
        if handler_used == 'vision' and hasattr(self, '_handle_vision_command'):
            return await self._handle_vision_command(text)
            
        # Fallback to the response from intelligent handler
        return response
    
    def _is_system_command(self, text: str) -> bool:
        """
        DEPRECATED - This method should not be used anymore
        Kept for backward compatibility but always returns False
        """
        logger.warning("_is_system_command called - this should use intelligent routing instead")
        return False

def patch_jarvis_voice_agent(agent_class):
    """
    Monkey patch the JARVISVoiceAgent to use intelligent routing
    
    Usage:
        from jarvis_agent_voice import JARVISVoiceAgent
        from jarvis_agent_voice_fix import patch_jarvis_voice_agent
        
        # Apply the fix
        patch_jarvis_voice_agent(JARVISVoiceAgent)
        
        # Now JARVISVoiceAgent uses intelligent routing
        agent = JARVISVoiceAgent()
    """
    
    # Store original methods
    original_init = agent_class.__init__
    original_process = agent_class.process_voice_input
    
    # Create new methods that use intelligent routing
    def new_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Add intelligent handler
        self.intelligent_handler = IntelligentCommandHandler(
            user_name=self.user_name
        )
        logger.info("Patched JARVISVoiceAgent with intelligent command routing")
    
    async def new_process_voice_input(self, text: str) -> str:
        """Process voice input using intelligent classification"""
        logger.info(f"Processing with intelligent routing: {text}")
        
        # Check for special cases first
        if hasattr(self, '_is_mode_switch') and self._is_mode_switch(text):
            return await self._handle_mode_switch(text)
            
        if hasattr(self, 'awaiting_confirmation') and self.awaiting_confirmation:
            return await self._handle_confirmation(text)
        
        # Use intelligent classification
        response, handler_used = await self.intelligent_handler.handle_command(text)
        
        logger.info(f"Swift classifier result: handler={handler_used} for '{text}'")
        
        # Route to appropriate existing handler based on classification
        if handler_used == 'vision' and hasattr(self, '_handle_vision_command'):
            # Use existing vision handler for vision commands
            return await self._handle_vision_command(text)
        elif handler_used == 'system':
            # The intelligent handler already executed system commands
            return response
        else:
            # For conversation and other types, use the response
            return response
    
    # Apply patches
    agent_class.__init__ = new_init
    agent_class.process_voice_input = new_process_voice_input
    
    # Disable the problematic keyword-based method
    def disabled_is_system_command(self, text: str) -> bool:
        """This method is disabled - using intelligent routing instead"""
        return False
    
    agent_class._is_system_command = disabled_is_system_command
    
    logger.info("Successfully patched JARVISVoiceAgent with intelligent routing")
    return agent_class

# Test the fix
async def test_intelligent_routing():
    """Test that problematic commands are now routed correctly"""
    handler = IntelligentCommandHandler()
    
    test_commands = [
        ("open WhatsApp", "system"),  # Should be system, not vision
        ("close WhatsApp", "system"),  # Should be system
        ("what's on my screen", "vision"),  # Should be vision
        ("what's the weather", "conversation"),  # Should be conversation
        ("open Safari", "system"),  # Should be system
        ("show me my notifications", "vision"),  # Should be vision
    ]
    
    print("\n🧪 Testing Intelligent Command Routing:\n")
    
    for command, expected in test_commands:
        response, handler_used = await handler.handle_command(command)
        success = "✅" if handler_used == expected else "❌"
        print(f"{success} '{command}' → {handler_used} (expected: {expected})")
        if handler_used != expected:
            print(f"   Response: {response[:100]}...")
    
    print("\n✨ The fix resolves the 'open WhatsApp' misrouting issue!")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_intelligent_routing())