#!/usr/bin/env python3
"""
Vision System v2.0 Integration for JARVIS Voice
Connects the ML-powered vision system with voice commands
"""

import logging
from typing import Optional, Dict, Any
from vision.vision_system_v2 import get_vision_system_v2

logger = logging.getLogger(__name__)

class VisionV2Integration:
    """Integration layer for Vision System v2.0 with JARVIS Voice"""
    
    def __init__(self):
        """Initialize Vision System v2.0 integration"""
        try:
            self.vision_system = get_vision_system_v2()
            self.enabled = True
            logger.info("Vision System v2.0 integration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Vision System v2.0: {e}")
            self.vision_system = None
            self.enabled = False
    
    async def handle_vision_command(self, command: str, context: Optional[Dict] = None) -> str:
        """
        Handle vision commands using Vision System v2.0
        
        Args:
            command: Natural language command
            context: Optional context dictionary
            
        Returns:
            Response string for voice output
        """
        if not self.enabled or not self.vision_system:
            return "Vision System v2.0 is not available. Please check the configuration."
        
        try:
            # Add voice context
            if context is None:
                context = {}
            context['source'] = 'voice'
            context['voice_mode'] = True
            
            # Process command through Vision System v2.0
            response = await self.vision_system.process_command(command, context)
            
            # Log statistics for debugging
            if response.confidence < 0.5:
                logger.warning(f"Low confidence response: {response.confidence}")
            
            # Format response for voice output
            if response.success:
                # Add voice-friendly formatting
                voice_response = response.message
                
                # Add confidence indicator for low confidence
                if response.confidence < 0.7:
                    voice_response += " I'm not entirely certain about this analysis."
                
                # Add suggestions if available
                if response.suggestions:
                    voice_response += f" You might also want to ask: {response.suggestions[0]}"
                
                return voice_response
            else:
                return f"I encountered an issue with the vision analysis: {response.message}"
                
        except Exception as e:
            logger.error(f"Error processing vision command: {e}", exc_info=True)
            return "I'm having trouble analyzing your screen right now. Please try again."
    
    async def get_vision_status(self) -> Dict[str, Any]:
        """Get Vision System v2.0 status"""
        if not self.enabled or not self.vision_system:
            return {
                "available": False,
                "error": "Vision System v2.0 not initialized"
            }
        
        try:
            stats = await self.vision_system.get_system_stats()
            return {
                "available": True,
                "version": stats.get("version", "unknown"),
                "phase": stats.get("phase", "unknown"),
                "success_rate": stats.get("success_rate", 0),
                "learned_patterns": stats.get("learned_patterns", 0),
                "transformer_routing": stats.get("transformer_routing", {}).get("enabled", False),
                "continuous_learning": stats.get("continuous_learning", {}).get("pipeline_version", "unknown")
            }
        except Exception as e:
            logger.error(f"Error getting vision status: {e}")
            return {
                "available": True,
                "error": str(e)
            }