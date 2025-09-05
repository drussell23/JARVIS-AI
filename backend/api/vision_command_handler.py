"""
Vision Command Handler for JARVIS
Handles commands related to screen monitoring and vision system
"""

import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class VisionCommandHandler:
    """Handles vision-related commands"""
    
    def __init__(self):
        self.vision_manager = None
        self.monitoring_commands = {
            "start monitoring my screen": self.start_monitoring,
            "start monitoring screen": self.start_monitoring,
            "monitor my screen": self.start_monitoring,
            "start screen monitoring": self.start_monitoring,
            "enable screen monitoring": self.start_monitoring,
            "activate vision": self.start_monitoring,
            "stop monitoring": self.stop_monitoring,
            "stop screen monitoring": self.stop_monitoring,
            "disable monitoring": self.stop_monitoring,
            "stop monitoring my screen": self.stop_monitoring,
        }
    
    async def initialize_vision_manager(self):
        """Initialize vision manager if not already done"""
        if not self.vision_manager:
            try:
                from api.vision_websocket import vision_manager
                self.vision_manager = vision_manager
                logger.info("Vision manager initialized for command handling")
            except Exception as e:
                logger.error(f"Failed to initialize vision manager: {e}")
    
    async def handle_command(self, command: str) -> Dict[str, Any]:
        """Process a vision-related command"""
        command_lower = command.lower().strip()
        
        # Check if this is a monitoring command
        for cmd_pattern, handler in self.monitoring_commands.items():
            if cmd_pattern in command_lower:
                return await handler()
        
        # Check for general vision queries
        if any(word in command_lower for word in ['see', 'screen', 'monitor', 'vision', 'looking']):
            return await self.analyze_screen(command)
        
        return {
            "handled": False,
            "response": None
        }
    
    async def start_monitoring(self) -> Dict[str, Any]:
        """Start screen monitoring"""
        await self.initialize_vision_manager()
        
        if not self.vision_manager:
            return {
                "handled": True,
                "response": "I'm having trouble accessing the vision system. Please make sure the backend is properly configured.",
                "error": True
            }
        
        try:
            # Start monitoring
            await self.vision_manager.start_monitoring()
            
            return {
                "handled": True,
                "response": "Screen monitoring activated. I can now see your screen in real-time. The purple recording indicator should appear in your menu bar.",
                "monitoring_active": True
            }
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return {
                "handled": True,
                "response": f"I encountered an error starting screen monitoring: {str(e)}",
                "error": True
            }
    
    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop screen monitoring"""
        await self.initialize_vision_manager()
        
        if not self.vision_manager:
            return {
                "handled": True,
                "response": "Vision system not available.",
                "error": True
            }
        
        try:
            await self.vision_manager.stop_monitoring()
            
            return {
                "handled": True,
                "response": "Screen monitoring has been stopped. The recording indicator should disappear.",
                "monitoring_active": False
            }
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return {
                "handled": True,
                "response": f"Error stopping monitoring: {str(e)}",
                "error": True
            }
    
    async def analyze_screen(self, query: str) -> Dict[str, Any]:
        """Analyze current screen content"""
        await self.initialize_vision_manager()
        
        if not self.vision_manager or not self.vision_manager.vision_analyzer:
            return {
                "handled": True,
                "response": "I need to start monitoring your screen first. Would you like me to do that?",
                "suggest_monitoring": True
            }
        
        try:
            # Take a screenshot and analyze
            result = await self.vision_manager.vision_analyzer.analyze_screen_with_query(query)
            
            return {
                "handled": True,
                "response": result.get('analysis', 'I captured the screen but couldn\'t analyze it properly.'),
                "screenshot_taken": True
            }
        except Exception as e:
            logger.error(f"Failed to analyze screen: {e}")
            return {
                "handled": True,
                "response": "I had trouble analyzing your screen. Please try again.",
                "error": True
            }

# Global instance
vision_command_handler = VisionCommandHandler()