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
        """Initialize vision manager with error handling"""
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
        logger.info(f"[VISION] analyze_screen called with query: {query}")
        
        await self.initialize_vision_manager()
        
        if not self.vision_manager or not self.vision_manager.vision_analyzer:
            logger.warning("[VISION] No vision manager available")
            return {
                "handled": True,
                "response": "I need to start monitoring your screen first. Would you like me to do that?",
                "suggest_monitoring": True
            }
        
        # Check if we have an API key
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.error("[VISION] No ANTHROPIC_API_KEY found")
            return {
                "handled": True,
                "response": "I can't analyze screens right now. The vision API is not configured properly.",
                "error": True,
                "missing_api_key": True
            }
        
        try:
            # First capture the screen
            logger.info("[VISION] Capturing screen...")
            screenshot = await self.vision_manager.vision_analyzer.capture_screen()
            
            if screenshot is None:
                logger.error("[VISION] Screenshot capture returned None")
                return {
                    "handled": True,
                    "response": "I couldn't capture your screen. Please make sure screen recording permissions are granted.",
                    "error": True
                }
            
            logger.info(f"[VISION] Screenshot captured, type: {type(screenshot)}")
            
            # Convert PIL Image to numpy array if needed
            import numpy as np
            from PIL import Image
            
            if isinstance(screenshot, Image.Image):
                logger.info(f"[VISION] Converting PIL Image to numpy array, size: {screenshot.size}")
                screenshot_array = np.array(screenshot)
            else:
                screenshot_array = screenshot
                logger.info(f"[VISION] Screenshot already numpy array, shape: {screenshot_array.shape if hasattr(screenshot_array, 'shape') else 'unknown'}")
            
            # Analyze the screenshot with timeout
            logger.info("[VISION] Starting screenshot analysis with Claude...")
            try:
                # For conversational queries about screen visibility, provide a natural response
                if any(phrase in query.lower() for phrase in [
                    'can you see', 'do you see', 'what do you see', 
                    'what\'s on my screen', 'what is on my screen',
                    'what am i looking at', 'what\'s happening',
                    'describe my screen', 'tell me what you see'
                ]):
                    # Use a more conversational prompt and force full-screen analysis
                    conversational_query = "You are JARVIS, Tony Stark's AI assistant. Describe what you see on this screen in a natural, conversational way as if you were looking over the user's shoulder. Focus on the main content, applications, or activities visible. Be helpful and observant like JARVIS would be. Don't break it down by regions or technical details - just tell them what's happening on their screen."
                    
                    # Force full-screen analysis (no sliding window) for conversational queries
                    result = await asyncio.wait_for(
                        self.vision_manager.vision_analyzer.analyze_screenshot_async(
                            screenshot_array, 
                            conversational_query,
                            use_sliding_window=False  # Force full screen analysis
                        ),
                        timeout=30.0  # 30 second timeout
                    )
                    
                    # Extract the natural response
                    response_text = result.get('description', result.get('analysis', ''))
                    
                    # If it still contains "Region" formatting, extract just the content
                    if "Region" in response_text:
                        # Try to extract meaningful content from the regions
                        import re
                        region_matches = re.findall(r'Region.*?:\s*(.+?)(?=Region|\Z)', response_text, re.DOTALL)
                        if region_matches:
                            # Combine the meaningful parts
                            combined = ' '.join([match.strip() for match in region_matches if match.strip()])
                            response_text = f"Yes, I can see your screen. {combined}"
                    
                    # Add a natural prefix if not already conversational
                    if not response_text.lower().startswith(('yes', 'i can see', 'i see')):
                        response_text = f"Yes, I can see your screen. {response_text}"
                    
                else:
                    # For other queries, use the original query
                    result = await asyncio.wait_for(
                        self.vision_manager.vision_analyzer.analyze_screenshot_async(screenshot_array, query),
                        timeout=30.0  # 30 second timeout
                    )
                    response_text = result.get('description', result.get('analysis', 'I captured the screen but couldn\'t analyze it properly.'))
                
                logger.info(f"[VISION] Analysis complete, result keys: {list(result.keys()) if result else 'None'}")
            except asyncio.TimeoutError:
                logger.error("[VISION] Screenshot analysis timed out after 30 seconds")
                return {
                    "handled": True,
                    "response": "I'm having trouble analyzing the screen right now. The analysis is taking too long. Please try again.",
                    "error": True,
                    "timeout": True
                }
            
            return {
                "handled": True,
                "response": response_text,
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