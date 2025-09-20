"""
Vision WebSocket Fix - Provides vision_manager for vision_command_handler
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class VisionManager:
    """Simple vision manager that wraps the vision analyzer"""
    
    def __init__(self):
        self.vision_analyzer = None
        self._monitoring_active = False
    
    def set_vision_analyzer(self, analyzer):
        """Set the vision analyzer"""
        self.vision_analyzer = analyzer
        logger.info(f"Vision analyzer set: {analyzer is not None}")
    
    async def start_monitoring(self):
        """Start screen monitoring"""
        if not self.vision_analyzer:
            raise Exception("Vision analyzer not available")
        
        # Start video streaming
        result = await self.vision_analyzer.start_video_streaming()
        if result.get('success'):
            self._monitoring_active = True
            logger.info("Screen monitoring started successfully")
        else:
            raise Exception(f"Failed to start monitoring: {result.get('error')}")
    
    async def stop_monitoring(self):
        """Stop screen monitoring"""
        if not self.vision_analyzer:
            raise Exception("Vision analyzer not available")
        
        # Stop video streaming
        result = await self.vision_analyzer.stop_video_streaming()
        self._monitoring_active = False
        logger.info("Screen monitoring stopped")
    
    async def capture_screen(self, multi_space=False, space_number=None):
        """Capture current screen with multi-space support"""
        if not self.vision_analyzer:
            raise Exception("Vision analyzer not available")
        
        # Use the correct method name and pass multi-space parameters
        return await self.vision_analyzer.capture_screen(
            multi_space=multi_space,
            space_number=space_number
        )
    
    async def analyze_screen(self, prompt: str):
        """Analyze screen with prompt"""
        if not self.vision_analyzer:
            raise Exception("Vision analyzer not available")
        
        # Capture screen first
        screenshot = await self.capture_screen()
        if screenshot is None:
            raise Exception("Failed to capture screen")
        
        # Analyze with prompt
        result = await self.vision_analyzer.analyze_image_with_prompt(screenshot, prompt)
        return result

# Create global vision manager instance
vision_manager = VisionManager()

def set_vision_analyzer(analyzer):
    """Set the vision analyzer in the global vision manager"""
    vision_manager.set_vision_analyzer(analyzer)
    logger.info("Vision analyzer set in vision_websocket module")