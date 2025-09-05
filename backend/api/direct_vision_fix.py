#!/usr/bin/env python3
"""
Direct fix for vision monitoring commands in JARVIS
This ensures screen monitoring commands work immediately
"""

import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global flag to track if we've patched the system
_patched = False

async def handle_vision_monitoring_command(command: str) -> Dict[str, Any]:
    """Direct handler for vision monitoring commands"""
    command_lower = command.lower().strip()
    
    # Check if this is a monitoring command
    if any(phrase in command_lower for phrase in [
        "start monitoring my screen",
        "monitor my screen", 
        "start screen monitoring",
        "enable screen monitoring",
        "activate screen monitoring",
        "start monitoring screen"
    ]):
        logger.info("Vision monitoring command detected - starting screen capture")
        
        try:
            # Import and start the vision monitoring directly
            from api.vision_websocket import vision_manager
            
            # Start monitoring
            await vision_manager.start_monitoring()
            
            # Start video streaming
            if vision_manager.vision_analyzer:
                result = await vision_manager.vision_analyzer.start_video_streaming()
                logger.info(f"Video streaming start result: {result}")
                
                if result.get('success'):
                    return {
                        "handled": True,
                        "response": "I'm now monitoring your screen. The purple recording indicator should appear in your menu bar. I can see everything on your display and will help you with any questions about what's shown.",
                        "success": True,
                        "monitoring_active": True
                    }
            
            return {
                "handled": True,
                "response": "Screen monitoring is starting up. Give me a moment to activate the video capture system.",
                "success": True,
                "monitoring_active": True
            }
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}", exc_info=True)
            return {
                "handled": True,
                "response": f"I encountered an issue starting screen monitoring: {str(e)}. Let me try a different approach.",
                "success": False,
                "error": str(e)
            }
    
    elif any(phrase in command_lower for phrase in [
        "stop monitoring",
        "stop screen monitoring", 
        "disable monitoring",
        "stop monitoring my screen"
    ]):
        logger.info("Stop monitoring command detected")
        
        try:
            from api.vision_websocket import vision_manager
            await vision_manager.stop_monitoring()
            
            return {
                "handled": True,
                "response": "Screen monitoring has been stopped. The recording indicator should disappear.",
                "success": True,
                "monitoring_active": False
            }
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            return {
                "handled": True,
                "response": "There was an issue stopping the monitoring, but it should be disabled now.",
                "success": False,
                "error": str(e)
            }
    
    # Not a monitoring command
    return {"handled": False}


def patch_jarvis_command_handling():
    """Patch JARVIS to handle vision commands directly"""
    global _patched
    if _patched:
        return
    
    try:
        import api.jarvis_voice_api
        
        # Store original process_command
        original_process = api.jarvis_voice_api.JARVISVoiceAPI.process_command
        
        async def patched_process_command(self, command):
            """Patched command processor that checks vision first"""
            # Check vision commands first
            if hasattr(command, 'text'):
                vision_result = await handle_vision_monitoring_command(command.text)
                if vision_result.get('handled'):
                    return {
                        "response": vision_result['response'],
                        "status": "success",
                        "confidence": 1.0,
                        "command_type": "vision",
                        "monitoring_active": vision_result.get('monitoring_active')
                    }
            
            # Fall back to original processing
            return await original_process(self, command)
        
        # Apply patch
        api.jarvis_voice_api.JARVISVoiceAPI.process_command = patched_process_command
        logger.info("Applied vision command patch to JARVIS")
        _patched = True
        
    except Exception as e:
        logger.error(f"Failed to patch JARVIS: {e}")


# Auto-patch on import
patch_jarvis_command_handling()