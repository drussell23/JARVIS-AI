"""
Vision Command Handler for JARVIS
Handles commands related to screen monitoring and vision system
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketLogger:
    """Logger that sends logs to WebSocket for browser console"""
    def __init__(self):
        self.websocket_callback: Optional[Callable] = None
        
    def set_websocket_callback(self, callback: Callable):
        """Set callback to send logs through WebSocket"""
        self.websocket_callback = callback
        
    async def log(self, message: str, level: str = "info"):
        """Send log message through WebSocket"""
        if self.websocket_callback:
            try:
                await self.websocket_callback({
                    "type": "debug_log",
                    "message": f"[VISION] {message}",
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to send WebSocket log: {e}")
        
        # Also log to server console
        if level == "error":
            logger.error(message)
        else:
            logger.info(message)

# Global WebSocket logger instance
ws_logger = WebSocketLogger()

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
                logger.info("[VISION INIT] Attempting to import vision_manager from vision_websocket...")
                try:
                    from api.vision_websocket import vision_manager
                except ImportError:
                    # Try relative import if absolute fails
                    from .vision_websocket import vision_manager
                self.vision_manager = vision_manager
                logger.info(f"[VISION INIT] Vision manager imported: {vision_manager}")
                logger.info(f"[VISION INIT] Vision manager type: {type(vision_manager)}")
                logger.info(f"[VISION INIT] Has vision_analyzer attr: {hasattr(vision_manager, 'vision_analyzer')}")
                logger.info(f"[VISION INIT] Has _vision_analyzer attr: {hasattr(vision_manager, '_vision_analyzer')}")
                if hasattr(vision_manager, '_vision_analyzer'):
                    logger.info(f"[VISION INIT] _vision_analyzer value: {vision_manager._vision_analyzer}")
                logger.info(f"[VISION INIT] Vision analyzer present: {hasattr(vision_manager, 'vision_analyzer') and vision_manager.vision_analyzer is not None}")
                
                # Check if vision_analyzer needs initialization
                if hasattr(vision_manager, 'vision_analyzer') and vision_manager.vision_analyzer is None:
                    logger.info("[VISION INIT] Vision analyzer is None, checking app state...")
                    # Try to get from app state
                    try:
                        from fastapi import FastAPI
                        import sys
                        import os
                        # Add parent directory to path to import main
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from main import app
                        if hasattr(app.state, 'vision_analyzer'):
                            vision_manager.vision_analyzer = app.state.vision_analyzer
                            logger.info("[VISION INIT] Set vision analyzer from app state")
                    except Exception as e:
                        logger.error(f"[VISION INIT] Failed to get vision analyzer from app state: {e}")
                        
                logger.info("[VISION INIT] Vision manager initialization complete")
                
                # Force check of vision_analyzer property to trigger getter
                if self.vision_manager:
                    analyzer = self.vision_manager.vision_analyzer
                    logger.info(f"[VISION INIT] Vision analyzer after property access: {analyzer}")
                
            except Exception as e:
                logger.error(f"[VISION INIT] Failed to initialize vision manager: {e}", exc_info=True)
    
    async def handle_command(self, command: str) -> Dict[str, Any]:
        """Process a vision-related command"""
        logger.info(f"[VISION HANDLER] handle_command called with: {command}")
        try:
            command_lower = command.lower().strip()
            
            # Check if this is a monitoring command
            for cmd_pattern, handler in self.monitoring_commands.items():
                if cmd_pattern in command_lower:
                    return await handler()
            
            # Check for general vision queries
            if any(word in command_lower for word in ['see', 'screen', 'monitor', 'vision', 'looking']):
                logger.info(f"[VISION HANDLER] Detected vision query, calling analyze_screen")
                return await self.analyze_screen(command)
            
            return {
                "handled": False,
                "response": None
            }
        except Exception as e:
            logger.error(f"Error in handle_command: {e}", exc_info=True)
            return {
                "handled": True,
                "response": "I'm having trouble processing your vision request. Please make sure the vision system is properly configured.",
                "error": True
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
        logger.info(f"[ANALYZE SCREEN] Starting with query: {query}")
        await ws_logger.log(f"analyze_screen called with query: {query}")
        
        # Try to get vision analyzer directly from app state first
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from main import app
            if hasattr(app.state, 'vision_analyzer') and app.state.vision_analyzer:
                logger.info(f"[ANALYZE SCREEN] Found vision_analyzer in app.state: {app.state.vision_analyzer}")
                await ws_logger.log(f"Vision analyzer found in app state: {type(app.state.vision_analyzer).__name__}")
            else:
                logger.warning("[ANALYZE SCREEN] No vision_analyzer in app.state")
                await ws_logger.log("No vision analyzer in app state", "warning")
        except Exception as e:
            logger.error(f"[ANALYZE SCREEN] Failed to check app.state: {e}")
            await ws_logger.log(f"Failed to check app state: {e}", "error")
        
        await ws_logger.log("Step 1: Initializing vision manager...")
        await self.initialize_vision_manager()
        logger.info(f"[ANALYZE SCREEN] Vision manager: {self.vision_manager}")
        
        if not self.vision_manager or not self.vision_manager.vision_analyzer:
            await ws_logger.log("No vision manager available", "warning")
            await ws_logger.log(f"Vision manager exists: {self.vision_manager is not None}")
            if self.vision_manager:
                await ws_logger.log(f"Vision analyzer exists: {hasattr(self.vision_manager, 'vision_analyzer') and self.vision_manager.vision_analyzer is not None}")
                if hasattr(self.vision_manager, 'vision_analyzer'):
                    await ws_logger.log(f"Vision analyzer value: {self.vision_manager.vision_analyzer}")
            
            logger.error(f"[ANALYZE SCREEN] Vision not available - manager: {self.vision_manager}, analyzer: {getattr(self.vision_manager, 'vision_analyzer', None) if self.vision_manager else None}")
            return {
                "handled": True,
                "response": "I need to start monitoring your screen first. Would you like me to do that?",
                "suggest_monitoring": True
            }
        
        # Check if we have an API key
        await ws_logger.log("Step 2: Checking API key...")
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            await ws_logger.log("No ANTHROPIC_API_KEY found", "error")
            return {
                "handled": True,
                "response": "I can't analyze screens right now. The vision API is not configured properly.",
                "error": True,
                "missing_api_key": True
            }
        await ws_logger.log("API key found")
        
        try:
            # First capture the screen
            await ws_logger.log("Step 3: Capturing screen...")
            
            # Check if vision analyzer exists
            if not hasattr(self.vision_manager, 'vision_analyzer') or not self.vision_manager.vision_analyzer:
                await ws_logger.log("Vision analyzer not available", "error")
                return {
                    "handled": True,
                    "response": "I'm having trouble with my vision system. Please ensure the backend is properly configured.",
                    "error": True,
                    "error_type": "vision_unavailable"
                }
            
            try:
                screenshot = await self.vision_manager.vision_analyzer.capture_screen()
            except Exception as capture_error:
                await ws_logger.log(f"Screenshot capture exception: {capture_error}", "error")
                return {
                    "handled": True,
                    "response": "I encountered an error capturing your screen. Please check screen recording permissions in System Preferences.",
                    "error": True,
                    "error_type": "capture_failed",
                    "error_details": str(capture_error)
                }
            
            await ws_logger.log(f"Step 3 complete: Screenshot captured = {screenshot is not None}")
            
            if screenshot is None:
                await ws_logger.log("Screenshot capture returned None", "error")
                return {
                    "handled": True,
                    "response": "I couldn't capture your screen. Please make sure screen recording permissions are granted in System Preferences > Security & Privacy > Privacy > Screen Recording.",
                    "error": True,
                    "error_type": "capture_none"
                }
            
            await ws_logger.log(f"Screenshot captured, type: {type(screenshot)}")
            
            # Convert PIL Image to numpy array if needed
            import numpy as np
            from PIL import Image
            
            if isinstance(screenshot, Image.Image):
                await ws_logger.log(f"Step 4: Converting PIL Image to numpy array, size: {screenshot.size}")
                screenshot_array = np.array(screenshot)
            else:
                screenshot_array = screenshot
                await ws_logger.log(f"Screenshot already numpy array, shape: {screenshot_array.shape if hasattr(screenshot_array, 'shape') else 'unknown'}")
            
            # Analyze the screenshot with timeout
            await ws_logger.log("Step 5: Starting screenshot analysis with Claude...")
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
                    await ws_logger.log("Step 6: Starting conversational analysis with 30s timeout...")
                    await ws_logger.log(f"Vision analyzer type: {type(self.vision_manager.vision_analyzer)}")
                    await ws_logger.log(f"Has client: {hasattr(self.vision_manager.vision_analyzer, 'client')}")
                    if hasattr(self.vision_manager.vision_analyzer, 'client'):
                        await ws_logger.log(f"Client type: {type(self.vision_manager.vision_analyzer.client) if self.vision_manager.vision_analyzer.client else 'None'}")
                    await ws_logger.log("Calling analyze_screenshot_async...")
                    
                    try:
                        # Check if the analyze method exists
                        if not hasattr(self.vision_manager.vision_analyzer, 'analyze_screenshot_async'):
                            await ws_logger.log("analyze_screenshot_async method not found", "error")
                            # Try alternative method
                            if hasattr(self.vision_manager.vision_analyzer, 'analyze_image_with_prompt'):
                                result = await asyncio.wait_for(
                                    self.vision_manager.vision_analyzer.analyze_image_with_prompt(
                                        screenshot_array,
                                        conversational_query
                                    ),
                                    timeout=30.0
                                )
                            else:
                                raise AttributeError("No suitable analysis method found")
                        else:
                            result = await asyncio.wait_for(
                                self.vision_manager.vision_analyzer.analyze_screenshot_async(
                                    screenshot_array, 
                                    conversational_query,
                                    use_sliding_window=False  # Force full screen analysis
                                ),
                                timeout=30.0  # Increased to 30 second timeout to match API timeout
                            )
                        await ws_logger.log("Step 6 complete: Analysis returned")
                    except asyncio.TimeoutError:
                        await ws_logger.log("Analysis timed out after 30 seconds", "error")
                        raise
                    except AttributeError as e:
                        await ws_logger.log(f"Method not found: {e}", "error")
                        raise
                    except Exception as e:
                        await ws_logger.log(f"Analysis error: {str(e)}", "error")
                        raise
                    
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
                
                await ws_logger.log(f"Analysis complete, result keys: {list(result.keys()) if result else 'None'}")
            except asyncio.TimeoutError:
                await ws_logger.log("Screenshot analysis timed out after 15 seconds", "error")
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
            await ws_logger.log(f"Failed to analyze screen: {e}", "error")
            logger.error(f"Failed to analyze screen: {e}", exc_info=True)
            return {
                "handled": True,
                "response": "I had trouble analyzing your screen. Please try again.",
                "error": True
            }

# Global instance
vision_command_handler = VisionCommandHandler()