"""
Vision Command Handler for JARVIS - Pure Intelligence Version
NO TEMPLATES. NO HARDCODING. PURE CLAUDE VISION INTELLIGENCE.

Every response is generated fresh by Claude based on what it sees.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .pure_vision_intelligence import (
    PureVisionIntelligence, 
    ProactiveIntelligence,
    WorkflowIntelligence,
    ConversationContext
)

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
    """
    Handles vision commands using pure Claude intelligence.
    Zero templates, zero hardcoded responses.
    """
    
    def __init__(self):
        self.vision_manager = None
        self.intelligence = None
        self.proactive = None
        self.workflow = None
        self.monitoring_active = False
        self.jarvis_api = None  # For voice integration
        
    async def initialize_intelligence(self, api_key: str = None):
        """Initialize pure vision intelligence system"""
        if not self.intelligence:
            # Try to get existing vision analyzer from app state
            vision_analyzer = None
            try:
                from api.jarvis_factory import get_app_state
                app_state = get_app_state()
                if app_state and hasattr(app_state, 'vision_analyzer'):
                    vision_analyzer = app_state.vision_analyzer
                    logger.info("[PURE VISION] Using existing vision analyzer from app state")
            except Exception as e:
                logger.debug(f"Could not get vision analyzer from app state: {e}")
                
            # If no app state analyzer and we have an API key, create one
            if not vision_analyzer and api_key:
                try:
                    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
                    vision_analyzer = ClaudeVisionAnalyzer(api_key)
                    logger.info("[PURE VISION] Created new vision analyzer with API key")
                except Exception as e:
                    logger.error(f"Failed to create vision analyzer: {e}")
                
            # If no existing analyzer, create a wrapper for the API
            if vision_analyzer:
                # Create a Claude client wrapper that uses the existing vision analyzer
                class ClaudeVisionWrapper:
                    def __init__(self, analyzer):
                        self.analyzer = analyzer
                        
                    async def analyze_image_with_prompt(self, image: Any, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
                        """Wrapper to use existing vision analyzer"""
                        try:
                            # Use the existing analyzer's analyze method
                            result = await self.analyzer.analyze_image_with_prompt(
                                image=image,
                                prompt=prompt
                            )
                            
                            # Extract the response text
                            if isinstance(result, dict):
                                # First check for 'content' key (from analyze_image_with_prompt)
                                if 'content' in result:
                                    return {'content': result['content']}
                                # Then check for description or response
                                return {'content': result.get('description', result.get('response', str(result)))}
                            else:
                                return {'content': str(result)}
                        except Exception as e:
                            logger.error(f"Vision analyzer error: {e}")
                            raise
                            
                claude_client = ClaudeVisionWrapper(vision_analyzer)
            else:
                # No vision analyzer available - use mock
                logger.warning("[PURE VISION] No vision analyzer available, using mock responses")
                claude_client = None
            
            # Initialize intelligence systems
            self.intelligence = PureVisionIntelligence(claude_client)
            self.proactive = ProactiveIntelligence(self.intelligence)
            self.workflow = WorkflowIntelligence(self.intelligence)
            
            logger.info("[PURE VISION] Intelligence systems initialized")
            
    async def handle_command(self, command_text: str) -> Dict[str, Any]:
        """
        Handle any vision command with pure intelligence.
        No pattern matching, no templates - Claude understands intent.
        """
        logger.info(f"[VISION] Handling command: {command_text}")
        await ws_logger.log(f"Processing vision command: {command_text}")
        
        # Ensure intelligence is initialized
        if not self.intelligence:
            await self.initialize_intelligence()
            
        # Capture current screen
        screenshot = await self._capture_screen()
        if not screenshot:
            # Even error messages come from Claude
            return await self._get_error_response("screenshot_failed", command_text)
            
        # Let Claude understand the command and respond naturally
        try:
            # Determine if this is a monitoring command through Claude
            is_monitoring_command = await self._is_monitoring_command(command_text, screenshot)
            
            if is_monitoring_command:
                return await self._handle_monitoring_command(command_text, screenshot)
            else:
                # Pure vision query - let Claude see and respond
                response = await self.intelligence.understand_and_respond(screenshot, command_text)
                
                return {
                    "handled": True,
                    "response": response,
                    "pure_intelligence": True,
                    "monitoring_active": self.monitoring_active,
                    "context": self.intelligence.context.get_temporal_context()
                }
                
        except Exception as e:
            logger.error(f"Intelligence error: {e}", exc_info=True)
            return await self._get_error_response("intelligence_error", command_text, str(e))
            
    async def _is_monitoring_command(self, command: str, screenshot: Any) -> bool:
        """Let Claude determine if this is a monitoring command"""
        prompt = f"""Look at the screen and the user's command: "{command}"

Is this command asking to start or stop screen monitoring/watching?
Respond with just "YES" or "NO".

Examples of monitoring commands:
- "start monitoring my screen"
- "stop watching"
- "activate vision monitoring"

Examples of non-monitoring commands:
- "what do you see?"
- "what's my battery?"
- "analyze this screen"
"""
        
        response = await self.intelligence._get_claude_vision_response(screenshot, prompt)
        return response.get('response', '').strip().upper() == 'YES'
        
    async def _handle_monitoring_command(self, command: str, screenshot: Any) -> Dict[str, Any]:
        """Handle monitoring commands with natural responses"""
        # Let Claude understand if this is start or stop
        intent_prompt = f"""The user said: "{command}"

Are they asking to START or STOP monitoring?
Respond with just "START" or "STOP".
"""
        
        response = await self.intelligence._get_claude_vision_response(screenshot, intent_prompt)
        intent = response.get('response', '').strip().upper()
        
        if intent == "START":
            self.monitoring_active = True
            self.proactive.monitoring_active = True
            
            # Get natural response for starting monitoring
            start_prompt = f"""The user asked: "{command}"

You're JARVIS. Respond naturally to confirm that you've started monitoring their screen.
Be specific about what you can see right now. Mention the purple indicator if appropriate.
Keep it natural and conversational - no generic phrases.
"""
            response = await self.intelligence._get_claude_vision_response(screenshot, start_prompt)
            
            # Start proactive monitoring
            asyncio.create_task(self._proactive_monitoring_loop())
            
        else:  # STOP
            self.monitoring_active = False
            self.proactive.monitoring_active = False
            
            # Get natural response for stopping monitoring
            stop_prompt = f"""The user asked: "{command}"

You're JARVIS. Respond naturally to confirm that you've stopped monitoring their screen.
Be conversational and natural - no generic phrases.
"""
            response = await self.intelligence._get_claude_vision_response(screenshot, stop_prompt)
            
        return {
            "handled": True,
            "response": response.get('response'),
            "monitoring_active": self.monitoring_active,
            "pure_intelligence": True
        }
        
    async def _proactive_monitoring_loop(self):
        """Proactive monitoring with pure intelligence"""
        logger.info("[VISION] Starting proactive monitoring loop")
        
        while self.monitoring_active:
            try:
                # Wait before next check
                await asyncio.sleep(5)
                
                if not self.monitoring_active:
                    break
                    
                # Capture screen and check for important changes
                screenshot = await self._capture_screen()
                if screenshot:
                    proactive_message = await self.proactive.observe_and_communicate(screenshot)
                    
                    if proactive_message and self.jarvis_api:
                        # Send proactive message through JARVIS voice
                        try:
                            await self.jarvis_api.speak_proactive(proactive_message)
                        except Exception as e:
                            logger.error(f"Failed to speak proactive message: {e}")
                            
            except Exception as e:
                logger.error(f"Proactive monitoring error: {e}")
                await asyncio.sleep(5)
                
    async def _capture_screen(self) -> Optional[Any]:
        """Capture current screen"""
        try:
            # Initialize vision manager if needed
            await self._ensure_vision_manager()
            
            if self.vision_manager and hasattr(self.vision_manager, 'vision_analyzer') and self.vision_manager.vision_analyzer:
                screenshot = await self.vision_manager.vision_analyzer.capture_screen()
                return screenshot
            else:
                # Try to capture screen directly as fallback
                logger.info("[VISION] Attempting direct screen capture...")
                try:
                    # Try macOS screencapture
                    import subprocess
                    import tempfile
                    from PIL import Image
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    # Capture screen
                    result = subprocess.run(
                        ['screencapture', '-x', tmp_path],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        # Load and return image
                        screenshot = Image.open(tmp_path)
                        os.unlink(tmp_path)  # Clean up
                        logger.info("[VISION] Direct screen capture successful")
                        return screenshot
                    else:
                        logger.error(f"screencapture failed: {result.stderr}")
                        
                except Exception as e:
                    logger.error(f"Direct capture failed: {e}")
                    
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            
        return None
        
    async def _ensure_vision_manager(self):
        """Initialize vision manager if not already done"""
        if not self.vision_manager:
            try:
                logger.info("[VISION INIT] Attempting to import vision_manager...")
                try:
                    from api.vision_websocket import vision_manager
                except ImportError:
                    from .vision_websocket import vision_manager
                    
                self.vision_manager = vision_manager
                logger.info(f"[VISION INIT] Vision manager imported: {vision_manager}")
                
                # Check if vision_analyzer needs initialization
                if hasattr(vision_manager, 'vision_analyzer') and vision_manager.vision_analyzer is None:
                    logger.info("[VISION INIT] Vision analyzer is None, checking app state...")
                    # Try to get from app state
                    try:
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from main import app
                        if hasattr(app.state, 'vision_analyzer'):
                            vision_manager.vision_analyzer = app.state.vision_analyzer
                            logger.info("[VISION INIT] Set vision analyzer from app state")
                    except Exception as e:
                        logger.error(f"[VISION INIT] Failed to get vision analyzer from app state: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to initialize vision manager: {e}")
                
    async def _get_error_response(self, error_type: str, command: str, details: str = "") -> Dict[str, Any]:
        """Even errors are communicated naturally by Claude"""
        error_prompt = f"""The user asked: "{command}"

An error occurred: {error_type}
{f"Details: {details}" if details else ""}

You're JARVIS. Respond naturally to explain the issue and suggest a solution.
Be helpful and specific, but keep it conversational.
Never use generic error messages or technical jargon.
"""
        
        # Use mock response if no Claude client
        if self.intelligence and self.intelligence.claude:
            response = await self.intelligence._get_claude_vision_response(None, error_prompt)
            error_message = response.get('response')
        else:
            # Natural fallback
            if error_type == "screenshot_failed":
                error_message = "I'm having trouble accessing your screen right now, Sir. Let me check the vision system configuration."
            elif error_type == "intelligence_error":
                error_message = "I encountered an issue processing that request. Let me recalibrate the vision systems."
            else:
                error_message = f"Something went wrong with that request, Sir. {details if details else 'Let me investigate.'}"
                
        return {
            "handled": True,
            "response": error_message,
            "error": True,
            "pure_intelligence": True,
            "monitoring_active": self.monitoring_active
        }
        
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        if self.intelligence and self.intelligence.context:
            return {
                "conversation_length": len(self.intelligence.context.history),
                "monitoring_active": self.monitoring_active,
                "workflow_state": self.intelligence.context.workflow_state,
                "emotional_state": self.intelligence.context.emotional_context.value if self.intelligence.context.emotional_context else "neutral"
            }
        return {
            "conversation_length": 0,
            "monitoring_active": False,
            "workflow_state": "unknown",
            "emotional_state": "neutral"
        }

# Singleton instance
vision_command_handler = VisionCommandHandler()