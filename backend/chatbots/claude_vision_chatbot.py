"""
Claude Vision-enabled Chatbot for JARVIS
Extends the basic Claude chatbot with vision capabilities for screen analysis
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import base64
import io
from PIL import Image
import numpy as np
import hashlib

logger = logging.getLogger(__name__)

# Check if required packages are available
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not installed. Install with: pip install anthropic")

try:
    import pyautogui
    SCREENSHOT_AVAILABLE = True
except ImportError:
    SCREENSHOT_AVAILABLE = False
    logger.warning("PyAutoGUI not installed. Install with: pip install pyautogui")

try:
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    VISION_ANALYZER_AVAILABLE = True
except ImportError:
    VISION_ANALYZER_AVAILABLE = False
    logger.warning("Claude vision analyzer not available")


class ClaudeVisionChatbot:
    """
    Vision-enabled Claude chatbot that can analyze screen content
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",  # Vision-capable model
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """Initialize Claude vision chatbot"""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter"
            )
            
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Enhanced JARVIS system prompt with vision capabilities
        self.system_prompt = system_prompt or """You are JARVIS, an intelligent AI assistant inspired by Tony Stark's AI from Iron Man. 
You have advanced vision capabilities and can see and analyze the user's screen when asked.
You are helpful, witty, and highly capable. You speak with a refined, professional tone while being personable and occasionally adding subtle humor. 
When analyzing screens, you provide detailed, accurate descriptions and helpful insights about what you observe.
You excel at understanding context and providing insightful, well-structured responses."""
        
        # Initialize client
        if ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
            
        # Initialize vision analyzer with real-time capabilities
        if VISION_ANALYZER_AVAILABLE and self.api_key:
            # Enable real-time monitoring by default for video streaming
            self.vision_analyzer = ClaudeVisionAnalyzer(
                self.api_key, 
                enable_realtime=True
            )
            logger.info("Initialized Claude Vision Analyzer with real-time monitoring capabilities")
        else:
            self.vision_analyzer = None
            logger.warning("Vision analyzer not available - real-time monitoring disabled")
            
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10
        
        # Monitoring state
        self._monitoring_active = False
        self._capture_method = 'unknown'
        
        # Screenshot cache (cache for 5 seconds to handle repeated requests)
        self._screenshot_cache = None
        self._screenshot_cache_time = None
        self._screenshot_cache_duration = timedelta(seconds=5)
        
        # Vision command patterns
        self.vision_patterns = [
            "can you see my screen",
            "what's on my screen",
            "what is on my screen",
            "analyze my screen",
            "describe my screen",
            "look at my screen",
            "check my screen",
            "show me what you see",
            "what do you see",
            "screen analysis",
            "analyze what i'm looking at"
        ]
        
    def is_vision_command(self, user_input: str) -> bool:
        """Check if the user input is asking for vision/screen analysis"""
        input_lower = user_input.lower()
        return any(pattern in input_lower for pattern in self.vision_patterns)
        
    async def capture_screenshot(self) -> Optional[Image.Image]:
        """Capture the current screen with caching"""
        # Check cache first
        if (self._screenshot_cache is not None and 
            self._screenshot_cache_time is not None and 
            datetime.now() - self._screenshot_cache_time < self._screenshot_cache_duration):
            logger.info("Using cached screenshot")
            return self._screenshot_cache
            
        # Try pyautogui first
        if SCREENSHOT_AVAILABLE:
            try:
                # Run in thread pool to avoid blocking
                screenshot = await asyncio.to_thread(pyautogui.screenshot)
                # Cache the screenshot
                self._screenshot_cache = screenshot
                self._screenshot_cache_time = datetime.now()
                return screenshot
            except Exception as e:
                logger.warning(f"PyAutoGUI screenshot failed: {e}")
                
        # Fallback to macOS native screen capture
        try:
            from vision.screen_capture_module import capture_screen_native
            screenshot_path = capture_screen_native()
            if screenshot_path:
                return Image.open(screenshot_path)
        except ImportError:
            logger.warning("Native screen capture not available")
        except Exception as e:
            logger.error(f"Native screenshot failed: {e}")
            
        # Final fallback - try using screencapture command
        try:
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                subprocess.run(['screencapture', '-x', tmp.name], check=True)
                return Image.open(tmp.name)
        except Exception as e:
            logger.error(f"All screenshot methods failed: {e}")
            return None
            
    async def analyze_screen_with_vision(self, user_input: str) -> str:
        """Analyze the screen using Claude's vision capabilities"""
        try:
            total_start = datetime.now()
            
            # Capture screenshot
            capture_start = datetime.now()
            screenshot = await self.capture_screenshot()
            capture_time = (datetime.now() - capture_start).total_seconds()
            logger.info(f"Screenshot capture took {capture_time:.2f}s")
            
            if not screenshot:
                return "I apologize, sir, but I'm unable to capture your screen at the moment. Please ensure screen recording permissions are enabled in System Preferences > Security & Privacy > Privacy > Screen Recording."
                
            # Resize image for faster processing if it's too large
            encode_start = datetime.now()
            max_dimension = 1920  # Reasonable size for Claude
            if screenshot.width > max_dimension or screenshot.height > max_dimension:
                ratio = min(max_dimension / screenshot.width, max_dimension / screenshot.height)
                new_size = (int(screenshot.width * ratio), int(screenshot.height * ratio))
                screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized screenshot to {new_size[0]}x{new_size[1]}")
            
            # Convert to base64 for Claude API with optimization
            buffer = io.BytesIO()
            # Convert RGBA to RGB if needed for JPEG
            if screenshot.mode == 'RGBA':
                # Create a white background
                rgb_image = Image.new('RGB', screenshot.size, (255, 255, 255))
                rgb_image.paste(screenshot, mask=screenshot.split()[3])  # Use alpha channel as mask
                screenshot = rgb_image
            # Use JPEG for smaller file size and faster encoding
            screenshot.save(buffer, format="JPEG", quality=85, optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            encode_time = (datetime.now() - encode_start).total_seconds()
            logger.info(f"Image encoding took {encode_time:.2f}s")
            
            # Build the vision message
            messages = self._build_messages_with_vision(user_input, image_base64)
            
            # Make API call with vision
            api_start = datetime.now()
            
            # Use a faster model for quick vision checks if available
            vision_model = self.model
            if "can you see" in user_input.lower() and "claude-3-haiku" in self.model:
                # For simple "can you see" queries, we can use the same model
                pass
            
            # For vision requests, we can't stream unfortunately
            # But we can use a shorter system prompt for faster processing
            quick_system_prompt = "You are JARVIS. You can see the user's screen. Be concise and helpful." if "can you see" in user_input.lower() else self.system_prompt
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=vision_model,
                max_tokens=256 if "can you see" in user_input.lower() else self.max_tokens,
                temperature=0.3 if "can you see" in user_input.lower() else self.temperature,
                system=quick_system_prompt,
                messages=messages
            )
            
            # Extract response
            ai_response = response.content[0].text
            
            # Log performance
            api_time = (datetime.now() - api_start).total_seconds()
            total_time = (datetime.now() - total_start).total_seconds()
            logger.info(f"Claude API call took {api_time:.2f}s")
            logger.info(f"Total vision processing took {total_time:.2f}s")
            
            # Update conversation history
            self._update_history(user_input, ai_response)
            
            return ai_response
            
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            return f"I encountered an API error while analyzing your screen: {str(e)}"
        except Exception as e:
            logger.error(f"Error in screen analysis: {e}")
            return f"I apologize, sir, but I encountered an error analyzing your screen: {str(e)}"
            
    def _build_messages_with_vision(self, user_input: str, image_base64: str) -> List[Dict[str, Any]]:
        """Build message list for Claude API including vision content"""
        messages = []
        
        # Add conversation history (last 3 exchanges for context)
        for entry in self.conversation_history[-3:]:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
            
        # Add current user input with image
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": user_input
                }
            ]
        })
        
        return messages
        
    async def _is_monitoring_command(self, user_input: str) -> bool:
        """Check if this is a continuous monitoring command"""
        monitoring_keywords = [
            'monitor', 'monitoring', 'watch', 'watching', 'track', 'tracking',
            'continuous', 'continuously', 'real-time', 'realtime', 'actively',
            'surveillance', 'observe', 'observing', 'stream', 'streaming'
        ]
        
        screen_keywords = ['screen', 'display', 'desktop', 'workspace', 'monitor']
        
        text_lower = user_input.lower()
        has_monitoring = any(keyword in text_lower for keyword in monitoring_keywords)
        has_screen = any(keyword in text_lower for keyword in screen_keywords)
        
        return has_monitoring and has_screen
        
    async def _handle_monitoring_command(self, user_input: str) -> str:
        """Handle continuous monitoring commands"""
        text_lower = user_input.lower()
        logger.info(f"[MONITORING] Processing monitoring command: {user_input}")
        
        # Check if we have the enhanced vision analyzer with video streaming
        if self.vision_analyzer is not None:
            logger.info("[MONITORING] Vision analyzer is available, attempting to handle monitoring")
            try:
                # Use the already initialized vision analyzer
                # Handle different monitoring commands
                if any(word in text_lower for word in ['start', 'enable', 'activate', 'begin', 'turn on']):
                    logger.info("[MONITORING] Starting video streaming...")
                    # Start video streaming
                    result = await self.vision_analyzer.start_video_streaming()
                    logger.info(f"[MONITORING] Video streaming result: {result}")
                    if result.get('success'):
                        # Update monitoring state
                        self._monitoring_active = True
                        capture_method = result.get('metrics', {}).get('capture_method', 'unknown')
                        self._capture_method = capture_method
                        
                        # Take a screenshot and describe what we see
                        screenshot = await self.vision_analyzer.capture_screen()
                        if screenshot:
                            # Analyze the current screen
                            analysis = await self.vision_analyzer.analyze_screenshot(
                                screenshot,
                                "Describe what you see on the screen in detail."
                            )
                            
                            if capture_method == 'macos_native':
                                return f"I've started monitoring your screen with native macOS capture at 30 FPS. The purple recording indicator should now be visible.\n\nCurrently, I can see: {analysis[1] if isinstance(analysis, tuple) else str(analysis)}\n\nI'll continue watching for any changes or important events."
                            else:
                                return f"I've started monitoring your screen in {capture_method} mode at 30 FPS.\n\nCurrently, I can see: {analysis[1] if isinstance(analysis, tuple) else str(analysis)}\n\nI'll continue watching for any changes or important events."
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        logger.error(f"[MONITORING] Failed to start video streaming: {error_msg}")
                        return f"I encountered an issue starting video streaming: {error_msg}. Let me try with standard screenshot monitoring instead."
                        
                elif any(word in text_lower for word in ['stop', 'disable', 'deactivate', 'end', 'turn off']):
                    # Stop video streaming
                    result = await self.vision_analyzer.stop_video_streaming()
                    if result.get('success'):
                        self._monitoring_active = False
                        self._capture_method = 'unknown'
                        return "I've stopped monitoring your screen. The video streaming has been disabled and the recording indicator should have disappeared."
                    else:
                        self._monitoring_active = False
                        self._capture_method = 'unknown'
                        return "The screen monitoring appears to be already stopped."
                        
                else:
                    # Generic monitoring request - start monitoring and describe
                    result = await self.vision_analyzer.start_video_streaming()
                    if result.get('success'):
                        # Update monitoring state
                        self._monitoring_active = True
                        self._capture_method = result.get('metrics', {}).get('capture_method', 'unknown')
                        
                        # Analyze for 5 seconds
                        analysis_result = await self.vision_analyzer.analyze_video_stream(
                            "Monitor the screen and describe any changes or important elements you see.",
                            duration_seconds=5.0
                        )
                        
                        if analysis_result.get('success'):
                            frames_analyzed = analysis_result.get('frames_analyzed', 0)
                            descriptions = []
                            
                            if 'results' in analysis_result:
                                for result in analysis_result['results'][:3]:  # First 3 analyses
                                    if 'analysis' in result:
                                        descriptions.append(str(result['analysis']))
                            
                            response = f"I'm now continuously monitoring your screen at 30 FPS. I've analyzed {frames_analyzed} frames in the last 5 seconds.\n\n"
                            
                            if descriptions:
                                response += "Here's what I observed:\n" + "\n".join(f"• {desc[:100]}..." for desc in descriptions)
                            else:
                                response += "I'm watching your screen for any changes or important events."
                                
                            return response
                        else:
                            return "I've started monitoring your screen. I'll watch for changes and alert you to anything important."
                            
            except Exception as e:
                logger.error(f"Error in monitoring command: {e}")
                return f"I encountered an error setting up continuous monitoring: {str(e)}. Let me fall back to standard screenshot analysis."
                
        else:
            logger.warning("[MONITORING] Vision analyzer is None - cannot start monitoring")
        
        # Fallback response if enhanced analyzer not available
        return "I'll need the enhanced vision system to enable continuous monitoring. Currently, I can only take screenshots on demand. Please ensure the vision system is properly initialized."
        
    async def generate_response(self, user_input: str) -> str:
        """
        Process user input and generate response, using vision when appropriate
        """
        if not self.is_available():
            return "Claude API is not available. Please install anthropic package and set API key."
            
        # Check for continuous monitoring commands
        is_monitoring = await self._is_monitoring_command(user_input)
        logger.info(f"[VISION DEBUG] Is monitoring command: {is_monitoring} for input: {user_input}")
        
        if is_monitoring:
            logger.info(f"[VISION DEBUG] Routing to _handle_monitoring_command")
            return await self._handle_monitoring_command(user_input)
            
        # Check if this is a vision command
        if self.is_vision_command(user_input):
            logger.info(f"Vision command detected: {user_input}")
            return await self.analyze_screen_with_vision(user_input)
            
        # Otherwise, use regular text processing
        try:
            # Build messages for the API
            messages = self._build_messages(user_input)
            
            # Make API call
            start_time = datetime.now()
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages
            )
            
            # Extract response
            ai_response = response.content[0].text
            
            # Log performance
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Claude API call completed in {response_time:.2f}s")
            
            # Update conversation history
            self._update_history(user_input, ai_response)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error: {str(e)}"
            
    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Build message list for Claude API including conversation history"""
        messages = []
        
        # Add conversation history
        for entry in self.conversation_history[-5:]:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
            
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
        
    def _update_history(self, user_input: str, ai_response: str):
        """Update conversation history"""
        self.conversation_history.append({
            "user": user_input,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
            
    async def generate_response_with_context(self, user_input: str) -> Dict[str, Any]:
        """Generate response with additional context information"""
        response = await self.generate_response(user_input)
        
        return {
            "response": response,
            "conversation_id": "claude-vision",
            "message_count": len(self.conversation_history),
            "vision_capable": True,
            "model": self.model
        }
        
    async def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
        
    def is_available(self) -> bool:
        """Check if Claude API is available"""
        return ANTHROPIC_AVAILABLE and self.client is not None
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "model": self.model,
            "history_length": len(self.conversation_history),
            "vision_capable": True,
            "screenshot_available": SCREENSHOT_AVAILABLE
        }
        
    @property
    def model_name(self) -> str:
        """Get model name for compatibility"""
        return self.model
        
    async def generate_response_stream(self, user_input: str):
        """Generate streaming response (falls back to non-streaming for vision)"""
        response = await self.generate_response(user_input)
        # Simulate streaming by yielding chunks
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            yield response[i:i+chunk_size]
            await asyncio.sleep(0.01)
            
    async def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history
        
    def set_system_prompt(self, prompt: str):
        """Update system prompt"""
        self.system_prompt = prompt
        
    async def get_response(self, prompt: str) -> str:
        """Alias for generate_response for compatibility"""
        return await self.generate_response(prompt)