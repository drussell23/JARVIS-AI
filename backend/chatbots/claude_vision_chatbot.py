"""
Claude Vision-enabled Chatbot for JARVIS
Extends the basic Claude chatbot with vision capabilities for screen analysis
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import base64
import io
from PIL import Image
import numpy as np

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
    from vision.claude_vision_analyzer import ClaudeVisionAnalyzer
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
            
        # Initialize vision analyzer
        if VISION_ANALYZER_AVAILABLE and self.api_key:
            self.vision_analyzer = ClaudeVisionAnalyzer(self.api_key)
        else:
            self.vision_analyzer = None
            
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10
        
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
        """Capture the current screen"""
        # Try pyautogui first
        if SCREENSHOT_AVAILABLE:
            try:
                screenshot = pyautogui.screenshot()
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
            # Capture screenshot
            screenshot = await self.capture_screenshot()
            if not screenshot:
                return "I apologize, sir, but I'm unable to capture your screen at the moment. Please ensure screen recording permissions are enabled in System Preferences > Security & Privacy > Privacy > Screen Recording."
                
            # Convert to base64 for Claude API
            buffer = io.BytesIO()
            screenshot.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Build the vision message
            messages = self._build_messages_with_vision(user_input, image_base64)
            
            # Make API call with vision
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
            logger.info(f"Claude vision API call completed in {response_time:.2f}s")
            
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
                        "media_type": "image/png",
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
        
    async def generate_response(self, user_input: str) -> str:
        """
        Process user input and generate response, using vision when appropriate
        """
        if not self.is_available():
            return "Claude API is not available. Please install anthropic package and set API key."
            
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