"""
Claude Vision-enabled Chatbot for JARVIS
Extends the basic Claude chatbot with vision capabilities for screen analysis
"""

# Fix import path for vision modules
import sys
import os
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

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

# Platform and system detection
import platform
import subprocess
import tempfile
import re
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

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
except (ImportError, AttributeError) as e:
    SCREENSHOT_AVAILABLE = False
    logger.warning(f"PyAutoGUI not available: {e}. Screenshot fallback disabled.")

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
        system_prompt: Optional[str] = None,
        vision_analyzer: Optional[Any] = None  # Allow passing existing analyzer
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
        
        # Dynamic JARVIS system prompt with real-time context
        self._initialize_dynamic_system_prompt(system_prompt)
        
        # Initialize client
        if ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
            
        # Use provided vision analyzer or initialize new one
        if vision_analyzer:
            # Use the provided analyzer (e.g., from app.state)
            self.vision_analyzer = vision_analyzer
            logger.info("Using provided vision analyzer instance")
        elif VISION_ANALYZER_AVAILABLE and self.api_key:
            # Create new analyzer with real-time monitoring by default
            self.vision_analyzer = ClaudeVisionAnalyzer(
                self.api_key, 
                enable_realtime=True
            )
            logger.info("Initialized new Claude Vision Analyzer with real-time monitoring capabilities")
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
        
        # Dynamic vision detection configuration
        self._initialize_vision_detection_system()
        
        # Platform-specific configuration
        self._platform = platform.system().lower()
        self._capture_methods = self._initialize_capture_methods()
        
        # Dynamic cache configuration
        self._cache_config = self._get_cache_configuration()
        
    def is_vision_command(self, user_input: str) -> bool:
        """Enhanced vision command detection with intent analysis"""
        input_lower = user_input.lower().strip()
        
        # Quick keyword pre-check for performance
        if not any(keyword in input_lower for keyword in self._vision_keywords):
            return False
            
        # Check regex patterns for more accurate detection
        for pattern in self._compiled_vision_patterns:
            if pattern.search(input_lower):
                # Analyze intent for later use
                self._last_vision_intent = self._analyze_vision_intent(input_lower)
                return True
                
        # Fuzzy matching for typos and variations
        words = input_lower.split()
        for word in words:
            for keyword in self._vision_keywords:
                # Allow for typos (edit distance)
                if self._similar_words(word, keyword, threshold=0.8):
                    self._last_vision_intent = self._analyze_vision_intent(input_lower)
                    return True
                    
        return False
        
    async def capture_screenshot(self) -> Optional[Image.Image]:
        """Enhanced screenshot capture with multiple methods and smart caching"""
        # Generate cache key based on intent
        cache_key = self._generate_cache_key()
        
        # Check intelligent cache
        cached_screenshot = await self._get_cached_screenshot(cache_key)
        if cached_screenshot:
            logger.info(f"Using cached screenshot (key: {cache_key})")
            return cached_screenshot
            
        # Try capture methods in platform-specific order
        screenshot = None
        used_method = None
        
        for method_info in self._capture_methods:
            method_name = method_info['name']
            method_func = method_info['function']
            
            try:
                logger.debug(f"Attempting screenshot capture via {method_name}")
                screenshot = await method_func()
                
                if screenshot:
                    used_method = method_name
                    logger.info(f"Successfully captured screenshot using {method_name}")
                    break
                    
            except Exception as e:
                logger.warning(f"{method_name} capture failed: {e}")
                continue
                
        if not screenshot:
            logger.error("All screenshot capture methods failed")
            return None
            
        # Post-process screenshot based on intent
        if hasattr(self, '_last_vision_intent'):
            screenshot = await self._optimize_screenshot(screenshot, self._last_vision_intent)
            
        # Cache the screenshot
        await self._cache_screenshot(cache_key, screenshot, used_method)
        
        return screenshot
            
    async def analyze_screen_with_vision(self, user_input: str) -> str:
        """Enhanced screen analysis with dynamic optimization"""
        try:
            total_start = datetime.now()
            
            # Capture screenshot
            capture_start = datetime.now()
            screenshot = await self.capture_screenshot()
            capture_time = (datetime.now() - capture_start).total_seconds()
            logger.info(f"Screenshot capture took {capture_time:.2f}s")
            
            if not screenshot:
                return "I apologize, sir, but I'm unable to capture your screen at the moment. Please ensure screen recording permissions are enabled in System Preferences > Security & Privacy > Privacy > Screen Recording."
                
            # Dynamic image optimization based on intent
            encode_start = datetime.now()
            screenshot = await self._prepare_screenshot_for_api(screenshot, user_input)
            
            # Convert to base64 for Claude API with optimization
            buffer = io.BytesIO()
            # Convert RGBA to RGB if needed for JPEG
            if screenshot.mode == 'RGBA':
                # Create a white background
                rgb_image = Image.new('RGB', screenshot.size, (255, 255, 255))
                rgb_image.paste(screenshot, mask=screenshot.split()[3])  # Use alpha channel as mask
                screenshot = rgb_image
            # Dynamic format and quality based on intent
            format_config = self._get_image_format_config(user_input)
            screenshot.save(
                buffer, 
                format=format_config['format'], 
                quality=format_config['quality'], 
                optimize=format_config['optimize']
            )
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            encode_time = (datetime.now() - encode_start).total_seconds()
            logger.info(f"Image encoding took {encode_time:.2f}s (format: {format_config['format']}, quality: {format_config['quality']})")
            
            # Build the vision message
            messages = self._build_messages_with_vision(user_input, image_base64)
            
            # Make API call with vision
            api_start = datetime.now()
            
            # Use a faster model for quick vision checks if available
            vision_model = self.model
            if "can you see" in user_input.lower() and "claude-3-haiku" in self.model:
                # For simple "can you see" queries, we can use the same model
                pass
            
            # Dynamic prompt generation based on query analysis
            system_prompt = self._generate_vision_system_prompt(user_input)
            
            # Dynamic API configuration based on intent
            api_config = self._get_vision_api_config(user_input)
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=api_config['model'],
                max_tokens=api_config['max_tokens'],
                temperature=api_config['temperature'],
                system=system_prompt,
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
            
    def _initialize_dynamic_system_prompt(self, custom_prompt: Optional[str] = None):
        """Initialize system prompt with dynamic context"""
        # Get dynamic time context
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        timezone = self._get_timezone_name()
        
        # Build dynamic prompt components
        base_components = [
            "You are JARVIS, an intelligent AI assistant inspired by Tony Stark's AI from Iron Man.",
            "You have advanced vision capabilities and can see and analyze the user's screen when asked.",
            "You are helpful, witty, and highly capable.",
            "You speak with a refined, professional tone while being personable and occasionally adding subtle humor.",
            "When analyzing screens, you provide detailed, accurate descriptions and helpful insights about what you observe.",
            "You excel at understanding context and providing insightful, well-structured responses."
        ]
        
        # Add dynamic context
        context_components = [
            f"Current date and time: {current_datetime}",
            f"Timezone: {timezone}" if timezone else None,
            f"Platform: {platform.system()} {platform.release()}"
        ]
        
        # Filter out None values and join
        all_components = base_components + [c for c in context_components if c]
        self.system_prompt = custom_prompt or " ".join(all_components)
    
    def _initialize_vision_detection_system(self):
        """Initialize dynamic vision pattern detection"""
        # Core vision keywords for quick checks
        self._vision_keywords = {
            'screen', 'see', 'look', 'view', 'show', 'display', 'vision',
            'visual', 'analyze', 'examine', 'check', 'monitor', 'desktop',
            'window', 'describe', 'watch', 'observe', 'inspect'
        }
        
        # Build regex patterns dynamically
        self._compiled_vision_patterns = self._compile_vision_patterns()
        
    def _compile_vision_patterns(self) -> List[re.Pattern]:
        """Compile vision detection patterns dynamically"""
        patterns = []
        
        # Action + target patterns
        action_words = ['see', 'look', 'view', 'show', 'analyze', 'check', 'examine', 'describe']
        target_words = ['screen', 'display', 'monitor', 'desktop', 'window']
        
        for action in action_words:
            for target in target_words:
                patterns.extend([
                    rf'\b{action}\s+(?:my\s+|the\s+)?{target}\b',
                    rf'\bcan\s+you\s+{action}\s+(?:my\s+|the\s+)?{target}\b',
                    rf'\b{action}\s+what(?:\'s|\s+is)\s+on\s+(?:my\s+|the\s+)?{target}\b'
                ])
        
        # Question patterns
        patterns.extend([
            r'\bwhat\s+do\s+you\s+see\b',
            r'\bwhat(?:\'s|\s+is)\s+on\s+(?:my\s+)?screen\b',
            r'\btell\s+me\s+what\s+you\s+see\b',
            r'\banalyze\s+this\b',
            r'\blook\s+at\s+this\b',
            r'\bscreen\s*shot\b',
            r'\bvisual\s+analysis\b'
        ])
        
        # Compile all patterns with case-insensitive flag
        return [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def _initialize_capture_methods(self) -> List[Dict[str, Any]]:
        """Initialize platform-specific capture methods"""
        methods = []
        
        # PyAutoGUI (cross-platform)
        if SCREENSHOT_AVAILABLE:
            methods.append({
                'name': 'pyautogui',
                'function': self._capture_pyautogui,
                'priority': 1
            })
        
        # Platform-specific methods
        if self._platform == 'darwin':  # macOS
            methods.extend([
                {
                    'name': 'native_macos',
                    'function': self._capture_native_macos,
                    'priority': 2
                },
                {
                    'name': 'screencapture_cmd',
                    'function': self._capture_screencapture_cmd,
                    'priority': 3
                }
            ])
        elif self._platform == 'win32':  # Windows
            methods.append({
                'name': 'windows_capture',
                'function': self._capture_windows,
                'priority': 2
            })
        elif 'linux' in self._platform:  # Linux
            methods.append({
                'name': 'linux_capture',
                'function': self._capture_linux,
                'priority': 2
            })
        
        # Vision analyzer as fallback
        if self.vision_analyzer:
            methods.append({
                'name': 'vision_analyzer',
                'function': self._capture_vision_analyzer,
                'priority': 99
            })
        
        # Sort by priority
        return sorted(methods, key=lambda x: x['priority'])
    
    def _get_cache_configuration(self) -> Dict[str, Any]:
        """Get dynamic cache configuration"""
        return {
            'duration': timedelta(seconds=5),
            'max_size': 10,
            'strategy': 'lru',  # Least Recently Used
            'hash_method': 'md5'
        }
    
    def _analyze_vision_intent(self, text: str) -> Dict[str, Any]:
        """Analyze the intent behind a vision query"""
        intent = {
            'urgency': 'normal',
            'detail_level': 'standard',
            'focus_areas': [],
            'query_type': 'general',
            'requires_interaction': False
        }
        
        # Detect urgency
        if any(word in text for word in ['urgent', 'quick', 'fast', 'immediately', 'asap']):
            intent['urgency'] = 'high'
        
        # Detect detail level
        if any(word in text for word in ['detail', 'comprehensive', 'thorough', 'everything', 'full']):
            intent['detail_level'] = 'high'
        elif any(word in text for word in ['brief', 'summary', 'quick', 'glance']):
            intent['detail_level'] = 'low'
        
        # Detect focus areas
        if any(word in text for word in ['error', 'bug', 'issue', 'problem']):
            intent['focus_areas'].append('errors')
        if any(word in text for word in ['code', 'program', 'script']):
            intent['focus_areas'].append('code')
        if any(word in text for word in ['text', 'document', 'content']):
            intent['focus_areas'].append('text')
        
        # Detect query type
        if 'can you see' in text.lower():
            intent['query_type'] = 'confirmation'
        elif any(word in text for word in ['analyze', 'examine']):
            intent['query_type'] = 'analysis'
        
        return intent
    
    def _similar_words(self, word1: str, word2: str, threshold: float = 0.8) -> bool:
        """Check if two words are similar (for typo detection)"""
        # Simple similarity check - can be enhanced with edit distance
        if word1 == word2:
            return True
        
        # Check if one is substring of other
        if len(word1) > 3 and len(word2) > 3:
            if word1 in word2 or word2 in word1:
                return True
        
        # Basic edit distance approximation
        if abs(len(word1) - len(word2)) > 2:
            return False
        
        matches = sum(1 for a, b in zip(word1, word2) if a == b)
        similarity = matches / max(len(word1), len(word2))
        
        return similarity >= threshold
    
    async def _capture_pyautogui(self) -> Optional[Image.Image]:
        """Capture using PyAutoGUI"""
        if not SCREENSHOT_AVAILABLE:
            return None
        try:
            return await asyncio.to_thread(pyautogui.screenshot)
        except Exception as e:
            raise Exception(f"PyAutoGUI capture failed: {e}")
    
    async def _capture_native_macos(self) -> Optional[Image.Image]:
        """Capture using native macOS methods"""
        try:
            from vision.screen_capture_module import capture_screen_native
            screenshot_path = capture_screen_native()
            if screenshot_path:
                return Image.open(screenshot_path)
        except:
            pass
        return None
    
    async def _capture_screencapture_cmd(self) -> Optional[Image.Image]:
        """Capture using macOS screencapture command"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cmd = ['screencapture', '-x']
                # Add cursor if needed based on intent
                if hasattr(self, '_last_vision_intent') and self._last_vision_intent.get('show_cursor'):
                    cmd.append('-C')
                cmd.append(tmp.name)
                
                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await result.communicate()
                
                if result.returncode == 0:
                    return Image.open(tmp.name)
        except:
            pass
        return None
    
    async def _capture_windows(self) -> Optional[Image.Image]:
        """Capture on Windows"""
        try:
            from PIL import ImageGrab
            return await asyncio.to_thread(ImageGrab.grab)
        except:
            return None
    
    async def _capture_linux(self) -> Optional[Image.Image]:
        """Capture on Linux"""
        commands = [
            ['gnome-screenshot', '-f'],
            ['scrot'],
            ['import', '-window', 'root']
        ]
        
        for cmd_template in commands:
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    cmd = cmd_template + [tmp.name]
                    result = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await result.communicate()
                    
                    if result.returncode == 0:
                        return Image.open(tmp.name)
            except:
                continue
        return None
    
    async def _capture_vision_analyzer(self) -> Optional[Image.Image]:
        """Capture using vision analyzer"""
        if self.vision_analyzer and hasattr(self.vision_analyzer, 'capture_screen'):
            return await self.vision_analyzer.capture_screen()
        return None
    
    def _generate_cache_key(self) -> str:
        """Generate intelligent cache key"""
        components = [
            str(datetime.now().timestamp()),
            getattr(self, '_last_vision_intent', {}).get('query_type', 'general')
        ]
        
        key_string = "_".join(components)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    async def _get_cached_screenshot(self, cache_key: str) -> Optional[Image.Image]:
        """Get screenshot from intelligent cache"""
        if not hasattr(self, '_screenshot_cache_store'):
            self._screenshot_cache_store = {}
        
        cached = self._screenshot_cache_store.get(cache_key)
        if cached:
            timestamp, screenshot, metadata = cached
            cache_duration = self._cache_config['duration']
            
            # Check if cache is still valid
            if datetime.now() - timestamp < cache_duration:
                # Extend cache for frequently accessed items
                if metadata.get('access_count', 0) > 3:
                    cache_duration *= 2
                
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                return screenshot
        
        return None
    
    async def _cache_screenshot(self, cache_key: str, screenshot: Image.Image, method: str):
        """Cache screenshot with metadata"""
        if not hasattr(self, '_screenshot_cache_store'):
            self._screenshot_cache_store = {}
        
        # Implement LRU cache
        max_size = self._cache_config['max_size']
        if len(self._screenshot_cache_store) >= max_size:
            # Remove least recently used
            oldest_key = min(self._screenshot_cache_store.keys(), 
                           key=lambda k: self._screenshot_cache_store[k][2].get('last_access', datetime.min))
            del self._screenshot_cache_store[oldest_key]
        
        self._screenshot_cache_store[cache_key] = (
            datetime.now(),
            screenshot,
            {
                'method': method,
                'access_count': 0,
                'last_access': datetime.now()
            }
        )
    
    async def _optimize_screenshot(self, screenshot: Image.Image, intent: Dict[str, Any]) -> Image.Image:
        """Optimize screenshot based on intent analysis"""
        # Dynamic resizing based on detail level
        if intent.get('detail_level') == 'low':
            max_dimension = 1280
        elif intent.get('detail_level') == 'high':
            max_dimension = 2560
        else:
            max_dimension = 1920
        
        # Resize if needed
        if screenshot.width > max_dimension or screenshot.height > max_dimension:
            ratio = min(max_dimension / screenshot.width, max_dimension / screenshot.height)
            new_size = (int(screenshot.width * ratio), int(screenshot.height * ratio))
            screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Optimized screenshot to {new_size} based on intent")
        
        return screenshot
    
    async def _prepare_screenshot_for_api(self, screenshot: Image.Image, user_input: str) -> Image.Image:
        """Prepare screenshot for API submission"""
        intent = getattr(self, '_last_vision_intent', {})
        
        # Apply optimizations
        screenshot = await self._optimize_screenshot(screenshot, intent)
        
        # Convert RGBA to RGB if needed
        if screenshot.mode == 'RGBA':
            rgb_image = Image.new('RGB', screenshot.size, (255, 255, 255))
            rgb_image.paste(screenshot, mask=screenshot.split()[3])
            screenshot = rgb_image
        
        return screenshot
    
    def _get_image_format_config(self, user_input: str) -> Dict[str, Any]:
        """Get dynamic image format configuration"""
        intent = getattr(self, '_last_vision_intent', {})
        
        # High quality for detailed analysis
        if intent.get('detail_level') == 'high':
            return {
                'format': 'PNG',
                'quality': 95,
                'optimize': True
            }
        # Fast processing for quick checks
        elif intent.get('urgency') == 'high' or intent.get('query_type') == 'confirmation':
            return {
                'format': 'JPEG',
                'quality': 70,
                'optimize': True
            }
        # Standard balanced approach
        else:
            return {
                'format': 'JPEG',
                'quality': 85,
                'optimize': True
            }
    
    def _generate_vision_system_prompt(self, user_input: str) -> str:
        """Generate dynamic system prompt for vision queries"""
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        intent = getattr(self, '_last_vision_intent', {})
        
        # Base prompt components
        base = "You are JARVIS, analyzing the user's screen."
        
        # Add specific instructions based on intent
        instructions = []
        
        if intent.get('query_type') == 'confirmation':
            instructions.append("Provide a brief confirmation of what you can see.")
        elif intent.get('detail_level') == 'high':
            instructions.append("Provide comprehensive analysis with full details.")
        elif intent.get('urgency') == 'high':
            instructions.append("Be concise and focus on key information.")
        
        if intent.get('focus_areas'):
            focus = ", ".join(intent['focus_areas'])
            instructions.append(f"Pay special attention to: {focus}")
        
        # Combine components
        prompt_parts = [base]
        if instructions:
            prompt_parts.extend(instructions)
        prompt_parts.append(f"Current date/time: {current_datetime}")
        
        return " ".join(prompt_parts)
    
    def _get_vision_api_config(self, user_input: str) -> Dict[str, Any]:
        """Get dynamic API configuration for vision requests"""
        intent = getattr(self, '_last_vision_intent', {})
        
        # Model selection based on requirements
        if intent.get('urgency') == 'high' and 'haiku' in self.model.lower():
            model = self.model  # Use faster model if available
        else:
            model = self.model
        
        # Token limits based on detail level
        if intent.get('detail_level') == 'high':
            max_tokens = 2048
        elif intent.get('query_type') == 'confirmation':
            max_tokens = 256
        else:
            max_tokens = self.max_tokens
        
        # Temperature based on task type
        if intent.get('query_type') == 'analysis':
            temperature = 0.3  # More focused
        else:
            temperature = self.temperature
        
        return {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature
        }
    
    def _get_timezone_name(self) -> Optional[str]:
        """Get system timezone name"""
        try:
            if self._platform == 'darwin':
                result = subprocess.run(['systemsetup', '-gettimezone'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if "Time Zone:" in output:
                        return output.split("Time Zone:")[1].strip()
            elif os.path.exists('/etc/timezone'):
                with open('/etc/timezone', 'r') as f:
                    return f.read().strip()
        except:
            pass
        return None
            
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
                    logger.info(f"[MONITORING] Vision analyzer available: {self.vision_analyzer is not None}")
                    logger.info(f"[MONITORING] Vision analyzer ID: {id(self.vision_analyzer) if self.vision_analyzer else None}")
                    
                    # Check video streaming module before starting
                    if hasattr(self.vision_analyzer, 'get_video_streaming'):
                        vs = await self.vision_analyzer.get_video_streaming()
                        logger.info(f"[MONITORING] Video streaming module exists: {vs is not None}")
                        if vs:
                            logger.info(f"[MONITORING] Video streaming already capturing: {vs.is_capturing}")
                    
                    # Start video streaming
                    try:
                        logger.info(f"[MONITORING] Vision analyzer type: {type(self.vision_analyzer).__name__}")
                        logger.info(f"[MONITORING] Vision analyzer has start_video_streaming: {hasattr(self.vision_analyzer, 'start_video_streaming')}")
                        logger.info(f"[MONITORING] Vision analyzer config: enable_video_streaming={self.vision_analyzer.config.enable_video_streaming}")
                        result = await self.vision_analyzer.start_video_streaming()
                        logger.info(f"[MONITORING] Video streaming result: {result}")
                        logger.info(f"[MONITORING] Result success: {result.get('success')}")
                        
                        if result.get('success'):
                            # Update monitoring state
                            self._monitoring_active = True
                            capture_method = result.get('metrics', {}).get('capture_method', 'unknown')
                            
                            # IMPORTANT: Return the proper response about video capture activation
                            if capture_method == 'macos_native':
                                return "I have successfully activated native macOS video capturing for monitoring your screen. The purple recording indicator should now be visible in your menu bar, confirming that screen recording is active. I'm capturing at 30 FPS and will continuously monitor for any changes or important events on your screen."
                            elif capture_method == 'swift_native':
                                return "I have successfully activated Swift-based macOS video capturing for monitoring your screen. The purple recording indicator should now be visible in your menu bar, confirming that screen recording is active. I'm capturing at 30 FPS with enhanced permission handling and will continuously monitor for any changes or important events on your screen."
                            elif capture_method == 'direct_swift':
                                return "I have successfully activated direct Swift video capturing for monitoring your screen. The purple recording indicator should now be visible in your menu bar, confirming that screen recording is active. I'm monitoring continuously at 30 FPS and will watch for any changes or important events on your screen until you tell me to stop."
                            elif capture_method == 'purple_indicator':
                                return "I have successfully started monitoring your screen. The purple recording indicator should now be visible in your menu bar, confirming that screen recording is active. I'm capturing your screen at 30 FPS and will continuously monitor for any changes or important events until you tell me to stop."
                            else:
                                return f"I've started monitoring your screen using {capture_method} capture mode at 30 FPS. I'll continuously watch for any changes or important events on your screen."
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            logger.error(f"[MONITORING] Failed to start video streaming: {error_msg}")
                            return f"I encountered an issue starting video streaming: {error_msg}. Please check that screen recording permissions are enabled in System Preferences."
                            
                    except Exception as e:
                        logger.error(f"[MONITORING] Exception starting video streaming: {e}")
                        return f"I encountered an error starting video monitoring: {str(e)}. Please ensure screen recording permissions are enabled."
                        
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
            
            # Update system prompt with current date
            # Use correct year - datetime.now() returns actual current date
            current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
            dynamic_system_prompt = f"""{self.system_prompt}

Note: The current date and time is {current_datetime}. Always use this as the reference for any time-related queries."""
            
            # Make API call
            start_time = datetime.now()
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=dynamic_system_prompt,
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