#!/usr/bin/env python3
"""
Enhanced Vision System - Combining Local Capture with Claude Intelligence
This revolutionary approach uses Claude's vision capabilities to provide
superhuman screen understanding while respecting macOS security.
"""

import base64
import json
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import io
import os

# Import existing capture methods
from .screen_vision import ScreenVisionSystem
from .screen_capture_fallback import capture_screen_fallback

class EnhancedVisionSystem:
    """
    Revolutionary vision system that combines local screen capture
    with Claude's advanced vision understanding.
    """
    
    def __init__(self, anthropic_api_key: str):
        """Initialize enhanced vision with Claude integration."""
        self.base_vision = ScreenVisionSystem()
        self.api_key = anthropic_api_key
        self._init_claude_client()
        
        # Intelligent caching to minimize API calls
        self.cache = {}
        self.cache_duration = timedelta(seconds=30)
        
        # Permission status tracking
        self.permission_granted = None
        self.last_permission_check = None
        
    def _init_claude_client(self):
        """Initialize Claude client for vision analysis."""
        try:
            from anthropic import Anthropic
            self.claude = Anthropic(api_key=self.api_key)
            self.vision_available = True
        except Exception as e:
            print(f"Claude Vision not available: {e}")
            self.vision_available = False
            
    async def capture_and_understand(self, query: str = None, 
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        The revolutionary method: Capture once, understand deeply.
        
        Args:
            query: Natural language query about the screen
            context: Additional context for better understanding
            
        Returns:
            Dict containing understanding, suggestions, and actions
        """
        # Step 1: Smart permission checking
        if not self._check_permission():
            return self._handle_no_permission()
            
        # Step 2: Intelligent capture (with caching)
        screenshot = await self._smart_capture(query)
        if not screenshot:
            return {"error": "Failed to capture screen", "suggestion": "Check permissions"}
            
        # Step 3: Claude-powered analysis
        if self.vision_available and query:
            analysis = await self._analyze_with_claude(screenshot, query, context)
            return self._process_claude_response(analysis)
        else:
            # Fallback to basic OCR if Claude unavailable
            return self._basic_analysis(screenshot)
            
    async def _smart_capture(self, query: str = None) -> Optional[np.ndarray]:
        """
        Smart capture that minimizes resource usage.
        Uses caching and selective capture based on query.
        """
        # Check cache first
        cache_key = f"screen_{query if query else 'full'}"
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data['time'] < self.cache_duration:
                return cached_data['image']
                
        # Capture using best available method
        screenshot = None
        
        # Try primary method
        screenshot = self.base_vision.capture_screen()
        
        # Try fallback if needed
        if screenshot is None:
            screenshot = capture_screen_fallback()
            
        # Cache the result
        if screenshot is not None:
            self.cache[cache_key] = {
                'image': screenshot,
                'time': datetime.now()
            }
            
        return screenshot
        
    async def _analyze_with_claude(self, image: np.ndarray, 
                                  query: str, context: Dict = None) -> Dict[str, Any]:
        """
        The magic happens here: Claude's superhuman vision understanding.
        """
        # Convert numpy array to base64
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Build intelligent prompt
        prompt = self._build_intelligent_prompt(query, context)
        
        try:
            response = self.claude.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            return {
                "success": True,
                "analysis": response.content[0].text,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback": self._basic_analysis(image)
            }
            
    def _build_intelligent_prompt(self, query: str, context: Dict = None) -> str:
        """
        Build context-aware prompts for Claude that maximize understanding.
        """
        base_prompt = f"""You are JARVIS, analyzing the user's screen. 
        User query: {query}
        
        Provide:
        1. Direct answer to the query
        2. Important observations about what you see
        3. Suggested actions the user might want to take
        4. Any warnings or issues detected
        
        Be concise but thorough. Focus on being helpful."""
        
        if context:
            base_prompt += f"\n\nAdditional context: {json.dumps(context)}"
            
        return base_prompt
        
    def _check_permission(self) -> bool:
        """
        Smart permission checking with caching.
        """
        # Cache permission status for 5 minutes
        if self.last_permission_check:
            if datetime.now() - self.last_permission_check < timedelta(minutes=5):
                return self.permission_granted
                
        # Test actual capture capability (synchronous version)
        try:
            import Quartz
            screenshot = Quartz.CGDisplayCreateImage(Quartz.CGMainDisplayID())
            self.permission_granted = screenshot is not None
        except:
            self.permission_granted = False
            
        self.last_permission_check = datetime.now()
        
        return self.permission_granted
        
    def _handle_no_permission(self) -> Dict[str, Any]:
        """
        Intelligent handling when permissions aren't granted.
        """
        return {
            "error": "Screen recording permission required",
            "instructions": self._get_permission_instructions(),
            "alternative_actions": [
                "You can describe what you see and I'll help",
                "Take a screenshot manually and I can analyze it",
                "Grant permission for the full JARVIS experience"
            ]
        }
        
    def _get_permission_instructions(self) -> List[str]:
        """
        Get step-by-step permission instructions.
        """
        return [
            "Open System Preferences → Security & Privacy → Privacy → Screen Recording",
            "Click the lock to make changes",
            "Find Terminal (or your current app) in the list",
            "Check the checkbox next to it",
            "Restart Terminal/IDE for changes to take effect",
            "Run 'python fix_screen_permission.py' if you need help"
        ]
        
    def _process_claude_response(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Claude's response into actionable intelligence.
        """
        if not analysis.get("success"):
            return analysis.get("fallback", {"error": "Analysis failed"})
            
        return {
            "understanding": analysis["analysis"],
            "timestamp": analysis["timestamp"],
            "cached": False,
            "intelligence_level": "advanced"
        }
        
    def _basic_analysis(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Fallback to basic OCR when Claude isn't available.
        """
        text = self.base_vision.extract_text_from_screen(screenshot)
        return {
            "understanding": f"Found {len(text)} text elements on screen",
            "text_elements": text[:10],  # First 10 elements
            "intelligence_level": "basic",
            "suggestion": "Enable Claude Vision for advanced understanding"
        }

class IntelligentVisionCommands:
    """
    Revolutionary command system that leverages Claude's understanding.
    """
    
    def __init__(self, vision_system: EnhancedVisionSystem):
        self.vision = vision_system
        
    async def process_command(self, command: str) -> str:
        """
        Process natural language vision commands with intelligence.
        """
        # Map commands to intelligent queries
        command_mapping = {
            "what's on my screen": self._analyze_screen,
            "find errors": self._find_errors,
            "check for updates": self._check_updates,
            "help me with this": self._contextual_help,
            "what should i do next": self._suggest_next_action,
            "summarize my work": self._summarize_activity,
            "find the button": self._find_ui_element,
            "read this for me": self._read_content,
            "is everything ok": self._health_check,
            "monitor this": self._start_monitoring
        }
        
        # Find best matching command
        for key, handler in command_mapping.items():
            if key in command.lower():
                return await handler(command)
                
        # Default: send directly to Claude
        return await self._direct_query(command)
        
    async def _analyze_screen(self, command: str) -> str:
        """Comprehensive screen analysis."""
        result = await self.vision.capture_and_understand(
            "Provide a comprehensive analysis of what you see on the screen. " +
            "Include open applications, current activity, and any notable items."
        )
        return self._format_response(result)
        
    async def _find_errors(self, command: str) -> str:
        """Intelligent error detection."""
        result = await self.vision.capture_and_understand(
            "Look for any error messages, warnings, alerts, or problems on the screen. " +
            "Include red text, error dialogs, warning icons, or anything that indicates an issue."
        )
        return self._format_response(result)
        
    async def _check_updates(self, command: str) -> str:
        """Smart update detection."""
        result = await self.vision.capture_and_understand(
            "Check for any software update notifications, badges, or indicators. " +
            "Look in menu bars, dock icons, browser tabs, and application windows."
        )
        return self._format_response(result)
        
    async def _contextual_help(self, command: str) -> str:
        """Provide contextual assistance."""
        result = await self.vision.capture_and_understand(
            "Analyze what the user is trying to do and provide helpful guidance. " +
            "Suggest next steps, point out important UI elements, or offer tips."
        )
        return self._format_response(result)
        
    async def _suggest_next_action(self, command: str) -> str:
        """Predictive assistance."""
        result = await self.vision.capture_and_understand(
            "Based on what you see, suggest the most logical next action the user should take. " +
            "Consider the current context, open applications, and visible UI state."
        )
        return self._format_response(result)
        
    async def _find_ui_element(self, command: str) -> str:
        """Find specific UI elements."""
        # Extract what they're looking for
        element = command.lower().replace("find the", "").replace("find", "").strip()
        result = await self.vision.capture_and_understand(
            f"Help the user find '{element}' on the screen. " +
            "Describe its location clearly (e.g., 'top right corner', 'in the sidebar')."
        )
        return self._format_response(result)
        
    async def _direct_query(self, command: str) -> str:
        """Direct natural language query to Claude."""
        result = await self.vision.capture_and_understand(command)
        return self._format_response(result)
        
    async def _summarize_activity(self, command: str) -> str:
        """Summarize user activity."""
        result = await self.vision.capture_and_understand(
            "Summarize what the user is currently working on based on the visible applications, " +
            "windows, and content. Be specific about the task at hand."
        )
        return self._format_response(result)
        
    async def _health_check(self, command: str) -> str:
        """Check overall system health."""
        result = await self.vision.capture_and_understand(
            "Check if everything looks normal on the screen. Look for any issues, " +
            "errors, warnings, or things that need attention."
        )
        return self._format_response(result)
        
    async def _read_content(self, command: str) -> str:
        """Read specific content from the screen."""
        result = await self.vision.capture_and_understand(
            "Read and summarize the main content visible on the screen. " +
            "Focus on the most important text and information."
        )
        return self._format_response(result)
        
    async def _start_monitoring(self, command: str) -> str:
        """Start monitoring for changes."""
        # This would integrate with a monitoring system
        return "Sir, I'll start monitoring your screen for important changes. I'll alert you when something needs your attention."
        
    def _format_response(self, result: Dict[str, Any]) -> str:
        """Format responses in JARVIS style."""
        if result.get("error"):
            return f"Sir, I encountered an issue: {result['error']}"
            
        understanding = result.get("understanding", "")
        
        if result.get("intelligence_level") == "basic":
            return f"I can see the screen but with limited understanding. {understanding}"
        else:
            return f"{understanding}"

# Example usage and integration points
async def main():
    """Example of the enhanced vision system in action."""
    
    # Initialize with API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    vision = EnhancedVisionSystem(api_key)
    commands = IntelligentVisionCommands(vision)
    
    # Example queries that showcase the power
    example_queries = [
        "What errors do you see on my screen?",
        "Help me fill out this form",
        "What should I click next?",
        "Is there anything important I'm missing?",
        "Summarize what I'm working on",
        "Find the submit button",
        "Check if any apps need updates",
        "What's that red notification?",
        "Help me debug this code",
        "What does this error mean?"
    ]
    
    print("Enhanced Vision System - Revolutionary Examples:")
    print("=" * 50)
    
    for query in example_queries[:3]:  # Demo first 3
        print(f"\nQuery: {query}")
        response = await commands.process_command(query)
        print(f"JARVIS: {response}")
        

if __name__ == "__main__":
    asyncio.run(main())