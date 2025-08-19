#!/usr/bin/env python3
"""
Intelligent Vision Integration for JARVIS
Combines screen capture permissions with Claude's intelligence
"""

import os
import asyncio
from typing import Optional
from .screen_vision import ScreenVisionSystem, JARVISVisionIntegration
from .screen_capture_fallback import capture_with_intelligence
from .claude_vision_analyzer import ClaudeVisionAnalyzer


class IntelligentJARVISVision:
    """Enhanced JARVIS vision that combines permissions with intelligence"""
    
    def __init__(self):
        self.vision_system = ScreenVisionSystem()
        self.jarvis_vision = JARVISVisionIntegration(self.vision_system)
        
        # Check for Claude API
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.claude_analyzer = None
        if self.api_key:
            self.claude_analyzer = ClaudeVisionAnalyzer(self.api_key)
            
    async def handle_intelligent_command(self, command: str) -> str:
        """
        Handle vision commands with enhanced intelligence
        
        This method provides the intelligent responses you're looking for,
        confirming permission status and adding Claude insights when available.
        """
        command_lower = command.lower()
        
        # Special handling for "can you see my screen?"
        if "can you see my screen" in command_lower:
            # Test screen capture permission
            result = capture_with_intelligence(use_claude=False)
            
            if not result["success"]:
                return (
                    "I'm unable to see your screen at the moment, sir. "
                    "Please grant me screen recording permission in "
                    "System Preferences → Security & Privacy → Privacy → Screen Recording. "
                    "Once granted, I'll be able to help you with visual tasks."
                )
            
            # Permission is granted! Build intelligent response
            response = "Yes sir, I can see your screen perfectly. "
            
            # Get screen details
            screenshot = result["image"]
            width, height = screenshot.shape[1], screenshot.shape[0]
            response += f"I'm viewing your {width}x{height} display. "
            
            # If Claude is available, add intelligent analysis
            if self.api_key and self.claude_analyzer:
                try:
                    # Get Claude's analysis
                    claude_result = capture_with_intelligence(
                        query="Briefly describe what applications are open and what the user appears to be working on",
                        use_claude=True
                    )
                    
                    if claude_result.get("intelligence_used") and claude_result.get("analysis"):
                        response += claude_result["analysis"]
                    else:
                        response += "I can provide basic screen analysis. "
                except:
                    response += "I can provide basic screen analysis. "
            else:
                # No Claude API - provide basic info
                basic_analysis = await self.jarvis_vision.vision.capture_and_describe()
                # Extract the relevant part after "Yes sir, I can see your screen."
                if "You have" in basic_analysis:
                    apps_info = basic_analysis[basic_analysis.find("You have"):]
                    response += apps_info
                else:
                    response += (
                        "I can capture your screen and perform basic text extraction. "
                        "To unlock my full visual intelligence capabilities, "
                        "consider adding an Anthropic API key for Claude Vision."
                    )
            
            return response
            
        # For other vision commands, use enhanced processing
        elif any(phrase in command_lower for phrase in [
            "what's on my screen", "analyze my screen", "what do you see",
            "check for updates", "look for updates"
        ]):
            # Use the intelligent capture if Claude is available
            if self.api_key:
                result = capture_with_intelligence(
                    query=command,
                    use_claude=True
                )
                
                if result.get("intelligence_used") and result.get("analysis"):
                    return f"Sir, {result['analysis']}"
                    
            # Fallback to standard vision handling
            return await self.jarvis_vision.handle_vision_command(command)
            
        # Default to standard vision handling
        return await self.jarvis_vision.handle_vision_command(command)


async def test_intelligent_vision():
    """Test the intelligent vision responses"""
    
    print("🧠 Testing Intelligent JARVIS Vision")
    print("=" * 50)
    
    vision = IntelligentJARVISVision()
    
    # Test the key command
    test_commands = [
        "Hey JARVIS, can you see my screen?",
        "What's on my screen?",
        "Check for software updates",
        "Analyze what I'm working on"
    ]
    
    for command in test_commands:
        print(f"\n🎤 You: {command}")
        response = await vision.handle_intelligent_command(command)
        print(f"🤖 JARVIS: {response}")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(test_intelligent_vision())