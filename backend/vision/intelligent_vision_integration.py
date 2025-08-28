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
        if "can you see my screen" in command_lower or "what's on my screen" in command_lower:
            # If Claude is available, use it directly
            if self.api_key:
                try:
                    # Get Claude's analysis directly
                    claude_result = capture_with_intelligence(
                        query="Please analyze this screenshot and describe what the user appears to be working on. Be specific about the applications open, the content visible, and any relevant details you can see.",
                        use_claude=True
                    )
                    
                    if claude_result.get("intelligence_used") and claude_result.get("analysis"):
                        return f"Yes sir, I can see your screen. {claude_result['analysis']}"
                    elif not claude_result["success"]:
                        return (
                            "I'm unable to see your screen at the moment, sir. "
                            "Please grant me screen recording permission in "
                            "System Preferences → Security & Privacy → Privacy → Screen Recording."
                        )
                    else:
                        return "I can see your screen but encountered an error analyzing it. Please try again."
                except Exception as e:
                    return f"I encountered an error accessing your screen: {str(e)}"
            else:
                # No Claude API key
                return (
                    "I need an Anthropic API key to analyze your screen. "
                    "Please configure ANTHROPIC_API_KEY in your environment."
                )
            
        # For other vision commands, use enhanced processing
        elif any(phrase in command_lower for phrase in [
            "what's on my screen", "analyze my screen", "what do you see",
            "check for updates", "look for updates",
            "what am i working", "what i'm working", "working on",
            "what are you seeing", "describe what you see",
            "tell me what", "show me what"
        ]):
            # Use the intelligent capture if Claude is available
            if self.api_key:
                # Create a more specific query based on the user's intent
                if "working" in command_lower:
                    query = "Analyze what the user is currently working on based on the open applications, visible windows, and content. Be specific about the applications, files, and tasks visible."
                elif "update" in command_lower:
                    query = "Check for any software update notifications, badges, or update-related windows on the screen."
                else:
                    query = f"Analyze the screen and respond to this request: {command}"
                
                result = capture_with_intelligence(
                    query=query,
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