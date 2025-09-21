#!/usr/bin/env python3
"""
Example of how JARVIS should use vision to understand and interact with the screen
This demonstrates real-world usage patterns
"""

import asyncio
import os
import logging
from datetime import datetime
import subprocess
import tempfile
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JARVISVisionAssistant:
    """Example JARVIS assistant with vision capabilities"""
    
    def __init__(self):
        self.vision_analyzer = None
        self._init_vision()
    
    def _init_vision(self):
        """Initialize vision analyzer"""
        try:
            from claude_vision_analyzer import ClaudeVisionAnalyzer
            api_key = os.getenv('ANTHROPIC_API_KEY')
            
            if api_key:
                self.vision_analyzer = ClaudeVisionAnalyzer(api_key)
                logger.info("✅ Vision system initialized")
            else:
                logger.warning("⚠️ No API key - vision disabled")
        except Exception as e:
            logger.error(f"❌ Vision init failed: {e}")
    
    async def capture_screen(self):
        """Capture current screen"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            subprocess.run(['screencapture', '-x', tmp_path], check=True)
            image = Image.open(tmp_path)
            screenshot = np.array(image)
            return screenshot
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def understand_context(self):
        """Understand current screen context"""
        if not self.vision_analyzer:
            return "Vision not available"
        
        try:
            context = await self.vision_analyzer.get_screen_context()
            return context.get('description', 'Unable to analyze screen')
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return f"Error: {e}"
    
    async def handle_command(self, command: str):
        """Handle user commands with visual context"""
        logger.info(f"\n🎤 User: {command}")
        
        if not self.vision_analyzer:
            return "I can't see the screen right now. Please check my vision settings."
        
        command_lower = command.lower()
        
        # Get current context
        screenshot = await self.capture_screen()
        
        # Different command patterns
        if "what" in command_lower and ("see" in command_lower or "screen" in command_lower):
            # Describe what's on screen
            result = await self.vision_analyzer.analyze_screenshot(
                screenshot,
                "Describe what's currently visible on the screen in detail."
            )
            return f"I can see: {result['description']}"
        
        elif "read" in command_lower:
            # Extract text
            result = await self.vision_analyzer.analyze_screenshot(
                screenshot,
                "Extract and list all readable text from the screen."
            )
            return f"Here's what I can read: {result['description']}"
        
        elif "click" in command_lower:
            # Find clickable elements
            result = await self.vision_analyzer.analyze_screenshot(
                screenshot,
                f"The user wants to: {command}. What UI elements are clickable? List them with descriptions."
            )
            return f"I found these clickable elements: {result['description']}"
        
        elif "help" in command_lower:
            # Provide contextual help
            result = await self.vision_analyzer.analyze_screenshot(
                screenshot,
                "What actions can the user take in this application? Provide helpful suggestions."
            )
            return f"Here's what you can do: {result['description']}"
        
        elif "find" in command_lower or "where" in command_lower:
            # Locate specific elements
            result = await self.vision_analyzer.analyze_screenshot(
                screenshot,
                f"Help the user with this request: {command}. Locate and describe relevant UI elements."
            )
            return f"I found: {result['description']}"
        
        else:
            # General context-aware response
            result = await self.vision_analyzer.analyze_screenshot(
                screenshot,
                f"The user said: '{command}'. Based on what's on screen, how should I help them?"
            )
            return result['description']
    
    async def demonstrate_capabilities(self):
        """Show what JARVIS can do with vision"""
        print("\n" + "="*60)
        print("🤖 JARVIS Vision Capabilities Demo")
        print("="*60)
        
        # Test various commands
        test_commands = [
            "What do you see on the screen?",
            "Can you read the text in the window?",
            "Where can I click?",
            "Help me with this application",
            "Find the menu options"
        ]
        
        for cmd in test_commands:
            response = await self.handle_command(cmd)
            print(f"\n👤 User: {cmd}")
            print(f"🤖 JARVIS: {response[:200]}...")
            await asyncio.sleep(1)  # Pause between commands
        
        # Cleanup
        if self.vision_analyzer:
            await self.vision_analyzer.cleanup_all_components()

async def main():
    """Run the demonstration"""
    jarvis = JARVISVisionAssistant()
    
    # Check if vision is available
    if not jarvis.vision_analyzer:
        print("❌ Vision not available. Set ANTHROPIC_API_KEY to enable.")
        return
    
    print("✅ JARVIS Vision System Ready!")
    print("\nTesting basic context understanding...")
    
    # Test context understanding
    context = await jarvis.understand_context()
    print(f"\n📊 Current Context: {context[:200]}...")
    
    # Run full demo
    await jarvis.demonstrate_capabilities()
    
    print("\n✅ Demo complete! JARVIS can now see and understand your screen.")

if __name__ == "__main__":
    asyncio.run(main())