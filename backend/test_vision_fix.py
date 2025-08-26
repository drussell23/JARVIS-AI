#!/usr/bin/env python3
"""
Test script to verify vision system fix
"""

import asyncio
import logging
import os
from system_control.claude_command_interpreter import ClaudeCommandInterpreter
from system_control.vision_action_handler import get_vision_action_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_vision_commands():
    """Test vision commands through the interpreter"""
    print("🔧 Testing Vision System Fix")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return
        
    # Initialize interpreter
    interpreter = ClaudeCommandInterpreter(api_key)
    
    # Test commands
    test_commands = [
        "describe what's on my screen",
        "what am I looking at?",
        "check my screen for errors",
        "analyze the current window",
        "tell me what you see"
    ]
    
    for command in test_commands:
        print(f"\n🎯 Testing: '{command}'")
        print("-" * 40)
        
        try:
            # Interpret command
            intent = await interpreter.interpret_command(command)
            print(f"✓ Intent: {intent.action}")
            print(f"✓ Category: {intent.category.value}")
            print(f"✓ Confidence: {intent.confidence}")
            
            # Execute command
            result = await interpreter.execute_intent(intent)
            print(f"✓ Success: {result.success}")
            print(f"✓ Response: {result.message}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            
    # Test direct vision handler
    print("\n\n🔧 Testing Direct Vision Handler")
    print("=" * 50)
    
    vision_handler = get_vision_action_handler()
    
    # Test describe_screen
    print("\n📷 Testing describe_screen...")
    result = await vision_handler.describe_screen()
    print(f"✓ Success: {result.success}")
    print(f"✓ Description: {result.description[:200]}...")
    
    # Test check_screen
    print("\n🔍 Testing check_screen for notifications...")
    result = await vision_handler.check_screen({"target": "notifications"})
    print(f"✓ Success: {result.success}")
    print(f"✓ Description: {result.description}")
    
    print("\n✨ Vision system test complete!")


if __name__ == "__main__":
    asyncio.run(test_vision_commands())