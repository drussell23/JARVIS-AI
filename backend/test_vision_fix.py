#!/usr/bin/env python3
"""
Test the vision command fix
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_control.claude_command_interpreter import ClaudeCommandInterpreter

async def test_vision_command():
    print("🧪 Testing Vision Command Fix")
    print("=" * 50)
    
    # Initialize interpreter
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set!")
        return
    
    interpreter = ClaudeCommandInterpreter(api_key)
    
    # Test various vision commands
    test_commands = [
        "can you see my screen",
        "what's on my screen",
        "describe what you see",
        "analyze my screen"
    ]
    
    for command in test_commands:
        print(f"\n📝 Testing: '{command}'")
        print("-" * 50)
        
        try:
            # Interpret the command
            intent = await interpreter.interpret_command(command)
            
            print(f"📊 Interpreted Intent:")
            print(f"  • Action: {intent.action}")
            print(f"  • Category: {intent.category}")
            print(f"  • Target: {intent.target}")
            print(f"  • Confidence: {intent.confidence}")
            
            # Execute the command
            result = await interpreter.execute_intent(intent)
            
            print(f"\n✅ Success: {result.success}")
            print(f"\n🤖 Response:")
            print("-" * 50)
            print(result.message[:500] + "..." if len(result.message) > 500 else result.message)
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vision_command())