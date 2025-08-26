#!/usr/bin/env python3
"""
Test the vision response fix for "can you see my screen?"
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice.intelligent_command_handler import IntelligentCommandHandler

async def test_vision_responses():
    print("🧪 Testing Vision Response Fix")
    print("=" * 50)
    
    # Initialize handler
    handler = IntelligentCommandHandler(user_name="Sir")
    
    if not handler.enabled:
        print("❌ Handler not enabled - check ANTHROPIC_API_KEY")
        return
    
    # Test commands
    test_commands = [
        "can you see my screen?",
        "are you able to see my screen?",
        "do you see what's on my display?",
        "what's on my screen?",
        "describe my screen",
        "analyze what I'm looking at"
    ]
    
    for command in test_commands:
        print(f"\n📝 Command: '{command}'")
        print("-" * 50)
        
        try:
            response, handler_type = await handler.handle_command(command)
            
            print(f"🏷️ Handler: {handler_type}")
            print(f"\n🤖 Response:")
            print(response[:500] + "..." if len(response) > 500 else response)
            
            # Check if we avoided the generic response
            if "Command executed successfully" in response:
                print("\n⚠️ WARNING: Generic response detected!")
            else:
                print("\n✅ Proper vision response!")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_vision_responses())