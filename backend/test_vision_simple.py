#!/usr/bin/env python3
"""
Simple test for vision system
"""

import asyncio
import logging
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)


async def test_vision():
    """Simple vision test"""
    print("🧪 Testing Vision System")
    print("=" * 50)
    
    # Test 1: Direct vision handler test
    print("\n1️⃣ Testing Vision Handler...")
    try:
        from system_control.vision_action_handler import get_vision_action_handler
        
        handler = get_vision_action_handler()
        
        # Test describe_screen
        result = await handler.describe_screen()
        print(f"✓ describe_screen success: {result.success}")
        if result.success:
            print(f"✓ Response: {result.description[:200]}...")
        else:
            print(f"❌ Error: {result.error}")
            
    except Exception as e:
        print(f"❌ Vision Handler Error: {e}")
        import traceback
        traceback.print_exc()
        
    # Test 2: Command interpreter integration
    print("\n\n2️⃣ Testing Command Interpreter Integration...")
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  ANTHROPIC_API_KEY not set")
        else:
            from system_control.claude_command_interpreter import ClaudeCommandInterpreter
            
            interpreter = ClaudeCommandInterpreter(api_key)
            
            # Test vision command
            intent = await interpreter.interpret_command("describe what's on my screen")
            print(f"✓ Intent action: {intent.action}")
            print(f"✓ Intent category: {intent.category.value}")
            
            if intent.action == "describe_screen":
                result = await interpreter.execute_intent(intent)
                print(f"✓ Execution success: {result.success}")
                if result.success:
                    print(f"✓ Result: {result.message[:200]}...")
                    
    except Exception as e:
        print(f"❌ Command Interpreter Error: {e}")
        
    print("\n✅ Vision test complete!")


if __name__ == "__main__":
    asyncio.run(test_vision())