#!/usr/bin/env python3
"""
Test complete vision integration - from command to execution
"""

import asyncio
import logging
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_vision_flow():
    """Test the complete vision flow from command interpreter to execution"""
    print("🧪 Testing Complete Vision Flow")
    print("=" * 50)
    
    # Test 1: Command Interpreter
    print("\n1️⃣ Testing Command Interpreter...")
    try:
        from system_control.claude_command_interpreter import ClaudeCommandInterpreter
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  No Claude API key - using mock interpreter")
            # Create mock intent
            class MockIntent:
                action = "describe_screen"
                category = type('obj', (object,), {'value': 'VISION'})()
                confidence = 0.9
                parameters = {}
            
            intent = MockIntent()
        else:
            interpreter = ClaudeCommandInterpreter(api_key)
            intent = await interpreter.interpret_command("describe what's on my screen")
            
        print(f"✓ Intent action: {intent.action}")
        print(f"✓ Intent category: {intent.category.value}")
        print(f"✓ Intent confidence: {intent.confidence}")
        
        # Test 2: Execute the intent
        print("\n2️⃣ Testing Intent Execution...")
        if api_key:
            result = await interpreter.execute_intent(intent)
            print(f"✓ Execution success: {result.success}")
            print(f"✓ Result message: {result.message[:200]}..." if result.message else "❌ No message")
        
    except Exception as e:
        print(f"❌ Command Interpreter Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Direct Vision Handler Test
    print("\n\n3️⃣ Testing Direct Vision Handler Execution...")
    try:
        from system_control.vision_action_handler import get_vision_action_handler
        handler = get_vision_action_handler()
        
        # Test with actual screen capture
        result = await handler.describe_screen()
        print(f"✓ Handler success: {result.success}")
        print(f"✓ Description length: {len(result.description)} chars")
        
        # Check if it's actually processing or just returning generic response
        if "I'm not quite sure what you'd like me to do" in result.description:
            print("❌ WARNING: Getting generic fallback response, not actual vision processing!")
            
            # Try to force actual processing
            print("\n4️⃣ Forcing actual vision processing...")
            
            # Try direct unified vision system
            from vision.unified_vision_system import get_unified_vision_system
            unified = get_unified_vision_system()
            
            # Check available components
            print(f"✓ Available components: {list(unified.components.keys())}")
            
            # Force a specific route
            from vision.unified_vision_system import VisionRequest
            req = VisionRequest(command="describe my screen")
            result = await unified._execute_with_component("dynamic_engine", req)
            print(f"✓ Dynamic engine result: {result.success}")
            if result.description and "I'm not quite sure" not in result.description:
                print(f"✓ Got actual vision response: {result.description[:200]}...")
            else:
                print("❌ Still getting generic response")
                
                # Try plugin system directly
                print("\n5️⃣ Testing Plugin System directly...")
                from vision.vision_plugin_system import get_vision_plugin_system
                plugin_system = get_vision_plugin_system()
                
                # Check capabilities
                all_caps = plugin_system.list_capabilities()
                print(f"✓ Available capabilities: {len(all_caps)}")
                for cap, providers in list(all_caps.items())[:3]:
                    print(f"  • {cap}: {providers}")
                    
        else:
            print(f"✓ Got actual vision response: {result.description[:200]}...")
            
    except Exception as e:
        print(f"❌ Vision Handler Error: {e}")
        import traceback
        traceback.print_exc()
        
    # Test 4: Check Screen Capture
    print("\n\n6️⃣ Testing Screen Capture...")
    try:
        from vision.screen_capture_fallback import capture_screen_fallback
        screenshot = capture_screen_fallback()
        
        if screenshot:
            print(f"✓ Screen captured: {screenshot.size} pixels")
        else:
            print("❌ Failed to capture screen")
            
    except Exception as e:
        print(f"❌ Screen Capture Error: {e}")


async def test_claude_vision_direct():
    """Test Claude vision analyzer directly"""
    print("\n\n7️⃣ Testing Claude Vision Analyzer Directly...")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  No API key - skipping Claude test")
        return
        
    try:
        from vision.claude_vision_analyzer import ClaudeVisionAnalyzer
        from vision.screen_capture_fallback import capture_screen_fallback
        
        analyzer = ClaudeVisionAnalyzer(api_key)
        screenshot = capture_screen_fallback()
        
        if screenshot:
            print("✓ Screen captured for Claude")
            # Don't actually call API to save costs
            print("✓ Claude analyzer ready (skipping API call to save costs)")
        else:
            print("❌ No screenshot for Claude")
            
    except Exception as e:
        print(f"❌ Claude Direct Test Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_vision_flow())
    asyncio.run(test_claude_vision_direct())