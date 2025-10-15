#!/usr/bin/env python3
"""
Test JARVIS vision system integration
Simulates how JARVIS would handle vision commands
"""

import asyncio
import json
import sys
import os
sys.path.append('backend')

from api.unified_vision_handler import UnifiedVisionHandler
from vision.vision_system_v2 import VisionSystemV2

async def test_vision_handler():
    """Test the unified vision handler as JARVIS would use it"""
    print("🤖 Testing JARVIS Vision Integration")
    print("=" * 50)
    
    # Initialize handler
    handler = UnifiedVisionHandler()
    
    # Create a mock context
    from api.unified_vision_handler import VisionContext
    context = VisionContext(
        client_id="test-client",
        capabilities=["vision", "claude"],
        metadata={}
    )
    
    # Test 1: Basic vision command
    print("\n1️⃣ Testing basic vision command...")
    message = {
        "type": "vision_command",
        "command": "What can you see on my screen?"
    }
    
    try:
        result = await handler.handle_vision_command(message, context)
        if result.get("type") == "error":
            print(f"❌ Error: {result.get('message')}")
        else:
            print(f"✅ Success: {result.get('type', 'vision_result')}")
            if result.get("result"):
                print(f"   Response: {str(result['result'])[:100]}...")
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Specific vision query
    print("\n2️⃣ Testing specific vision query...")
    message = {
        "type": "vision_command", 
        "command": "What applications are currently open?"
    }
    
    try:
        result = await handler.handle_vision_command(message, context)
        if result.get("type") == "error":
            print(f"❌ Error: {result.get('message')}")
        else:
            print(f"✅ Success: Vision command processed")
    except Exception as e:
        print(f"❌ Exception: {e}")

async def test_direct_vision_system():
    """Test Vision System V2 directly"""
    print("\n\n3️⃣ Testing Vision System V2 directly...")
    
    vision = VisionSystemV2()
    
    # Test process_command
    try:
        response = await vision.process_command("Can you see my screen?")
        print(f"✅ Direct test success: {response.success}")
        print(f"   Message: {response.message[:100]}...")
        print(f"   Intent: {response.intent_type}")
        print(f"   Confidence: {response.confidence}")
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_websocket_simulation():
    """Simulate WebSocket message handling"""
    print("\n\n4️⃣ Simulating WebSocket message flow...")
    
    # This simulates what happens when JARVIS receives a vision command
    from api.unified_vision_handler import handle_websocket_message
    
    ws_message = {
        "type": "vision_command",
        "command": "What do you see on the screen?",
        "id": "test-123"
    }
    
    try:
        # Simulate WebSocket context
        result = await handle_websocket_message(ws_message)
        print(f"✅ WebSocket simulation successful")
        print(f"   Result type: {result.get('type', 'unknown')}")
        if result.get("error"):
            print(f"   ⚠️  Error in result: {result['error']}")
    except Exception as e:
        print(f"❌ WebSocket simulation failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests"""
    await test_vision_handler()
    await test_direct_vision_system()
    await test_websocket_simulation()
    
    print("\n\n✨ Vision integration test complete!")
    print("\nIf all tests passed, JARVIS should be able to:")
    print("  • See your screen")
    print("  • Analyze what's visible")
    print("  • Answer questions about what it sees")
    print("  • Use Claude Vision for enhanced understanding")

if __name__ == "__main__":
    asyncio.run(main())