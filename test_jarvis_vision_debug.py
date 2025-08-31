#!/usr/bin/env python3
"""
Debug test for JARVIS vision command
"""

import asyncio
import json
import sys
import os
sys.path.append('backend')

async def test_vision_message():
    """Test the exact message flow JARVIS would use"""
    print("🔍 Testing JARVIS Vision Message Flow")
    print("=" * 50)
    
    # Import the handler
    from api.unified_vision_handler import handle_vision_command
    
    # Create the exact message JARVIS would send
    message = {
        "type": "vision_command",
        "command": "can you see my screen"
    }
    
    print(f"\n📤 Sending message: {json.dumps(message, indent=2)}")
    
    try:
        # Call the handler with minimal kwargs
        result = await handle_vision_command(message)
        
        print(f"\n📥 Received result:")
        print(json.dumps(result, indent=2, default=str))
        
        # Check if it's an error
        if result.get("type") == "error":
            print(f"\n❌ Error detected: {result.get('message')}")
            if result.get("traceback"):
                print("\nTraceback:")
                print(result["traceback"])
        else:
            print(f"\n✅ Success! Result type: {result.get('type')}")
            
    except Exception as e:
        print(f"\n❌ Exception during test: {e}")
        import traceback
        traceback.print_exc()

async def test_direct_capture():
    """Test direct screen capture"""
    print("\n\n🖼️  Testing direct screen capture...")
    
    try:
        from vision.screen_capture_fallback import capture_with_intelligence
        
        result = capture_with_intelligence("can you see my screen", use_claude=True)
        
        if result.get("success"):
            print("✅ Direct capture successful")
            if result.get("analysis"):
                print(f"Analysis: {result['analysis'][:200]}...")
        else:
            print(f"❌ Direct capture failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Exception in direct capture: {e}")
        import traceback
        traceback.print_exc()

async def test_vision_system_init():
    """Test VisionSystemV2 initialization"""
    print("\n\n🤖 Testing VisionSystemV2 initialization...")
    
    try:
        from vision.vision_system_v2 import VisionSystemV2
        vision = VisionSystemV2()
        print("✅ VisionSystemV2 initialized successfully")
        
        # Test process_command
        result = await vision.process_command("can you see my screen")
        print(f"Command result: {result}")
        
    except Exception as e:
        print(f"❌ VisionSystemV2 initialization failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await test_vision_message()
    await test_direct_capture()
    await test_vision_system_init()

if __name__ == "__main__":
    asyncio.run(main())