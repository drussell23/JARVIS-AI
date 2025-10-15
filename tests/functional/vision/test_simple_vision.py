#!/usr/bin/env python3
"""
Simple vision test without WebSocket complexity
"""

import asyncio
import sys
import os
sys.path.append('backend')

async def test_vision():
    print("🔍 Simple Vision Test")
    print("=" * 50)
    
    # Test 1: Basic screen capture
    print("\n1. Testing basic screen capture...")
    try:
        from vision.screen_capture_fallback import capture_screen_fallback
        screenshot = capture_screen_fallback()
        if screenshot:
            print(f"✅ Screen captured: {screenshot.size}")
        else:
            print("❌ Failed to capture screen")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Vision System V2
    print("\n2. Testing Vision System V2...")
    try:
        from vision.vision_system_v2 import VisionSystemV2
        vision = VisionSystemV2()
        
        # Simple command
        response = await vision.process_command("What's on my screen?")
        print(f"✅ Vision response: {response.success}")
        print(f"   Message: {response.message[:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Dynamic Vision Engine
    print("\n3. Testing Dynamic Vision Engine...")
    try:
        from vision.dynamic_vision_engine import get_dynamic_vision_engine
        engine = get_dynamic_vision_engine()
        
        # Check if process_vision_command exists
        if hasattr(engine, 'process_vision_command'):
            result = await engine.process_vision_command("Can you see the screen?")
            print(f"✅ Engine has process_vision_command method")
            print(f"   Result: {result}")
        else:
            print("❌ Engine missing process_vision_command method")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✨ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_vision())