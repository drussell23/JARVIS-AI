#!/usr/bin/env python3
"""
Complete test of vision functionality - ensures Claude Vision API is working correctly
"""

import asyncio
import sys
import os
from pathlib import Path
import time

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_vision_complete():
    """Test all vision functionality"""
    
    print("🔍 Complete Vision System Test")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found!")
        return
    print("✅ API key found")
    
    # Test 1: Direct screen vision
    print("\n1️⃣ Testing direct screen vision...")
    try:
        from vision.screen_vision import ScreenVisionSystem
        vision_system = ScreenVisionSystem()
        
        start_time = time.time()
        result = await vision_system.capture_and_describe()
        elapsed = time.time() - start_time
        
        print(f"✅ Screen captured and analyzed in {elapsed:.2f}s")
        print(f"Result preview: {result[:200]}...")
        
        # Check for unwanted generic responses
        if "I can read 45 text elements" in result:
            print("❌ WARNING: Getting generic response instead of Claude analysis!")
        elif "Yes sir, I can see your screen" in result:
            print("✅ Claude Vision is working correctly!")
        
    except Exception as e:
        print(f"❌ Direct vision error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Vision action handler
    print("\n2️⃣ Testing vision action handler...")
    try:
        from system_control.vision_action_handler import get_vision_action_handler
        handler = get_vision_action_handler()
        
        start_time = time.time()
        result = await handler.describe_screen()
        elapsed = time.time() - start_time
        
        print(f"✅ Vision action completed in {elapsed:.2f}s")
        print(f"Success: {result.success}")
        print(f"Description preview: {result.description[:200]}...")
        
    except Exception as e:
        print(f"❌ Action handler error: {e}")
    
    # Test 3: Full command flow simulation
    print("\n3️⃣ Testing full command flow...")
    try:
        from voice.intelligent_command_handler import IntelligentCommandHandler
        command_handler = IntelligentCommandHandler()
        
        # Test various vision commands
        test_commands = [
            "can you see my screen",
            "what's on my screen",
            "describe what you see"
        ]
        
        for cmd in test_commands:
            print(f"\n   Testing: '{cmd}'")
            start_time = time.time()
            response, handler_type = await command_handler.handle_command(cmd)
            elapsed = time.time() - start_time
            
            print(f"   Handler: {handler_type}")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Response: {response[:150]}...")
            
            if elapsed < 1.0:
                print("   ✅ Performance goal met (<1s)")
            else:
                print(f"   ⚠️  Performance slower than target: {elapsed:.2f}s > 1.0s")
                
    except Exception as e:
        print(f"❌ Command flow error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Memory usage check
    print("\n4️⃣ Checking memory usage...")
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.1f} MB")
        
        if memory_mb < 500:
            print("✅ Memory usage is excellent (<500MB)")
        elif memory_mb < 1000:
            print("✅ Memory usage is good (<1GB)")
        else:
            print(f"⚠️  Memory usage is high: {memory_mb:.1f} MB")
            
    except:
        print("Unable to check memory usage")
    
    print("\n" + "=" * 60)
    print("Vision test complete!")

if __name__ == "__main__":
    print("Starting complete vision test...\n")
    asyncio.run(test_vision_complete())