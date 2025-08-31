#!/usr/bin/env python3
"""
Test script to verify vision capture functionality
"""

import sys
import os
sys.path.append('backend')

from vision.screen_capture_fallback import capture_screen_fallback, capture_with_intelligence

def test_basic_capture():
    """Test basic screen capture"""
    print("Testing basic screen capture...")
    screenshot = capture_screen_fallback()
    
    if screenshot is None:
        print("❌ Failed to capture screen")
        print("Possible reasons:")
        print("  - Screen recording permission not granted")
        print("  - screencapture command not available")
        return False
    else:
        print(f"✅ Successfully captured screen: {screenshot.size}")
        return True

def test_claude_capture():
    """Test Claude-enhanced capture"""
    print("\nTesting Claude-enhanced capture...")
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set, skipping Claude test")
        return True
    
    result = capture_with_intelligence("What applications are open on the screen?", use_claude=True)
    
    if result["success"]:
        if result.get("intelligence_used"):
            print("✅ Claude analysis successful")
            if result.get("analysis"):
                print(f"Analysis: {result['analysis'][:200]}...")
        else:
            print("⚠️  Capture succeeded but Claude was not used")
    else:
        print(f"❌ Capture failed: {result.get('error', 'Unknown error')}")
        return False
    
    return True

def test_vision_system():
    """Test full vision system"""
    print("\nTesting Vision System V2...")
    
    try:
        from vision.vision_system_v2 import VisionSystemV2
        import asyncio
        
        async def test_async():
            vision = VisionSystemV2()
            
            # Test simple command
            result = await vision.process_command("What can you see on the screen?")
            
            if result.success:
                print("✅ Vision System V2 working")
                print(f"Response: {result.message[:200]}...")
                return True
            else:
                print(f"❌ Vision System V2 failed: {result.message}")
                return False
        
        return asyncio.run(test_async())
        
    except Exception as e:
        print(f"❌ Error loading Vision System V2: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔍 JARVIS Vision System Test")
    print("=" * 50)
    
    results = []
    
    # Test basic capture
    results.append(("Basic Capture", test_basic_capture()))
    
    # Test Claude capture
    results.append(("Claude Capture", test_claude_capture()))
    
    # Test full vision system
    results.append(("Vision System V2", test_vision_system()))
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 50)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✨ All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()