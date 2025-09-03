#!/usr/bin/env python3
"""Test core vision functionality without optional dependencies"""

import asyncio
import os
import logging
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_core():
    """Test core vision functionality"""
    # Import from the main file directly
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("Please set ANTHROPIC_API_KEY")
        return False
    
    print("\n🧪 Testing Core Vision Functionality")
    print("="*50)
    
    # Initialize without real-time features
    jarvis = ClaudeVisionAnalyzer(api_key, enable_realtime=False)
    print("✅ Initialized vision analyzer")
    
    # Test 1: Screen capture
    print("\n📸 Test 1: Screen capture")
    screenshot = await jarvis.capture_screen()
    if screenshot:
        print("✅ Screen capture working")
    else:
        print("❌ Screen capture failed")
        return False
    
    # Test 2: Basic analysis
    print("\n🔍 Test 2: Basic analysis")
    import numpy as np
    from PIL import Image
    
    # Convert to numpy array if needed
    if isinstance(screenshot, Image.Image):
        screenshot = np.array(screenshot)
    
    # Test the clean method (returns dict)
    result = await jarvis.analyze_screenshot_clean(screenshot, "What's on screen?")
    if result and 'description' in result:
        print("✅ Basic analysis working")
        print(f"   Description: {result['description'][:100]}...")
    else:
        print("❌ Basic analysis failed")
        return False
    
    # Test 3: Screen context
    print("\n📍 Test 3: Screen context")
    context = await jarvis.get_screen_context()
    if context and 'description' in context:
        print("✅ Screen context working")
        print(f"   Apps: {', '.join(context.get('applications', []))[:50]}...")
    else:
        print("❌ Screen context failed")
        return False
    
    # Test 4: Command response
    print("\n💬 Test 4: Command response")
    response = await jarvis.see_and_respond("What application is currently active?")
    if response.get('success'):
        print("✅ Command response working")
        print(f"   Response: {response['response'][:100]}...")
    else:
        print("❌ Command response failed")
        return False
    
    # Test 5: Memory health
    print("\n💾 Test 5: Memory health")
    health = await jarvis.check_memory_health()
    print(f"✅ Memory health check working")
    print(f"   Process RAM: {health['process_mb']:.1f} MB")
    print(f"   System available: {health['system_available_gb']:.1f} GB")
    print(f"   Health status: {'Healthy' if health['healthy'] else 'Unhealthy'}")
    
    # Cleanup
    await jarvis.cleanup_all_components()
    
    print("\n✅ All core tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_core())
    if success:
        print("\n🎉 Core vision system is working perfectly!")
        print("\nYou can now use the vision analyzer for JARVIS with:")
        print("  from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer")
        print("\nAll wrapper methods are available directly from the main class.")
    else:
        print("\n❌ Some tests failed. Check the output above.")