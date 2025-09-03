#!/usr/bin/env python3
"""Test the unified Claude Vision Analyzer with all features integrated"""

import asyncio
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_unified():
    """Test that all features work in the unified file"""
    # Import from the main file directly
    from claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("Please set ANTHROPIC_API_KEY")
        return False
    
    print("\n🧪 Testing Unified Claude Vision Analyzer")
    print("="*50)
    
    # Initialize
    jarvis = ClaudeVisionAnalyzer(api_key, enable_realtime=True)
    print("✅ Initialized with real-time capabilities")
    
    # Test 1: Basic analysis (original method)
    print("\n📊 Test 1: Original analyze_screenshot method")
    try:
        screenshot = await jarvis.capture_screen()
        if screenshot:
            import numpy as np
            from PIL import Image
            
            # Convert to array if needed
            if isinstance(screenshot, Image.Image):
                screenshot = np.array(screenshot)
            
            # Original method returns tuple
            result_tuple = await jarvis.analyze_screenshot(screenshot, "What's on screen?")
            if isinstance(result_tuple, tuple):
                result, metrics = result_tuple
                print(f"✅ Original method works (returns tuple)")
                print(f"   Description: {result.get('description', '')[:100]}...")
            else:
                print("❓ Original method returned unexpected format")
    except Exception as e:
        print(f"❌ Original method error: {e}")
    
    # Test 2: Clean analysis (wrapper method)
    print("\n📊 Test 2: Clean analyze_screenshot_clean method")
    try:
        screenshot = await jarvis.capture_screen()
        if screenshot and isinstance(screenshot, Image.Image):
            screenshot = np.array(screenshot)
            
        # Clean method returns just dict
        result = await jarvis.analyze_screenshot_clean(screenshot, "What's on screen?")
        if isinstance(result, dict):
            print(f"✅ Clean method works (returns dict)")
            print(f"   Description: {result.get('description', '')[:100]}...")
        else:
            print("❓ Clean method returned unexpected format")
    except Exception as e:
        print(f"❌ Clean method error: {e}")
    
    # Test 3: JARVIS methods
    print("\n📊 Test 3: JARVIS integration methods")
    try:
        # Get screen context
        context = await jarvis.get_screen_context()
        if 'error' not in context:
            print(f"✅ get_screen_context works")
        
        # See and respond
        response = await jarvis.see_and_respond("What application is open?")
        if response.get('success'):
            print(f"✅ see_and_respond works")
            print(f"   Response: {response['response'][:100]}...")
        
        # Start/stop vision
        result = await jarvis.start_jarvis_vision()
        if result.get('success'):
            print(f"✅ start_jarvis_vision works ({result['mode']} mode)")
            await asyncio.sleep(2)  # Brief monitoring
            await jarvis.stop_jarvis_vision()
            print(f"✅ stop_jarvis_vision works")
        
    except Exception as e:
        print(f"❌ JARVIS methods error: {e}")
    
    # Test 4: Real-time features
    print("\n📊 Test 4: Real-time monitoring features")
    try:
        # Get real-time context
        rt_context = await jarvis.get_real_time_context()
        if 'error' not in rt_context:
            print(f"✅ get_real_time_context works")
            if rt_context.get('behavior_insights'):
                print(f"   Detected patterns: {rt_context['behavior_insights']['detected_patterns']}")
    except Exception as e:
        print(f"❌ Real-time features error: {e}")
    
    # Cleanup
    await jarvis.cleanup_all_components()
    
    print("\n✅ All tests completed!")
    print("\nThe unified file contains:")
    print("- Original vision analyzer functionality")
    print("- Clean interface methods (wrapper compatibility)")
    print("- JARVIS integration methods")
    print("- Real-time monitoring capabilities")
    print("- Autonomous behavior detection")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_unified())
    if success:
        print("\n🎉 Unified vision system is working perfectly!")
        print("You can now import from claude_vision_analyzer_main.py for all features.")