#!/usr/bin/env python3
"""
Debug script to test vision system and diagnose the data type error
"""

import os
import sys
import asyncio
from pathlib import Path
import numpy as np
from PIL import Image

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_vision_debug():
    """Debug the vision system step by step"""
    
    print("🔍 Vision System Debug Test")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found!")
        return
    print("✅ API key found")
    
    # Test 1: Screen capture
    print("\n1. Testing screen capture...")
    try:
        from vision.screen_vision import ScreenVisionSystem
        vision_system = ScreenVisionSystem()
        
        # Try to capture screen
        screenshot = vision_system.capture_screen()
        if screenshot:
            print(f"✅ Screen captured: {type(screenshot)}")
            print(f"   Image size: {screenshot.size}")
            print(f"   Image mode: {screenshot.mode}")
            
            # Check numpy conversion
            try:
                np_array = np.array(screenshot)
                print(f"   Numpy dtype: {np_array.dtype}")
                print(f"   Numpy shape: {np_array.shape}")
            except Exception as e:
                print(f"   ❌ Numpy conversion error: {e}")
        else:
            print("❌ Failed to capture screen")
            
    except Exception as e:
        print(f"❌ Screen capture error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Claude Vision Analyzer
    print("\n2. Testing Claude Vision Analyzer...")
    try:
        from vision.claude_vision_analyzer import ClaudeVisionAnalyzer
        
        analyzer = ClaudeVisionAnalyzer(api_key)
        print("✅ Claude analyzer initialized")
        
        # Test with a simple image
        test_image = Image.new('RGB', (100, 100), color='red')
        print(f"   Test image: {type(test_image)}")
        
        result = await analyzer.analyze_screenshot(
            test_image, 
            "This is a test image. What do you see?"
        )
        print(f"✅ Claude analysis successful: {result.get('description', '')[:100]}...")
        
    except Exception as e:
        print(f"❌ Claude analyzer error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Full capture_and_describe flow
    print("\n3. Testing capture_and_describe...")
    try:
        result = await vision_system.capture_and_describe()
        print(f"Result: {result[:200]}...")
        
        if "error analyzing your screen" in result:
            print("❌ Error in capture_and_describe")
        else:
            print("✅ capture_and_describe completed")
            
    except Exception as e:
        print(f"❌ capture_and_describe error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Intelligent vision integration
    print("\n4. Testing intelligent vision integration...")
    try:
        from vision.intelligent_vision_integration import IntelligentJARVISVision
        
        intelligent_vision = IntelligentJARVISVision()
        response = await intelligent_vision.handle_intelligent_command("can you see my screen?")
        print(f"Response: {response[:200]}...")
        
    except Exception as e:
        print(f"❌ Intelligent vision error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting vision debug test...\n")
    asyncio.run(test_vision_debug())
    print("\n" + "=" * 60)
    print("Debug test complete.")