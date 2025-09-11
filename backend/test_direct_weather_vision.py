#!/usr/bin/env python3
"""Direct test of weather vision analysis"""

import asyncio
import os
import subprocess
import time

async def test_direct_vision():
    """Test vision analysis of Weather app directly"""
    print("🌤️ Direct Weather Vision Test\n")
    
    # Open Weather app
    print("1. Opening Weather app...")
    subprocess.run(['open', '-a', 'Weather'], check=False)
    time.sleep(3)
    
    # Navigate to My Location
    print("2. Navigating to My Location...")
    subprocess.run(['osascript', '-e', '''
    tell application "System Events"
        key code 126
        delay 0.2
        key code 126  
        delay 0.2
        key code 125
        delay 0.2
        key code 36
    end tell
    '''])
    time.sleep(1)
    
    print("\n3. Initializing vision analyzer...")
    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ No API key")
            return
            
        vision = ClaudeVisionAnalyzer(api_key)
        
        # Capture screen
        print("4. Capturing screen...")
        screenshot = await vision.capture_screen()
        
        if screenshot is None:
            print("❌ Failed to capture screen")
            return
            
        # Convert PIL Image to numpy array if needed
        import numpy as np
        from PIL import Image
        
        if isinstance(screenshot, Image.Image):
            screenshot_array = np.array(screenshot)
            print(f"✅ Captured screen (PIL): {screenshot.size}")
        else:
            screenshot_array = screenshot
            print(f"✅ Captured screen (numpy): {screenshot.shape}")
            
        screenshot = screenshot_array
        
        # Test simple analysis
        print("\n5. Analyzing weather...")
        
        prompt = """Look at the Weather app and tell me:
1. What location is showing?
2. What is the current temperature?
3. What is the weather condition?

Be specific and concise."""

        result = await vision.analyze_screenshot_async(
            screenshot,
            prompt,
            quick_mode=True
        )
        
        print(f"\nResult type: {type(result)}")
        print(f"Success: {result.get('success', False)}")
        
        if result.get('success'):
            print(f"\nAnalysis content:")
            print(result.get('analysis', 'No analysis'))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            
        # Also try the simpler analyze_weather_directly method if it exists
        if hasattr(vision, 'analyze_weather_directly'):
            print("\n6. Testing analyze_weather_directly method...")
            weather = await vision.analyze_weather_directly()
            print(f"Direct weather result: {weather}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_direct_vision())