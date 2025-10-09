#!/usr/bin/env python3
"""
Test clicking directly on My Location using coordinates
"""

import asyncio
import os
import subprocess

async def test_click_my_location():
    """Test direct clicking on My Location"""
    print("🎯 Testing Direct Click on My Location")
    print("="*60)
    
    try:
        from system_control.macos_controller import MacOSController
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        controller = MacOSController()
        vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
        
        # Open Weather app
        print("1. Opening Weather app...")
        subprocess.run(['open', '-a', 'Weather'], check=False)
        await asyncio.sleep(3)
        
        # First, let's see what's on screen
        print("\n2. Analyzing screen to find My Location...")
        screenshot = await vision.capture_screen()
        
        # Ask Claude to find the exact position of "My Location"
        result = await vision.analyze_screenshot_async(
            screenshot,
            "Find the text 'My Location' or 'Home' in the Weather app sidebar. Tell me approximately where it is on screen (top/middle/bottom of sidebar, rough pixel coordinates if possible).",
            quick_mode=True
        )
        
        print(f"\nAnalysis: {result}")
        
        # Based on typical Weather app layout, My Location is usually at top of sidebar
        # Sidebar is typically on the left, about 250px wide
        # First item is usually around 100-150px from top
        
        print("\n3. Clicking on My Location coordinates...")
        # Try clicking at typical My Location position
        x = 125  # Middle of sidebar
        y = 150  # Top item position
        
        success, msg = await controller.click_at(x, y)
        print(f"Click result: {msg}")
        
        await asyncio.sleep(2)
        
        # Verify what's showing now
        print("\n4. Verifying selection...")
        weather_result = await vision.analyze_weather_fast()
        
        if weather_result.get('success'):
            analysis = weather_result.get('analysis', '')
            print(f"\n✅ Current weather display: {analysis}")
            
            if 'toronto' in analysis.lower() or 'home' in analysis.lower():
                print("\n✅ SUCCESS: Showing My Location (Toronto)")
            else:
                print("\n⚠️  Still showing different location")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_click_my_location())