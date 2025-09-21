#!/usr/bin/env python3
"""Find Toronto by its temperature (74°F)"""

import asyncio
import os
import subprocess

async def find_toronto_by_temp():
    """Find Toronto by looking for 74°F in sidebar"""
    print("🔍 Finding Toronto by Temperature (74°F)")
    print("="*60)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    
    # Open Weather
    print("\n1. Opening Weather app...")
    controller.open_application("Weather")
    await asyncio.sleep(3)
    
    # Ask Claude to find 74°F
    print("\n2. Looking for Toronto (74°F) in sidebar...")
    screenshot = await vision.capture_screen()
    
    result = await vision.analyze_screenshot_async(
        screenshot,
        """Look at the Weather app sidebar. Find the location that shows 74°F. 
        What is the exact Y coordinate (vertical position) of the 74°F location?
        Count from top: is it the 1st, 2nd, 3rd, or 4th item in the list?""",
        quick_mode=True
    )
    
    print(f"\nAnalysis: {result.get('description', 'Unknown')}")
    
    # Based on the analysis, try different Y coordinates
    print("\n3. Testing different Y coordinates...")
    
    # Weather app typically has items spaced about 30-40 pixels apart
    # Starting from top of sidebar
    test_coords = [
        (125, 85, "Very top"),
        (125, 105, "First item area"), 
        (125, 125, "Between first and second"),
        (125, 145, "Second item area"),
        (125, 90, "Extreme top")
    ]
    
    for x, y, desc in test_coords:
        print(f"\n   Testing {desc} at Y={y}...")
        
        # Ensure Weather is active
        controller.execute_applescript('''
            tell application "Weather"
                activate
                set frontmost to true
            end tell
        ''')
        await asyncio.sleep(0.3)
        
        # Double-click
        await controller.click_at(x, y)
        await asyncio.sleep(0.2)
        await controller.click_at(x, y)
        await asyncio.sleep(2)
        
        # Check result
        weather_result = await vision.analyze_weather_fast()
        if weather_result.get('success'):
            analysis = weather_result.get('analysis', '')
            temp_str = analysis[:50]
            print(f"      Result: {temp_str}...")
            
            if '74' in analysis:
                print(f"\n   🎉 FOUND IT! Toronto is at Y={y}")
                print(f"   Full weather: {analysis}")
                return y

    print("\n❌ Could not find Toronto by temperature")
    return None

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    y_coord = asyncio.run(find_toronto_by_temp())
    
    if y_coord:
        print(f"\n\n✅ SOLUTION: Use Y coordinate {y_coord} for Toronto!")
        print("Update weather_navigation_helper.py with this value.")