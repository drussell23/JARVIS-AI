#!/usr/bin/env python3
"""Test that Toronto selection persists"""

import asyncio
import os
import subprocess
import time

async def test_persistent_toronto():
    """Ensure Toronto stays selected"""
    print("🌤️ Testing Persistent Toronto Selection")
    print("="*60)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    from system_control.weather_navigation_helper import WeatherNavigationHelper
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    nav_helper = WeatherNavigationHelper(controller)
    
    # Open Weather
    print("\n1. Opening Weather app...")
    subprocess.run(['open', '-a', 'Weather'], check=False)
    await asyncio.sleep(3)
    
    # Navigate to Toronto
    print("\n2. Navigating to Toronto...")
    success = await nav_helper.select_my_location_robust()
    print(f"   Navigation success: {success}")
    
    # Wait a moment
    await asyncio.sleep(3)
    
    # Check multiple times to see if it stays on Toronto
    print("\n3. Checking if Toronto selection persists...")
    
    for i in range(3):
        print(f"\n   Check #{i+1} (after {i*2} seconds)...")
        
        # Ensure Weather is still frontmost
        controller.execute_applescript('''
            tell application "Weather"
                activate
                set frontmost to true
            end tell
        ''')
        await asyncio.sleep(0.5)
        
        # Check what's showing
        result = await vision.analyze_weather_fast()
        
        if result.get('success'):
            analysis = result.get('analysis', '')
            location = "Unknown"
            
            # Extract location
            if "Location:" in analysis:
                location = analysis.split("Location:")[1].split("\n")[0].strip()
                
            print(f"      Showing: {location}")
            
            if 'toronto' in location.lower():
                print("      ✅ Still on Toronto!")
            elif 'new york' in location.lower():
                print("      ❌ Reverted to New York")
                print("\n   Trying to re-select Toronto...")
                await nav_helper.select_my_location_robust()
            else:
                print(f"      ⚠️  Showing: {location}")
        
        if i < 2:
            await asyncio.sleep(2)
    
    # Final test through full weather system
    print("\n4. Testing through full weather system...")
    from system_control.unified_vision_weather import UnifiedVisionWeather
    
    weather = UnifiedVisionWeather(vision, controller)
    result = await weather.get_weather("What's the weather?")
    
    if result.get('success'):
        response = result.get('formatted_response', '')
        print(f"\nFinal response: {response}")
        
        if 'toronto' in response.lower():
            print("\n✅ SUCCESS: Weather system correctly reads Toronto")
        else:
            print("\n❌ ISSUE: Weather system not reading Toronto")
    else:
        print(f"\n❌ Weather system failed: {result.get('error')}")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_persistent_toronto())