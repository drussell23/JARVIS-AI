#!/usr/bin/env python3
"""Debug weather navigation and reading"""

import asyncio
import os
import subprocess
import time

async def debug_weather():
    """Step by step debug of weather functionality"""
    print("🔍 Weather Debug Test")
    print("="*60)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    
    # Step 1: Open Weather
    print("\n1. Opening Weather app...")
    controller.open_application("Weather")
    await asyncio.sleep(3)
    
    # Step 2: Make sure it's frontmost
    print("\n2. Ensuring Weather is frontmost...")
    script = '''
    tell application "Weather"
        activate
        set frontmost to true
    end tell
    '''
    success, msg = controller.execute_applescript(script)
    print(f"   Frontmost result: {success} - {msg}")
    await asyncio.sleep(1)
    
    # Step 3: Take screenshot and see what's visible
    print("\n3. Capturing screen...")
    screenshot = await vision.capture_screen()
    if screenshot:
        print("   ✅ Screenshot captured")
    
    # Step 4: Quick analysis to see what app is visible
    print("\n4. Checking what's on screen...")
    result = await vision.analyze_screenshot_async(
        screenshot, 
        "What application is currently visible on screen? Is it the Weather app?",
        quick_mode=True
    )
    print(f"   Visible: {result.get('description', 'Unknown')[:200]}...")
    
    # Step 5: Click on My Location
    print("\n5. Clicking on My Location...")
    # First click to focus sidebar
    await controller.click_at(125, 150)
    await asyncio.sleep(0.5)
    
    # Ensure Weather still frontmost
    controller.execute_applescript(script)
    await asyncio.sleep(0.5)
    
    # Step 6: Analyze weather
    print("\n6. Analyzing weather...")
    weather_result = await vision.analyze_weather_fast()
    
    if weather_result.get('success'):
        print(f"\n✅ Weather analysis: {weather_result.get('analysis')}")
    else:
        print(f"\n❌ Analysis failed: {weather_result.get('error')}")
    
    # Step 7: Check location
    print("\n7. Final check...")
    if weather_result.get('success'):
        analysis = weather_result.get('analysis', '').lower()
        if 'toronto' in analysis:
            print("✅ SUCCESS: Showing Toronto weather!")
        elif 'new york' in analysis:
            print("❌ ISSUE: Still showing New York")
        else:
            print(f"⚠️  Showing: {weather_result.get('analysis')[:100]}")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(debug_weather())