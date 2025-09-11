#!/usr/bin/env python3
"""Diagnose why Toronto isn't being selected"""

import asyncio
import os
import subprocess

async def diagnose_selection():
    """Diagnose the selection issue"""
    print("🔍 Diagnosing Weather App Selection Issue")
    print("="*60)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    
    # Open Weather
    print("\n1. Opening Weather app...")
    controller.open_application("Weather")
    await asyncio.sleep(3)
    
    # Take screenshot
    print("\n2. Analyzing current state...")
    screenshot = await vision.capture_screen()
    
    # Ask Claude specific questions
    print("\n3. Asking Claude to analyze the Weather app...")
    
    analysis = await vision.analyze_screenshot_async(
        screenshot,
        """Look at the macOS Weather app and answer these questions:
        1. What location is currently selected/highlighted in the sidebar?
        2. What location's weather is shown in the main display?
        3. List ALL locations visible in the sidebar from top to bottom
        4. Is "My Location - Home" visible? If yes, what position is it in?
        5. What temperature is shown for the selected location?""",
        quick_mode=True
    )
    
    print(f"\nClaude's Analysis:\n{analysis.get('description', 'No analysis')}")
    
    # Try different selection methods
    print("\n\n4. Testing Selection Methods:")
    print("-"*40)
    
    methods = [
        ("Keyboard Command+1", lambda: controller.execute_applescript(
            'tell application "System Events" to keystroke "1" using command down')),
        ("Click at Y=45", lambda: asyncio.create_task(controller.click_at(125, 45))),
        ("Arrow keys", lambda: controller.execute_applescript(
            '''tell application "System Events"
                key code 126
                key code 126
                key code 126
                key code 36
            end tell'''))
    ]
    
    for name, method in methods:
        print(f"\nTrying: {name}")
        
        # Ensure Weather is active
        controller.execute_applescript('tell application "Weather" to activate')
        await asyncio.sleep(0.5)
        
        # Try the method
        result = await method() if asyncio.iscoroutine(method()) else method()
        await asyncio.sleep(2)
        
        # Quick check what's showing
        weather = await vision.analyze_weather_fast()
        if weather.get('success'):
            location = weather.get('analysis', '')[:50]
            print(f"   Result: {location}...")
            if 'toronto' in location.lower() or '74' in location:
                print("   ✅ This method selected Toronto!")
                break
    
    print("\n\n5. Summary:")
    print("-"*40)
    print("The issue appears to be that Weather app is:")
    print("- Either not responding to our selection commands")
    print("- Or immediately defaulting back to New York")
    print("- Or the sidebar coordinates are different than expected")
    print("\nManual verification needed:")
    print("1. Open Weather app manually")
    print("2. Click on 'My Location - Home' (Toronto)")
    print("3. Verify it stays selected and shows Toronto weather")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(diagnose_selection())