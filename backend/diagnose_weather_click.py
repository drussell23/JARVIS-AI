#!/usr/bin/env python3
"""Diagnose why Weather app doesn't respond to automated clicks"""

import asyncio
import os
import subprocess
import time

async def diagnose_weather_clicking():
    """Diagnose Weather app clicking issues"""
    print("🔍 Diagnosing Weather App Click Behavior")
    print("="*60)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    
    # Test 1: Check Weather app state
    print("\n1. Checking Weather app initial state...")
    subprocess.run(['open', '-a', 'Weather'], check=False)
    await asyncio.sleep(3)
    
    # Check what's selected
    result = await vision.analyze_weather_fast()
    if result.get('success'):
        initial_location = "Unknown"
        analysis = result.get('analysis', '')
        if "Location:" in analysis:
            initial_location = analysis.split("Location:")[1].split("\n")[0].strip()
        print(f"   Initial location: {initial_location}")
    
    # Test 2: Try different click methods
    print("\n2. Testing different click methods...")
    
    # Method A: System Events click
    print("\n   A. System Events click...")
    script = '''
    tell application "System Events"
        tell process "Weather"
            click at {125, 65}
        end tell
    end tell
    '''
    controller.execute_applescript(script)
    await asyncio.sleep(2)
    
    # Check result
    result = await vision.analyze_weather_fast()
    if result.get('success'):
        location = "Unknown"
        if "Location:" in result.get('analysis', ''):
            location = result.get('analysis').split("Location:")[1].split("\n")[0].strip()
        print(f"      Result: {location}")
    
    # Method B: Direct coordinate click
    print("\n   B. Direct coordinate click...")
    await controller.click_at(125, 65)
    await asyncio.sleep(2)
    
    result = await vision.analyze_weather_fast()
    if result.get('success'):
        location = "Unknown"
        if "Location:" in result.get('analysis', ''):
            location = result.get('analysis').split("Location:")[1].split("\n")[0].strip()
        print(f"      Result: {location}")
    
    # Method C: Click and hold
    print("\n   C. Click and hold...")
    await controller.click_and_hold(125, 65, 0.5)
    await asyncio.sleep(2)
    
    result = await vision.analyze_weather_fast()
    if result.get('success'):
        location = "Unknown"
        if "Location:" in result.get('analysis', ''):
            location = result.get('analysis').split("Location:")[1].split("\n")[0].strip()
        print(f"      Result: {location}")
    
    # Test 3: Check if Weather app has focus
    print("\n3. Checking Weather app focus...")
    script = '''
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
        return frontApp
    end tell
    '''
    success, front_app = controller.execute_applescript(script)
    print(f"   Front app: {front_app}")
    
    # Test 4: Check accessibility permissions
    print("\n4. Checking accessibility...")
    script = '''
    tell application "System Events"
        tell process "Weather"
            properties of window 1
        end tell
    end tell
    '''
    success, props = controller.execute_applescript(script)
    if success:
        print("   ✅ Accessibility working")
    else:
        print("   ❌ Accessibility issue:", props)
    
    # Test 5: Try keyboard navigation
    print("\n5. Testing keyboard navigation...")
    script = '''
    tell application "System Events"
        tell process "Weather"
            set frontmost to true
            -- Up arrows to go to top
            key code 126
            delay 0.2
            key code 126
            delay 0.2
            -- Enter to select
            key code 36
            delay 1
        end tell
    end tell
    '''
    controller.execute_applescript(script)
    await asyncio.sleep(2)
    
    result = await vision.analyze_weather_fast()
    if result.get('success'):
        location = "Unknown"
        if "Location:" in result.get('analysis', ''):
            location = result.get('analysis').split("Location:")[1].split("\n")[0].strip()
        print(f"   Keyboard result: {location}")
    
    # Summary
    print("\n" + "="*60)
    print("Diagnosis Summary:")
    print("The Weather app may have special handling that prevents")
    print("automated clicks from registering the same as manual clicks.")
    print("\nPossible reasons:")
    print("1. Weather app requires genuine user input for location changes")
    print("2. macOS security features blocking automated location selection")
    print("3. The app uses a different event handling mechanism")
    print("\nRecommendation: Manual selection remains the most reliable method")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(diagnose_weather_clicking())