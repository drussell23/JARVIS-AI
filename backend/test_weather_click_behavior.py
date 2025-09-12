#!/usr/bin/env python3
"""Test to understand why automated clicks don't work like manual clicks"""

import asyncio
import os
import subprocess

async def test_click_behavior():
    """Test different clicking approaches on Weather app"""
    print("🔬 Testing Weather App Click Behavior")
    print("="*60)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    
    # Ensure Weather is open
    print("\n1. Opening Weather app...")
    subprocess.run(['open', '-a', 'Weather'], check=False)
    await asyncio.sleep(3)
    
    # Check initial state
    print("\n2. Checking initial state...")
    result = await vision.analyze_weather_fast()
    if result.get('success'):
        analysis = result.get('analysis', '')
        print(f"   Currently showing: {analysis[:50]}...")
        initial_location = "new york" in analysis.lower()
        print(f"   Is New York selected? {initial_location}")
    
    # Test 1: Try the exact coordinates where Toronto appears
    print("\n3. Testing precise Toronto click coordinates...")
    print("   Clicking at (125, 65) where Toronto should be...")
    
    # Make Weather absolutely frontmost
    script = '''
    tell application "Weather"
        activate
        set frontmost to true
    end tell
    tell application "System Events"
        set frontmost of process "Weather" to true
    end tell
    '''
    controller.execute_applescript(script)
    await asyncio.sleep(1)
    
    # Try multiple click approaches
    print("\n   a) Single click...")
    await controller.click_at(125, 65)
    await asyncio.sleep(2)
    
    result1 = await vision.analyze_weather_fast()
    if result1.get('success'):
        if 'toronto' in result1.get('analysis', '').lower():
            print("      ✅ Toronto selected with single click!")
        else:
            print("      ❌ Still not showing Toronto")
    
    print("\n   b) Double click...")
    await controller.click_at(125, 65)
    await asyncio.sleep(0.2)
    await controller.click_at(125, 65)
    await asyncio.sleep(2)
    
    result2 = await vision.analyze_weather_fast()
    if result2.get('success'):
        if 'toronto' in result2.get('analysis', '').lower():
            print("      ✅ Toronto selected with double click!")
        else:
            print("      ❌ Still not showing Toronto")
    
    print("\n   c) Click and hold...")
    await controller.click_and_hold(125, 65, 0.5)
    await asyncio.sleep(2)
    
    result3 = await vision.analyze_weather_fast()
    if result3.get('success'):
        if 'toronto' in result3.get('analysis', '').lower():
            print("      ✅ Toronto selected with click and hold!")
        else:
            print("      ❌ Still not showing Toronto")
    
    # Test 2: Try UI element selection
    print("\n4. Testing UI element selection...")
    script = '''
    tell application "System Events"
        tell process "Weather"
            set frontmost to true
            
            -- Try to find and click Toronto in the sidebar
            try
                tell window 1
                    tell scroll area 1 of splitter group 1
                        tell table 1 of scroll area 1
                            -- First row should be Toronto/My Location
                            click row 1
                            delay 1
                            return "clicked row 1"
                        end tell
                    end tell
                end tell
            on error errMsg
                return "error: " & errMsg
            end try
        end tell
    end tell
    '''
    
    success, result = controller.execute_applescript(script)
    print(f"   AppleScript result: {result}")
    await asyncio.sleep(2)
    
    result4 = await vision.analyze_weather_fast()
    if result4.get('success'):
        if 'toronto' in result4.get('analysis', '').lower():
            print("   ✅ Toronto selected with UI element click!")
        else:
            print("   ❌ Still not showing Toronto")
    
    # Test 3: Simulate exact mouse events
    print("\n5. Testing mouse event simulation...")
    # Try using cliclick if available
    try:
        # Check if cliclick is available
        subprocess.run(['which', 'cliclick'], check=True, capture_output=True)
        print("   Using cliclick for precise mouse control...")
        
        # Move mouse to position and click
        subprocess.run(['cliclick', 'c:125,65'], check=True)
        await asyncio.sleep(2)
        
        result5 = await vision.analyze_weather_fast()
        if result5.get('success'):
            if 'toronto' in result5.get('analysis', '').lower():
                print("   ✅ Toronto selected with cliclick!")
            else:
                print("   ❌ Still not showing Toronto")
    except:
        print("   cliclick not available")
    
    # Final check
    print("\n6. Final state check...")
    final_result = await vision.analyze_weather_fast()
    if final_result.get('success'):
        print(f"   Final location: {final_result.get('analysis', '')[:50]}...")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("The Weather app appears to have special behavior that")
    print("differentiates between automated and manual clicks.")
    print("\nThis could be due to:")
    print("1. macOS security features protecting location data")
    print("2. Weather app requiring genuine user input events")
    print("3. Different event handling for programmatic vs user clicks")
    
    print("\n🔍 IMPORTANT: Please try this manually:")
    print("1. Click on Toronto in the Weather app")
    print("2. Close the Weather app completely")
    print("3. Reopen it - does it stay on Toronto?")
    print("\nThis will help us understand if the issue is with")
    print("selection persistence or click registration.")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_click_behavior())