#!/usr/bin/env python3
"""Direct test - click exactly on Toronto in Weather sidebar"""

import asyncio
import os
import subprocess
import time

async def test_direct_toronto():
    """Click directly on Toronto coordinates"""
    print("🎯 Direct Toronto Click Test")
    print("="*60)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    
    # Open Weather
    print("\n1. Opening Weather app...")
    controller.open_application("Weather")
    await asyncio.sleep(3)
    
    # Make sure Weather is active
    print("\n2. Activating Weather...")
    script = '''
    tell application "Weather"
        activate
        set frontmost to true
    end tell
    '''
    controller.execute_applescript(script)
    await asyncio.sleep(1)
    
    # From your screenshot, Toronto/My Location appears to be at the very top
    # Let's click directly on it
    print("\n3. Clicking on Toronto (My Location - top of sidebar)...")
    
    # Toronto is at the top of the sidebar
    # X coordinate: center of sidebar (about 125px)
    # Y coordinate: first item (about 120-140px from top)
    
    toronto_x = 125
    toronto_y = 130  # Adjusted for top position
    
    print(f"   Clicking at ({toronto_x}, {toronto_y})")
    await controller.click_at(toronto_x, toronto_y)
    await asyncio.sleep(1)
    
    # Keep Weather active
    controller.execute_applescript(script)
    await asyncio.sleep(2)
    
    # Analyze what's showing
    print("\n4. Analyzing current display...")
    result = await vision.analyze_weather_fast()
    
    if result.get('success'):
        analysis = result.get('analysis', '')
        print(f"\n✅ Weather display: {analysis}")
        
        # Check location
        if 'toronto' in analysis.lower():
            print("\n🎉 SUCCESS: Showing Toronto weather!")
        elif 'my location' in analysis.lower() or 'home' in analysis.lower():
            print("\n🎉 SUCCESS: Showing My Location (Toronto)!")
        else:
            print("\n❌ Still showing wrong location")
            print("\nTrying alternative coordinates...")
            
            # Try slightly different Y coordinates
            for y_offset in [150, 170, 110]:
                print(f"\n   Trying Y={y_offset}...")
                await controller.click_at(toronto_x, y_offset)
                await asyncio.sleep(1.5)
                
                # Keep Weather active
                controller.execute_applescript(script)
                
                quick_result = await vision.analyze_weather_fast()
                if quick_result.get('success'):
                    quick_analysis = quick_result.get('analysis', '')
                    print(f"   Result: {quick_analysis[:100]}...")
                    if 'toronto' in quick_analysis.lower():
                        print("\n   🎉 Found Toronto at Y={y_offset}!")
                        break

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_direct_toronto())