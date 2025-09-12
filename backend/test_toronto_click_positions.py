#!/usr/bin/env python3
"""Test different click positions to find Toronto"""

import asyncio
import os
import subprocess

async def test_toronto_positions():
    """Try different Y coordinates to find Toronto"""
    print("🎯 Testing Different Click Positions for Toronto")
    print("="*60)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    
    # Open Weather
    print("\n1. Opening Weather app...")
    subprocess.run(['open', '-a', 'Weather'], check=False)
    await asyncio.sleep(3)
    
    # Test different Y coordinates
    # Based on your screenshot, Toronto should be at the very top
    test_positions = [
        (125, 55, "Very top of sidebar"),
        (125, 65, "Slightly lower"), 
        (125, 75, "Toronto area"),
        (125, 85, "Between Toronto and NY"),
        (125, 95, "Closer to NY"),
        (125, 105, "NY area")
    ]
    
    print("\n2. Testing different click positions...")
    
    for x, y, desc in test_positions:
        print(f"\n   Testing Y={y} ({desc})...")
        
        # Ensure Weather is active
        controller.execute_applescript('''
            tell application "Weather"
                activate
                set frontmost to true
            end tell
        ''')
        await asyncio.sleep(0.5)
        
        # Double-click at position
        await controller.click_at(x, y)
        await asyncio.sleep(0.3)
        await controller.click_at(x, y)
        await asyncio.sleep(2.5)
        
        # Check what's showing
        result = await vision.analyze_weather_fast()
        
        if result.get('success'):
            analysis = result.get('analysis', '')
            # Extract location
            location = "Unknown"
            if "Location:" in analysis:
                location = analysis.split("Location:")[1].split("\n")[0].strip()
            
            print(f"      → Showing: {location}")
            
            if 'toronto' in location.lower():
                print(f"\n   🎉 FOUND TORONTO at Y={y}!")
                
                # Verify it stays
                print("   Verifying Toronto stays selected...")
                await asyncio.sleep(2)
                verify = await vision.analyze_weather_fast()
                if verify.get('success'):
                    verify_location = "Unknown"
                    if "Location:" in verify.get('analysis', ''):
                        verify_location = verify.get('analysis').split("Location:")[1].split("\n")[0].strip()
                    print(f"   After 2 seconds: {verify_location}")
                    
                    if 'toronto' in verify_location.lower():
                        print("\n   ✅ SUCCESS: Toronto selection is stable!")
                        print(f"   Use Y={y} for Toronto selection")
                    else:
                        print("   ❌ Selection reverted")
                        
                return y
            
    print("\n❌ Could not find Toronto in tested positions")
    
    # Try one more thing - ask Claude where Toronto is
    print("\n3. Asking Claude to locate Toronto in sidebar...")
    screenshot = await vision.capture_screen()
    location_check = await vision.analyze_screenshot_async(
        screenshot,
        "In the Weather app sidebar, at what Y coordinate (pixel position from top) is 'My Location - Home' or Toronto located? Be specific.",
        quick_mode=True
    )
    print(f"\nClaude says: {location_check.get('description', 'No response')[:200]}...")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_toronto_positions())