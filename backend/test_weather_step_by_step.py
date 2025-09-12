#!/usr/bin/env python3
"""Step by step weather test to identify the exact failure point"""

import asyncio
import os
import subprocess
import time

async def test_step_by_step():
    """Test each step individually to find the failure"""
    print("🔍 Step-by-Step Weather Debug")
    print("="*60)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    from system_control.weather_navigation_helper import WeatherNavigationHelper
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    nav_helper = WeatherNavigationHelper(controller)
    
    # Step 1: Open Weather
    print("\n1. Opening Weather app...")
    start = time.time()
    subprocess.run(['open', '-a', 'Weather'], check=False)
    await asyncio.sleep(3)
    print(f"   ✅ Done ({time.time() - start:.1f}s)")
    
    # Step 2: Navigate to Toronto
    print("\n2. Navigating to Toronto...")
    start = time.time()
    try:
        success = await nav_helper.select_my_location_robust()
        print(f"   Navigation result: {success}")
        print(f"   ✅ Done ({time.time() - start:.1f}s)")
    except Exception as e:
        print(f"   ❌ Navigation failed: {e}")
        
    # Wait for navigation to settle
    print("\n3. Waiting for weather to load...")
    await asyncio.sleep(2)
    
    # Step 3: Take screenshot
    print("\n4. Taking screenshot...")
    start = time.time()
    try:
        screenshot = await vision.capture_screen()
        if screenshot:
            print(f"   ✅ Screenshot captured ({time.time() - start:.1f}s)")
        else:
            print("   ❌ Screenshot failed")
    except Exception as e:
        print(f"   ❌ Screenshot error: {e}")
    
    # Step 4: Analyze weather
    print("\n5. Analyzing weather...")
    start = time.time()
    try:
        # Use timeout to prevent hanging
        result = await asyncio.wait_for(
            vision.analyze_weather_fast(),
            timeout=10.0
        )
        elapsed = time.time() - start
        
        if result.get('success'):
            analysis = result.get('analysis', '')
            print(f"   ✅ Analysis successful ({elapsed:.1f}s)")
            print(f"   Result: {analysis}")
            
            # Check which city
            if 'toronto' in analysis.lower():
                print("\n   🎉 SUCCESS: Reading Toronto!")
            elif 'new york' in analysis.lower():
                print("\n   ⚠️  Reading New York")
        else:
            print(f"   ❌ Analysis failed ({elapsed:.1f}s): {result.get('error')}")
            
    except asyncio.TimeoutError:
        print(f"   ❌ Analysis timed out after 10 seconds")
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
    
    # Step 5: Test through weather system
    print("\n6. Testing through weather system...")
    from system_control.unified_vision_weather import UnifiedVisionWeather
    
    weather = UnifiedVisionWeather(vision, controller)
    start = time.time()
    
    try:
        result = await asyncio.wait_for(
            weather.get_weather("What's the weather?"),
            timeout=20.0
        )
        elapsed = time.time() - start
        
        if result.get('success'):
            print(f"   ✅ Weather system success ({elapsed:.1f}s)")
            print(f"   Response: {result.get('formatted_response', '')[:100]}...")
        else:
            print(f"   ❌ Weather system failed ({elapsed:.1f}s): {result.get('error')}")
            
    except asyncio.TimeoutError:
        print(f"   ❌ Weather system timed out after 20 seconds")
    except Exception as e:
        print(f"   ❌ Weather system error: {e}")
        
    print("\n" + "="*60)
    print("Debug complete. Check which step failed or was slow.")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_step_by_step())