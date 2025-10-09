#!/usr/bin/env python3
"""Quick test to verify weather is working"""

import asyncio
import os
import subprocess
import time

async def quick_weather_test():
    """Quick test of the weather functionality"""
    print("🌤️ Quick Weather Test\n")
    
    # Step 1: Open Weather app
    print("1. Opening Weather app...")
    subprocess.run(['open', '-a', 'Weather'], check=False)
    time.sleep(3)
    
    # Step 2: Initialize vision
    print("2. Initializing vision analyzer...")
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ No ANTHROPIC_API_KEY")
        return
        
    vision = ClaudeVisionAnalyzer(api_key)
    
    # Step 3: Test fast weather analysis
    print("3. Analyzing weather...")
    try:
        result = await vision.analyze_weather_fast()
        
        if result.get('success'):
            print("\n✅ Success!")
            print(f"Analysis: {result.get('analysis')}")
        else:
            print(f"\n❌ Failed: {result.get('error')}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    # Step 4: Test through weather system
    print("\n4. Testing through weather system...")
    try:
        from system_control.macos_controller import MacOSController
        from system_control.weather_system_config import initialize_weather_system
        
        controller = MacOSController()
        weather_system = initialize_weather_system(vision, controller)
        
        result = await weather_system.get_weather("What's the weather?")
        
        if result.get('success'):
            print("✅ Weather system success!")
            print(f"Response: {result.get('formatted_response')}")
        else:
            print(f"❌ Weather system failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Weather system error: {e}")


if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(quick_weather_test())