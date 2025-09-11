#!/usr/bin/env python3
"""
Test that JARVIS stays on Toronto (My Location) and doesn't switch to New York
"""

import asyncio
import os
import time
import subprocess

async def test_toronto_weather():
    """Test weather reading for Toronto location"""
    print("🌤️ Testing Toronto Weather (My Location)")
    print("="*60)
    
    try:
        # Setup
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        from system_control.macos_controller import MacOSController
        from system_control.weather_system_config import initialize_weather_system
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise Exception("No ANTHROPIC_API_KEY")
            
        print("1. Initializing components...")
        vision = ClaudeVisionAnalyzer(api_key)
        controller = MacOSController()
        weather_system = initialize_weather_system(vision, controller)
        print("✅ Components ready")
        
        # Open Weather app
        print("\n2. Opening Weather app...")
        subprocess.run(['open', '-a', 'Weather'], check=False)
        await asyncio.sleep(3)
        
        # Test navigation
        print("\n3. Testing navigation to My Location...")
        from system_control.unified_vision_weather import UnifiedVisionWeather
        unified = UnifiedVisionWeather(vision, controller)
        
        # Call the navigation method directly
        await unified._select_my_location()
        
        print("\n4. Waiting to observe if it stays on Toronto...")
        await asyncio.sleep(3)
        
        # Capture and analyze
        print("\n5. Analyzing current weather view...")
        result = await vision.analyze_weather_fast()
        
        if result.get('success'):
            analysis = result.get('analysis', '')
            print(f"\n✅ Analysis: {analysis}")
            
            # Check if we're seeing Toronto or New York
            if 'toronto' in analysis.lower():
                print("\n✅ SUCCESS: Stayed on Toronto (My Location)")
                return True
            elif 'new york' in analysis.lower():
                print("\n❌ PROBLEM: Switched to New York instead of staying on Toronto")
                return False
            else:
                print(f"\n⚠️  Could not determine location from: {analysis}")
                
        else:
            print(f"\n❌ Analysis failed: {result.get('error')}")
            
        # Also test through weather system
        print("\n6. Testing through weather system...")
        weather_result = await weather_system.get_weather("What's the weather today?")
        
        if weather_result.get('success'):
            location = weather_result.get('data', {}).get('location', 'Unknown')
            response = weather_result.get('formatted_response', '')
            
            print(f"\nLocation detected: {location}")
            print(f"Response: {response[:100]}...")
            
            if 'toronto' in location.lower():
                print("\n✅ Weather system correctly reading Toronto")
            else:
                print(f"\n❌ Weather system reading wrong location: {location}")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_toronto_weather())