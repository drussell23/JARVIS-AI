#!/usr/bin/env python3
"""Test the enhanced Toronto selection with JARVIS"""

import asyncio
import os

async def test_enhanced_selection():
    """Test JARVIS with enhanced Toronto selection"""
    print("🌟 Testing Enhanced Toronto Selection")
    print("="*60)
    
    from api.jarvis_voice_api import JARVISVoiceAPI, JARVISCommand
    from api.jarvis_factory import set_app_state
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    from system_control.weather_system_config import initialize_weather_system
    from types import SimpleNamespace
    
    # Setup
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    weather_system = initialize_weather_system(vision, controller)
    
    app_state = SimpleNamespace(
        vision_analyzer=vision,
        weather_system=weather_system
    )
    set_app_state(app_state)
    
    jarvis_api = JARVISVoiceAPI()
    
    # Test 1: Direct enhanced navigation test
    print("\n1. Testing direct enhanced navigation...")
    from system_control.weather_navigation_enhanced import WeatherNavigationEnhanced
    nav = WeatherNavigationEnhanced(controller, vision)
    
    success = await nav.select_toronto_human_like()
    print(f"   Enhanced navigation result: {'✅ Success' if success else '❌ Failed'}")
    
    if success:
        # Verify what's showing
        result = await vision.analyze_weather_fast()
        if result.get('success'):
            analysis = result.get('analysis', '')
            print(f"\n   Currently showing: {analysis[:100]}...")
            
            if 'toronto' in analysis.lower():
                print("   🎉 Toronto is selected!")
            elif 'new york' in analysis.lower():
                print("   ⚠️  Still showing New York")
    
    # Small delay before main test
    await asyncio.sleep(2)
    
    # Test 2: Full JARVIS weather command
    print("\n2. Testing full JARVIS weather command...")
    print("   Asking: 'What's the weather for today?'")
    
    command = JARVISCommand(text="What's the weather for today?")
    result = await jarvis_api.process_command(command)
    
    response = result.get('response', '')
    print(f"\n   JARVIS Response:")
    print(f"   {response}")
    
    # Analysis
    print("\n" + "-"*60)
    print("Analysis:")
    
    if 'toronto' in response.lower():
        print("✅ SUCCESS: JARVIS is reading Toronto weather!")
        print("   The enhanced navigation worked!")
    elif 'new york' in response.lower():
        print("⚠️  JARVIS is still reading New York weather")
        print("   Enhanced navigation may need adjustment")
    elif 'having trouble' in response.lower():
        print("❌ JARVIS couldn't read the Weather app")
    else:
        print("🔍 Check the response for location information")
    
    print("\n" + "="*60)
    print("Test complete!")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_enhanced_selection())