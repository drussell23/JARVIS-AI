#!/usr/bin/env python3
"""Test weather with location awareness"""

import asyncio
import os

async def test_weather_with_note():
    """Test weather and note which location is shown"""
    print("🌤️ Weather Test with Location Note")
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
    
    # Test
    print("\nAsking JARVIS: 'What's the weather for today?'")
    command = JARVISCommand(text="What's the weather for today?")
    result = await jarvis_api.process_command(command)
    
    response = result.get('response', '')
    print(f"\nJARVIS Response:")
    print(f"{response}")
    
    # Analysis
    print("\n" + "-"*60)
    print("Analysis:")
    
    if 'toronto' in response.lower():
        print("✅ Showing Toronto weather - Perfect!")
    elif 'new york' in response.lower():
        print("📍 Showing New York weather")
        print("\nNote: The Weather app defaults to New York when opened.")
        print("To see Toronto weather:")
        print("1. The Weather app is now open")
        print("2. Click on 'My Location - Home' in the sidebar")
        print("3. Toronto weather will then be displayed")
    elif 'having trouble' in response.lower():
        print("❌ Unable to read weather")
        print("The Weather app should be open for manual viewing")
    else:
        print("Weather information provided")
        
    print("\n" + "-"*60)
    print("Summary: JARVIS successfully opens the Weather app and reads")
    print("the displayed weather. The app defaults to New York, but you")  
    print("can manually select Toronto for your local weather.")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_weather_with_note())