#!/usr/bin/env python3
"""Test JARVIS weather functionality end-to-end"""

import asyncio
import subprocess
import time
import os

async def test_weather_end_to_end():
    """Test the complete weather flow"""
    print("🌤️ Testing JARVIS Weather System - Full Mode\n")
    
    # Step 1: Ensure Weather app is open
    print("1. Opening Weather app...")
    subprocess.run(['open', '-a', 'Weather'], check=False)
    time.sleep(3)
    
    # Step 2: Initialize the system like main.py does
    print("\n2. Initializing JARVIS components...")
    
    try:
        # Import and initialize vision
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not set!")
            return
            
        vision_analyzer = ClaudeVisionAnalyzer(api_key)
        print("✅ Vision analyzer initialized")
        
        # Initialize controller
        from system_control.macos_controller import MacOSController
        controller = MacOSController()
        print("✅ MacOS controller initialized")
        
        # Initialize weather system
        from system_control.weather_system_config import initialize_weather_system
        weather_system = initialize_weather_system(vision_analyzer, controller)
        print("✅ Weather system initialized with vision")
        
        # Step 3: Test weather directly
        print("\n3. Testing weather retrieval...")
        result = await weather_system.get_weather("What's the weather today?")
        
        print("\nWeather Result:")
        print(f"Success: {result.get('success')}")
        print(f"Source: {result.get('source')}")
        
        if result.get('success'):
            print(f"\n📍 Location: {result.get('data', {}).get('location', 'Unknown')}")
            print(f"\n🌡️ Weather Data:")
            current = result.get('data', {}).get('current', {})
            print(f"   Temperature: {current.get('temperature', 'N/A')}")
            print(f"   Condition: {current.get('condition', 'N/A')}")
            
            print(f"\n💬 Response: {result.get('formatted_response', 'No response')}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
        # Step 4: Test through JARVIS API
        print("\n\n4. Testing through JARVIS API...")
        
        # Create a mock app state
        from types import SimpleNamespace
        app_state = SimpleNamespace(
            vision_analyzer=vision_analyzer,
            weather_system=weather_system
        )
        
        # Set it in the factory
        from api.jarvis_factory import set_app_state
        set_app_state(app_state)
        print("✅ App state configured")
        
        # Now test the API
        from api.jarvis_voice_api import JARVISVoiceAPI, JARVISCommand
        api = JARVISVoiceAPI()
        
        command = JARVISCommand(text="What's the weather like today?")
        api_result = await api.process_command(command)
        
        print(f"\nAPI Response:")
        print(f"Status: {api_result.get('status')}")
        print(f"Mode: {api_result.get('mode', 'not specified')}")
        print(f"Command Type: {api_result.get('command_type')}")
        print(f"\n💬 Response: {api_result.get('response', 'No response')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*60)
    print("\n✅ Test Complete!")
    print("\nFor best results:")
    print("1. Ensure Weather app shows 'My Location' in the sidebar")
    print("2. Run the backend server: python main.py")
    print("3. Ask JARVIS: 'What's the weather today?'")


if __name__ == "__main__":
    # Ensure we're in the right directory
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    
    # Check API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        exit(1)
        
    asyncio.run(test_weather_end_to_end())