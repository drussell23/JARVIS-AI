#!/usr/bin/env python3
"""Final comprehensive test for Toronto weather selection"""

import asyncio
import os
import subprocess
import time

async def test_toronto_final():
    """Final test with all improvements"""
    print("🌤️ FINAL TORONTO WEATHER TEST")
    print("="*60)
    
    from api.jarvis_voice_api import JARVISVoiceAPI, JARVISCommand
    from api.jarvis_factory import set_app_state
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    from system_control.weather_system_config import initialize_weather_system
    from types import SimpleNamespace
    
    # Initialize
    print("\n1. Initializing components...")
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    weather_system = initialize_weather_system(vision, controller)
    
    app_state = SimpleNamespace(
        vision_analyzer=vision,
        weather_system=weather_system
    )
    set_app_state(app_state)
    
    jarvis_api = JARVISVoiceAPI()
    print("✅ Components initialized")
    
    # Make sure no other apps interfere
    print("\n2. Closing potential interfering apps...")
    subprocess.run(['osascript', '-e', 'tell application "System Events" to set visible of process "Code" to false'], 
                   capture_output=True)
    subprocess.run(['osascript', '-e', 'tell application "System Events" to set visible of process "Chrome" to false'], 
                   capture_output=True)
    
    # Test weather command
    print("\n3. Asking JARVIS: 'What's the weather for today?'")
    print("-"*60)
    
    start_time = time.time()
    command = JARVISCommand(text="What's the weather for today?")
    result = await jarvis_api.process_command(command)
    elapsed = time.time() - start_time
    
    response = result.get('response', '')
    print(f"\nTime: {elapsed:.1f}s")
    print(f"JARVIS Response: {response}")
    
    # Analyze response
    print("\n4. Analysis:")
    print("-"*60)
    
    # Check for cities
    response_lower = response.lower()
    
    if 'toronto' in response_lower:
        print("✅ SUCCESS: JARVIS is reading Toronto weather!")
        print("   Your location is being correctly selected.")
        
    elif 'new york' in response_lower:
        print("❌ ISSUE: JARVIS is reading New York instead of Toronto")
        print("   The click is hitting the second item in the sidebar.")
        
        # Extract temperature to verify
        import re
        temp_match = re.search(r'(\d+)°', response)
        if temp_match:
            temp = temp_match.group(1)
            print(f"   Temperature: {temp}°F")
            if temp in ['79', '80']:
                print("   This matches New York from your screenshot (79-80°F)")
            
    elif 'my location' in response_lower or 'home' in response_lower:
        print("✅ SUCCESS: JARVIS is reading My Location (Toronto)!")
        
    else:
        print("⚠️  Could not determine which location JARVIS is reading")
        
        # Extract any temperature mentioned
        import re
        temp_match = re.search(r'(\d+)°', response)
        if temp_match:
            temp = temp_match.group(1)
            print(f"   Temperature detected: {temp}°F")
            print("   Reference: Toronto=74°F, New York=79-80°F")
    
    # Final check
    print("\n5. Summary:")
    print("-"*60)
    
    if any(indicator in response_lower for indicator in ['toronto', 'my location', '74°']):
        print("✅ Weather system is correctly reading your location (Toronto)")
    else:
        print("❌ Weather system needs coordinate adjustment")
        print("   Current behavior: Selecting New York instead of Toronto")
        print("   Solution: Need to click higher in the sidebar (Y < 80)")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_toronto_final())