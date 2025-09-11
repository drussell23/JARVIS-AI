#!/usr/bin/env python3
"""Test weather functionality in limited mode"""

import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, '.')

async def test_weather_in_limited_mode():
    """Test that weather works even in limited mode"""
    print("🌤️ Testing Weather in Limited Mode\n")
    
    # First test: Check if weather system is available
    try:
        # Add parent directory to Python path for imports
        import sys
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            
        from system_control.weather_system_config import get_weather_system
        weather_system = get_weather_system()
        
        if weather_system:
            print("✅ Weather system is available")
        else:
            print("⚠️ Weather system not initialized, will initialize now")
            
            # Initialize weather system
            from system_control.weather_system_config import initialize_weather_system
            from system_control.macos_controller import MacOSController
            
            # Try to get vision analyzer from app state
            vision_analyzer = None
            try:
                from api.jarvis_factory import get_vision_analyzer
                vision_analyzer = get_vision_analyzer()
            except:
                print("ℹ️ Vision analyzer not available from factory")
            
            controller = MacOSController()
            weather_system = initialize_weather_system(vision_analyzer, controller)
            print("✅ Weather system initialized")
            
    except Exception as e:
        print(f"❌ Failed to load weather system: {e}")
        return False
    
    # Second test: Try to get weather
    try:
        print("\n📍 Testing weather retrieval...")
        result = await weather_system.get_weather("What's the weather today?")
        
        if result.get('success'):
            print("✅ Weather retrieved successfully!")
            print(f"   Location: {result.get('data', {}).get('location', 'Unknown')}")
            print(f"   Response: {result.get('formatted_response', 'No response')[:100]}...")
        else:
            print(f"⚠️ Weather retrieval failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error getting weather: {e}")
        import traceback
        traceback.print_exc()
        
    # Third test: Test JARVIS API handling of weather in limited mode
    try:
        print("\n🤖 Testing JARVIS API weather handling...")
        
        # Clear API key to simulate limited mode
        old_key = os.environ.get('ANTHROPIC_API_KEY')
        os.environ.pop('ANTHROPIC_API_KEY', None)
        
        from api.jarvis_voice_api import JARVISVoiceAPI
        from api.jarvis_voice_api import JARVISCommand
        
        api = JARVISVoiceAPI()
        
        # Test weather command
        command = JARVISCommand(text="What's the weather today?")
        result = await api.process_command(command)
        
        print(f"✅ API Response: {result.get('response', 'No response')[:150]}...")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Command Type: {result.get('command_type', 'unknown')}")
        
        # Restore API key
        if old_key:
            os.environ['ANTHROPIC_API_KEY'] = old_key
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return True


async def test_my_location_click():
    """Test clicking on My Location in Weather app"""
    print("\n🖱️ Testing My Location Click Functionality\n")
    
    try:
        from system_control.macos_controller import MacOSController
        controller = MacOSController()
        
        # First open Weather app
        print("Opening Weather app...")
        success, msg = controller.open_application("Weather")
        if success:
            print("✅ Weather app opened")
            await asyncio.sleep(2)  # Wait for app to open
        else:
            print(f"❌ Failed to open Weather app: {msg}")
            return False
            
        # Test keyboard navigation
        print("\nTesting keyboard navigation to select My Location...")
        
        # Navigate to top of location list
        success, msg = await controller.key_press('up')
        print(f"   Up arrow: {msg}")
        await asyncio.sleep(0.5)
        
        success, msg = await controller.key_press('up')
        print(f"   Up arrow: {msg}")
        await asyncio.sleep(0.5)
        
        # Select first item (usually My Location)
        success, msg = await controller.key_press('down')
        print(f"   Down arrow: {msg}")
        await asyncio.sleep(0.5)
        
        success, msg = await controller.key_press('return')
        print(f"   Return key: {msg}")
        
        print("\n✅ Keyboard navigation test completed")
        
    except Exception as e:
        print(f"❌ Click test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("JARVIS Weather System Test - Limited Mode")
    print("=" * 60)
    
    # Run tests
    asyncio.run(test_weather_in_limited_mode())
    asyncio.run(test_my_location_click())
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\nNOTE: For best results, make sure:")
    print("1. Weather app is installed and configured")
    print("2. You have at least one location in Weather app")
    print("3. 'My Location' is available in the sidebar")
    print("4. Vision analyzer has access to screen capture")