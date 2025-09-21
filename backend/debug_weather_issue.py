#!/usr/bin/env python3
"""
Debug Weather Issue - Diagnose why JARVIS gets stuck
"""

import asyncio
import logging
import sys
import os
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def debug_weather_components():
    """Debug each component of the weather system"""
    print("🔍 Debugging Weather System Components")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY not found")
        return
    else:
        print("✅ API key found")
    
    # Test 1: Basic Vision Analyzer
    print("\n1. Testing Vision Analyzer Initialization:")
    print("-" * 40)
    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzerMain
        analyzer = ClaudeVisionAnalyzerMain(api_key)
        print("✅ Vision analyzer initialized")
        
        # Test screen capture
        print("\n2. Testing Screen Capture:")
        print("-" * 40)
        start = time.time()
        screenshot = await analyzer.capture_screen()
        capture_time = time.time() - start
        
        if screenshot:
            print(f"✅ Screen captured in {capture_time:.2f}s")
        else:
            print("❌ Screen capture failed")
            
    except Exception as e:
        print(f"❌ Vision analyzer error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Weather Bridge
    print("\n3. Testing Weather Bridge:")
    print("-" * 40)
    try:
        from system_control.weather_bridge import WeatherBridge
        bridge = WeatherBridge()
        
        # Check if weather query detection works
        test_query = "what's the weather today"
        is_weather = bridge.is_weather_query(test_query)
        print(f"Query '{test_query}' detected as weather: {is_weather}")
        
    except Exception as e:
        print(f"❌ Weather bridge error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Controller
    print("\n4. Testing MacOS Controller:")
    print("-" * 40)
    try:
        from system_control.macos_controller import MacOSController
        controller = MacOSController()
        
        # List open apps
        apps = controller.list_open_applications()
        print(f"✅ Controller working, {len(apps)} apps open")
        
        # Check if Weather app can be opened
        success, msg = controller.open_application("Weather")
        if success:
            print("✅ Weather app can be opened")
        else:
            print(f"⚠️  Weather app issue: {msg}")
            
    except Exception as e:
        print(f"❌ Controller error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Weather System Config
    print("\n5. Testing Weather System Configuration:")
    print("-" * 40)
    try:
        from system_control.weather_system_config import get_weather_system, initialize_weather_system
        
        # Initialize system
        weather_system = initialize_weather_system(analyzer, controller)
        if weather_system:
            print("✅ Weather system initialized")
        else:
            print("❌ Weather system initialization failed")
            
    except Exception as e:
        print(f"❌ Weather system config error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Unified Weather Workflow
    print("\n6. Testing Unified Weather Workflow:")
    print("-" * 40)
    try:
        from workflows.weather_app_vision_unified import execute_weather_app_workflow
        
        print("Executing weather workflow (10s timeout)...")
        start = time.time()
        
        try:
            result = await asyncio.wait_for(
                execute_weather_app_workflow(controller, analyzer, "What's the weather?"),
                timeout=10.0
            )
            elapsed = time.time() - start
            
            print(f"✅ Workflow completed in {elapsed:.2f}s")
            print(f"Result: {result[:100]}...")
            
        except asyncio.TimeoutError:
            print("❌ Workflow timed out after 10 seconds!")
            
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 7: Direct Vision Weather Read
    print("\n7. Testing Direct Vision Weather Read:")
    print("-" * 40)
    try:
        # Try to read weather directly
        print("Attempting direct weather analysis...")
        result = await analyzer.analyze_weather_directly()
        
        if result:
            print(f"✅ Direct analysis succeeded: {result[:100]}...")
        else:
            print("❌ Direct analysis returned None")
            
    except Exception as e:
        print(f"❌ Direct analysis error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\n🔍 Diagnosis Summary:")
    print("=" * 60)
    print("Check the results above to identify where the issue occurs.")
    print("\nCommon issues:")
    print("1. API timeout - Claude API might be slow")
    print("2. Weather app not opening - macOS permissions")
    print("3. Vision capture failing - screen recording permissions")
    print("4. Import errors - missing dependencies")


async def test_jarvis_handler():
    """Test JARVIS weather handler directly"""
    print("\n\n🤖 Testing JARVIS Weather Handler Directly")
    print("=" * 60)
    
    try:
        from voice.jarvis_agent_voice import JARVISAgentVoice
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzerMain
        
        # Initialize
        api_key = os.getenv("ANTHROPIC_API_KEY")
        vision_analyzer = ClaudeVisionAnalyzerMain(api_key)
        jarvis = JARVISAgentVoice(user_name="Sir", vision_analyzer=vision_analyzer)
        
        # Test weather handler with timeout
        print("Testing _handle_weather_command with 15s timeout...")
        start = time.time()
        
        try:
            response = await asyncio.wait_for(
                jarvis._handle_weather_command("what's the weather today"),
                timeout=15.0
            )
            elapsed = time.time() - start
            
            print(f"✅ Handler completed in {elapsed:.2f}s")
            print(f"Response: {response}")
            
        except asyncio.TimeoutError:
            print("❌ Handler timed out after 15 seconds!")
            print("This is likely where JARVIS gets stuck.")
            
    except Exception as e:
        print(f"❌ JARVIS handler error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run debug tests
    asyncio.run(debug_weather_components())
    asyncio.run(test_jarvis_handler())