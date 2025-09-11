#!/usr/bin/env python3
"""
Test JARVIS Weather Vision Integration
Verifies that JARVIS uses vision to read weather instead of generic responses
"""

import asyncio
import sys
import os
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_jarvis_weather():
    """Test JARVIS weather functionality with vision"""
    print("🤖 Testing JARVIS Weather Vision Integration")
    print("=" * 60)
    
    # Import JARVIS components
    from voice.jarvis_agent_voice import JARVISAgentVoice
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzerMain
    from system_control.macos_controller import MacOSController
    
    # Initialize vision analyzer
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY not found in environment")
        return
    
    print("\n1. Initializing vision analyzer...")
    vision_analyzer = ClaudeVisionAnalyzerMain(api_key)
    
    # Initialize JARVIS with vision
    print("2. Initializing JARVIS with vision handler...")
    jarvis = JARVISAgentVoice(user_name="Sir", vision_analyzer=vision_analyzer)
    
    # Test weather queries
    test_queries = [
        "What's the weather today?",
        "What's today's weather?",
        "Tell me the weather",
        "How's the weather?",
        "What's the temperature?",
        "Is it going to rain today?",
        "What's the weather forecast?",
    ]
    
    print("\n3. Testing weather queries:")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📍 Test {i}: '{query}'")
        
        # Check if it's detected as weather command
        is_weather = jarvis._is_weather_command(query.lower())
        print(f"   Detected as weather: {is_weather}")
        
        if is_weather:
            # Test the weather handler
            try:
                print("   Executing weather handler...")
                response = await asyncio.wait_for(
                    jarvis._handle_weather_command(query),
                    timeout=25.0  # 25 seconds timeout
                )
                
                print(f"   Response: {response}")
                
                # Check for generic fallbacks
                generic_phrases = [
                    "I'll check the weather",
                    "For now, you can open",
                    "Let me open the Weather app",
                    "Please open the Weather app"
                ]
                
                has_generic = any(phrase in response for phrase in generic_phrases)
                if has_generic:
                    print("   ⚠️  WARNING: Response contains generic fallback!")
                else:
                    print("   ✅ Response uses actual weather data!")
                
                # Check for weather-specific content
                weather_indicators = ['°', 'degrees', 'temperature', 'cloudy', 'sunny', 'rain', 'forecast']
                has_weather_data = any(indicator in response for indicator in weather_indicators)
                
                if has_weather_data:
                    print("   ✅ Response contains weather-specific data")
                else:
                    print("   ❌ Response lacks weather-specific data")
                    
            except asyncio.TimeoutError:
                print("   ❌ TIMEOUT: Weather handler took too long")
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("   ❌ Not detected as weather command!")
    
    print("\n\n4. Testing direct vision weather workflow:")
    print("-" * 60)
    
    try:
        from workflows.weather_app_vision_unified import execute_weather_app_workflow
        
        controller = MacOSController()
        
        print("   Executing unified weather workflow...")
        workflow_response = await asyncio.wait_for(
            execute_weather_app_workflow(controller, vision_analyzer, "What's today's weather?"),
            timeout=20.0
        )
        
        print(f"   Workflow response: {workflow_response}")
        
    except Exception as e:
        print(f"   ❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\n✅ Weather vision integration test complete!")
    print("\nSummary:")
    print("- JARVIS should use vision to read Weather app")
    print("- No generic fallback responses should appear")
    print("- Responses should contain actual weather data (temperature, conditions)")


if __name__ == "__main__":
    asyncio.run(test_jarvis_weather())