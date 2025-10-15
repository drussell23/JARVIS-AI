#!/usr/bin/env python3
"""
Test Vision Weather Integration
Verifies the complete weather vision pipeline works correctly
"""

import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

async def test_vision_weather():
    """Test the vision weather integration"""
    print("🌤️ Testing Vision Weather Integration")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY not found")
        return
    
    print("\n1. Testing Vision Analyzer Weather Methods:")
    print("-" * 40)
    
    # Test vision analyzer
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzerMain
    
    analyzer = ClaudeVisionAnalyzerMain(api_key)
    
    # Test weather query method
    print("   Testing query_screen_for_weather()...")
    weather_result = await analyzer.query_screen_for_weather()
    
    if weather_result:
        print(f"   ✅ Weather result: {weather_result[:100]}...")
    else:
        print("   ⚠️  No weather data returned")
    
    # Test describe_screen with weather query
    print("\n   Testing describe_screen() with weather query...")
    result = await analyzer.describe_screen({
        'query': 'Read the Weather app and tell me the current temperature and conditions'
    })
    
    if result['success']:
        print(f"   ✅ Success: {result['description'][:100]}...")
    else:
        print(f"   ❌ Failed: {result}")
    
    print("\n2. Testing Unified Weather System:")
    print("-" * 40)
    
    # Test unified weather system
    from system_control.unified_vision_weather import UnifiedVisionWeather
    from system_control.macos_controller import MacOSController
    
    controller = MacOSController()
    weather_system = UnifiedVisionWeather(analyzer, controller)
    
    test_queries = [
        "What's the weather?",
        "What's the temperature?",
        "Is it going to rain?"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        result = await weather_system.get_weather(query)
        
        if result['success']:
            print(f"   ✅ Response: {result['formatted_response'][:100]}...")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n3. Testing Weather Workflow:")
    print("-" * 40)
    
    # Test workflow
    from workflows.weather_app_vision_unified import execute_weather_app_workflow
    
    workflow_result = await execute_weather_app_workflow(
        controller, 
        analyzer,
        "What's today's weather forecast?"
    )
    
    print(f"   Workflow result: {workflow_result[:150]}...")
    
    # Check for generic responses
    generic_phrases = ["I'll check", "Let me open", "Please open"]
    has_generic = any(phrase in workflow_result for phrase in generic_phrases)
    
    if has_generic:
        print("   ⚠️  WARNING: Response contains generic phrases!")
    else:
        print("   ✅ Response appears to contain actual weather data")
    
    print("\n✅ Vision weather integration test complete!")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_vision_weather())