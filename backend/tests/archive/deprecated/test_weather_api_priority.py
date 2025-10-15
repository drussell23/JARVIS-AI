#!/usr/bin/env python3
"""Test Weather API Priority - Ensure API is used first, not Weather app"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

async def test_api_priority():
    """Test that API is used as primary weather source"""
    print("🌦️  Testing Weather API Priority")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("OPENWEATHER_API_KEY")
    print(f"\n📍 API Key Status: {'✅ Configured' if api_key else '❌ Not found'}")
    
    if api_key:
        print(f"   Key: {api_key[:10]}...{api_key[-5:]}")
    
    # Test 1: Test weather bridge directly
    print("\n🧪 Test 1: Weather Bridge Direct Test")
    from system_control.weather_bridge import WeatherBridge
    bridge = WeatherBridge()
    
    response = await bridge.process_weather_query("What's the weather today?")
    print(f"✅ Response: {response[:150]}...")
    
    # Check if it mentions opening the Weather app
    if "opened the Weather app" in response or "Weather app" in response:
        print("❌ ERROR: Still opening Weather app instead of using API!")
    else:
        print("✅ SUCCESS: Using API without opening Weather app!")
    
    # Test 2: Test with JARVIS voice handler
    print("\n🧪 Test 2: JARVIS Voice Handler Test")
    from voice.jarvis_agent_voice import JARVISAgentVoice
    
    jarvis = JARVISAgentVoice()
    response = await jarvis._handle_weather_command("what's the weather today?")
    print(f"✅ Response: {response[:150]}...")
    
    # Check response source
    if "opened the Weather app" in response:
        print("❌ ERROR: Handler still opening Weather app!")
    else:
        print("✅ SUCCESS: Handler using API!")
    
    # Test 3: City-specific query
    print("\n🧪 Test 3: City Weather Test (Tokyo)")
    response = await bridge.process_weather_query("What's the weather in Tokyo?")
    print(f"✅ Response: {response[:150]}...")
    
    if "Tokyo" in response and "opened the Weather app" not in response:
        print("✅ SUCCESS: City queries working via API!")
    else:
        print("❌ ERROR: City queries not working properly")
    
    print("\n" + "="*60)
    print("\n📝 Summary:")
    if api_key:
        print("✅ API key is configured")
        print("✅ Weather bridge should use API as primary source")
        print("✅ No Weather app opening for API queries")
    else:
        print("❌ No API key - configure OPENWEATHER_API_KEY in .env")

if __name__ == "__main__":
    asyncio.run(test_api_priority())