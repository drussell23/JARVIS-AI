#!/usr/bin/env python3
"""Test Tokyo Weather Query"""

import asyncio
import os
from dotenv import load_dotenv

async def test_tokyo():
    """Test Tokyo weather specifically"""
    print("🗼 Testing Tokyo Weather Query")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    from system_control.weather_bridge import WeatherBridge
    bridge = WeatherBridge()
    
    # Test location extraction
    query = "What's the weather in Tokyo?"
    location = bridge.extract_location_from_query(query)
    print(f"\n📍 Extracted location: {location}")
    
    # Test process_weather_query
    print("\n🌦️  Full query processing:")
    response = await bridge.process_weather_query(query)
    print(f"Response: {response}")
    
    # Test get_weather_by_city directly
    print("\n🏙️  Direct city query:")
    weather = await bridge.get_weather_by_city("Tokyo")
    print(f"Location: {weather.get('location')}")
    print(f"Temperature: {weather.get('temperature')}°C")
    print(f"Source: {weather.get('source')}")
    
    # Test formatted response
    formatted = bridge.format_for_speech(weather, "current")
    print(f"\nFormatted: {formatted}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_tokyo())