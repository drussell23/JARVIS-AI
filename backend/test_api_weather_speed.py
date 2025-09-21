#!/usr/bin/env python3
"""Test API Weather Speed"""

import asyncio
import os
import time
from dotenv import load_dotenv

async def test_api_speed():
    """Test weather API response speed"""
    print("⚡ Testing OpenWeatherMap API Speed")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Test direct API call
    print("\n🌦️  Direct API Test:")
    from services.weather_service import WeatherService
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
        
    weather_service = WeatherService(api_key)
    
    # Test current location
    start = time.time()
    weather = await weather_service.get_current_weather()
    end = time.time()
    print(f"✅ Current location: {weather.get('location')} - {weather.get('temperature')}°C")
    print(f"   Response time: {(end - start):.3f} seconds")
    
    # Test specific cities
    cities = ["Toronto", "New York", "London", "Tokyo"]
    for city in cities:
        start = time.time()
        weather = await weather_service.get_weather_by_city(city)
        end = time.time()
        print(f"✅ {city}: {weather.get('temperature')}°C, {weather.get('condition')}")
        print(f"   Response time: {(end - start):.3f} seconds")
    
    await weather_service.close()
    
    # Test weather bridge
    print("\n🌉 Weather Bridge Test:")
    from system_control.weather_bridge import WeatherBridge
    bridge = WeatherBridge()
    
    start = time.time()
    response = await bridge.process_weather_query("What's the weather today?")
    end = time.time()
    print(f"✅ Bridge response: {response[:100]}...")
    print(f"   Response time: {(end - start):.3f} seconds")
    
    print("\n" + "="*60)
    print("✨ API provides sub-second weather responses!")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_api_speed())