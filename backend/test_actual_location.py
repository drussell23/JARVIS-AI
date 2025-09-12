#!/usr/bin/env python3
"""Test actual OpenWeatherMap location detection"""

import asyncio
import os
import json
from dotenv import load_dotenv

async def test_location():
    """Test what OpenWeatherMap actually returns for Toronto coordinates"""
    print("🌍 Testing OpenWeatherMap Location Detection")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    from services.weather_service import WeatherService
    import aiohttp
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
    
    # Get coordinates for Toronto area
    weather_service = WeatherService(api_key)
    lat, lon, detected_city = await weather_service.get_current_location()
    
    print(f"\n📍 IP Geolocation:")
    print(f"   Detected city: {detected_city}")
    print(f"   Coordinates: {lat}, {lon}")
    
    # Make direct API call to see raw response
    print(f"\n🔍 OpenWeatherMap Raw Response:")
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                raw_data = await response.json()
                print(f"   City name from API: {raw_data.get('name')}")
                print(f"   Country: {raw_data.get('sys', {}).get('country')}")
                print(f"   Coordinates in response: {raw_data.get('coord')}")
                
                # Pretty print full response
                print("\n📋 Full API Response:")
                print(json.dumps(raw_data, indent=2))
    
    # Test through weather service
    print(f"\n🌦️  Weather Service Result:")
    weather = await weather_service.get_current_weather()
    print(f"   Location: {weather.get('location')}")
    print(f"   Detected location: {weather.get('detected_location')}")
    print(f"   Temperature: {weather.get('temperature')}°C")
    
    await weather_service.close()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_location())