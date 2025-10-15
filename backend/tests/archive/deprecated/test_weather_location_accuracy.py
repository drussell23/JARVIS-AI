#!/usr/bin/env python3
"""Test OpenWeatherMap Location Detection Accuracy"""

import asyncio
import os
import logging
from dotenv import load_dotenv
import geocoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_location_accuracy():
    """Test the accuracy of location detection methods"""
    print("📍 Testing Location Detection Accuracy")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Method 1: IP-based geolocation (used by weather service)
    print("\n🌐 IP-Based Geolocation:")
    try:
        g = geocoder.ip('me')
        if g.ok:
            print(f"   Location: {g.city}, {g.state}, {g.country}")
            print(f"   Coordinates: {g.latlng[0]}, {g.latlng[1]}")
            print(f"   Timezone: {g.timezone}")
            print(f"   ISP: {g.org}")
        else:
            print("   ❌ Failed to get IP location")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 2: Test with weather service
    print("\n🌦️  Weather Service Location Detection:")
    try:
        from services.weather_service import WeatherService
        
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if api_key:
            weather_service = WeatherService(api_key)
            lat, lon, city = await weather_service.get_current_location()
            print(f"   ✅ Weather service detected: {city}")
            print(f"   ✅ Coordinates: {lat}, {lon}")
            
            # Get weather for detected location
            weather = await weather_service.get_current_weather()
            print(f"\n   Weather in {weather.get('location')}:")
            print(f"   Temperature: {weather.get('temperature')}°C")
            print(f"   Condition: {weather.get('condition')}")
            
            await weather_service.close()
        else:
            print("   ❌ No API key configured")
            
            # Test without API
            weather_service = WeatherService(None)
            lat, lon, city = await weather_service.get_current_location()
            print(f"   ℹ️  Fallback location: {city}")
            print(f"   ℹ️  Coordinates: {lat}, {lon}")
            
    except Exception as e:
        print(f"   ❌ Weather service error: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 3: Test weather bridge
    print("\n🌉 Weather Bridge Test:")
    try:
        from system_control.weather_bridge import WeatherBridge
        bridge = WeatherBridge()
        
        # Test current location weather
        weather = await bridge.get_current_weather()
        print(f"   ✅ Bridge detected location: {weather.get('location')}")
        print(f"   ✅ Source: {weather.get('source')}")
        print(f"   ✅ Temperature: {weather.get('temperature')}°C")
        
    except Exception as e:
        print(f"   ❌ Bridge error: {e}")
    
    # Method 4: Test with different cities
    print("\n🏙️  City-Specific Weather Tests:")
    test_cities = ["Toronto", "New York", "London", "Tokyo", "Sydney"]
    
    if api_key:
        weather_service = WeatherService(api_key)
        for city in test_cities:
            try:
                weather = await weather_service.get_weather_by_city(city)
                if not weather.get('error'):
                    print(f"   ✅ {city}: {weather.get('temperature')}°C, {weather.get('condition')}")
                else:
                    print(f"   ❌ {city}: Failed to get weather")
            except Exception as e:
                print(f"   ❌ {city}: Error - {e}")
        
        await weather_service.close()
    else:
        print("   ⚠️  Skipping city tests - no API key")
    
    print("\n" + "="*60)
    print("\n📝 Summary:")
    print("- IP geolocation provides approximate location")
    print("- OpenWeatherMap API needed for accurate weather data")
    print("- Vision fallback works but limited to displayed location")
    print("- Hybrid system ensures weather is always available")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_location_accuracy())