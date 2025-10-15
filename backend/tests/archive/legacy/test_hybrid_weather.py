#!/usr/bin/env python3
"""Test Hybrid Weather Provider"""

import asyncio
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_hybrid_weather():
    """Test the hybrid weather provider"""
    print("🌦️  Testing Hybrid Weather Provider")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("OPENWEATHER_API_KEY")
    print(f"\n📍 API Key Available: {'✅' if api_key else '❌'}")
    
    # Initialize weather service
    from services.weather_service import WeatherService
    weather_service = WeatherService(api_key) if api_key else None
    
    # Initialize vision extractor
    vision_extractor = None
    try:
        from system_control.vision_weather_extractor import VisionWeatherExtractor
        vision_extractor = VisionWeatherExtractor()
        print("📍 Vision Extractor: ✅")
    except Exception as e:
        print(f"📍 Vision Extractor: ❌ ({e})")
    
    # Initialize hybrid provider
    from system_control.hybrid_weather_provider import HybridWeatherProvider
    hybrid = HybridWeatherProvider(
        api_weather_service=weather_service,
        vision_extractor=vision_extractor
    )
    
    print("\n" + "-"*60)
    
    # Test 1: Current location weather
    print("\n🧪 Test 1: Current Location Weather")
    try:
        weather = await hybrid.get_current_weather()
        if weather.get('error'):
            print(f"❌ Error: {weather.get('message')}")
        else:
            print(f"✅ Location: {weather.get('location')}")
            print(f"✅ Temperature: {weather.get('temperature')}°C")
            print(f"✅ Condition: {weather.get('condition')}")
            print(f"✅ Source: {weather.get('source')}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: Specific city weather
    print("\n🧪 Test 2: Specific City Weather (New York)")
    try:
        weather = await hybrid.get_weather_by_city("New York")
        if weather.get('error'):
            print(f"❌ Error: {weather.get('message')}")
        else:
            print(f"✅ Location: {weather.get('location')}")
            print(f"✅ Temperature: {weather.get('temperature')}°C")
            print(f"✅ Condition: {weather.get('condition')}")
            print(f"✅ Source: {weather.get('source')}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: Test with weather bridge
    print("\n🧪 Test 3: Weather Bridge Integration")
    try:
        from system_control.weather_bridge import WeatherBridge
        bridge = WeatherBridge()
        
        # Test weather query processing
        response = await bridge.process_weather_query("What's the weather for today?")
        print(f"✅ JARVIS Response: {response}")
        
    except Exception as e:
        print(f"❌ Bridge test failed: {e}")
    
    # Clean up
    if weather_service:
        await weather_service.close()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_hybrid_weather())