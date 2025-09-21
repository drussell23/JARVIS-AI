#!/usr/bin/env python3
"""Test weather data validation"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def test_validation():
    """Test the weather data validation"""
    from system_control.weather_bridge import WeatherBridge
    from system_control.macos_weather_provider import MacOSWeatherProvider
    
    # Create instances
    weather_bridge = WeatherBridge()
    provider = MacOSWeatherProvider()
    
    print("Testing weather data validation...\n")
    
    # Get weather data
    weather_data = await provider.get_weather_data()
    print(f"Raw weather data: {weather_data}\n")
    
    # Test validation
    is_valid = weather_bridge._is_valid_weather_data(weather_data)
    print(f"Is valid: {is_valid}\n")
    
    # Check individual conditions
    print("Validation checks:")
    print(f"  - Has temperature: {weather_data.get('temperature') is not None}")
    print(f"  - Not error: {not weather_data.get('error', False)}")
    print(f"  - Not unavailable: {weather_data.get('source') != 'unavailable'}")
    print(f"  - Location: '{weather_data.get('location')}' (valid: {bool(weather_data.get('location') and weather_data['location'] != 'your location')})")
    print(f"  - Condition: '{weather_data.get('condition')}' (valid: {bool(weather_data.get('condition') and weather_data['condition'] not in ['unknown', 'unavailable'])})")
    print(f"  - Description: '{weather_data.get('description')}' (valid: {bool(weather_data.get('description') and weather_data['description'] not in ['unknown', 'unavailable', 'Weather data temporarily unavailable'])})")

if __name__ == "__main__":
    asyncio.run(test_validation())