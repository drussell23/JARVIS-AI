#!/usr/bin/env python3
"""Test the weather system to ensure it works correctly with Toronto location"""

import asyncio
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_weather_system():
    """Test the weather system"""
    print("=== Testing JARVIS Weather System ===\n")
    
    try:
        # Import the weather bridge
        from system_control.weather_bridge import WeatherBridge
        
        # Create weather bridge instance
        weather_bridge = WeatherBridge()
        
        # Test 1: Check if query is recognized as weather
        test_queries = [
            "what's the weather for today",
            "weather today",
            "what's the temperature",
            "is it raining",
            "how cold is it outside"
        ]
        
        print("1. Testing weather query recognition:")
        for query in test_queries:
            is_weather = weather_bridge.is_weather_query(query)
            print(f"   '{query}' -> Weather query: {is_weather}")
        print()
        
        # Test 2: Get current weather
        print("2. Getting current weather data:")
        weather_data = await weather_bridge.get_current_weather()
        print(f"   Location: {weather_data.get('location', 'Unknown')}")
        print(f"   Temperature: {weather_data.get('temperature', 'N/A')}°C")
        print(f"   Condition: {weather_data.get('condition', 'Unknown')}")
        print(f"   Description: {weather_data.get('description', 'Unknown')}")
        print(f"   Source: {weather_data.get('source', 'Unknown')}")
        print()
        
        # Test 3: Process weather queries
        print("3. Processing weather queries:")
        for query in test_queries[:3]:
            response = await weather_bridge.process_weather_query(query)
            print(f"   Query: '{query}'")
            print(f"   Response: {response}")
            print()
        
        # Test 4: Check location detection
        print("4. Testing location detection:")
        from system_control.macos_system_integration import MacOSSystemIntegration
        system_integration = MacOSSystemIntegration()
        location_data = await system_integration.get_system_location()
        if location_data:
            print(f"   City: {location_data.get('city', 'Unknown')}")
            print(f"   Region: {location_data.get('region', 'Unknown')}")
            print(f"   Latitude: {location_data.get('latitude', 'Unknown')}")
            print(f"   Longitude: {location_data.get('longitude', 'Unknown')}")
            print(f"   Source: {location_data.get('source', 'Unknown')}")
        else:
            print("   Could not get location data")
        print()
        
        # Test 5: Widget extraction
        print("5. Testing widget extraction:")
        from system_control.weather_widget_extractor import WeatherWidgetExtractor
        widget_extractor = WeatherWidgetExtractor()
        widget_data = await widget_extractor.extract_weather_data()
        if widget_data:
            print(f"   Temperature: {widget_data.get('temperature', 'N/A')}°C")
            print(f"   Location: {widget_data.get('location', 'Unknown')}")
            print(f"   Condition: {widget_data.get('condition', 'Unknown')}")
            print(f"   Source: {widget_data.get('source', 'Unknown')}")
        else:
            print("   Could not extract widget data")
        
        print("\n=== Test Complete ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_weather_system())