"""
Test the Unified Weather System
Ensures the consolidated vision-based weather system works correctly
"""

import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test without vision handler first (dry run)
from system_control.unified_vision_weather import UnifiedVisionWeather
from system_control.weather_bridge_unified import UnifiedWeatherBridge


async def test_unified_weather_system():
    """Test the unified weather system with various queries"""
    print("🌤️  Testing Unified Weather System")
    print("=" * 60)
    
    # Create weather system without handlers (for testing logic)
    weather = UnifiedVisionWeather()
    
    # Test queries
    test_queries = [
        ("What's the weather today?", "Current weather"),
        ("Will it rain tomorrow?", "Tomorrow's precipitation"),
        ("What's the temperature?", "Current temperature"),
        ("How's the weather this week?", "Weekly forecast"),
        ("Is it windy outside?", "Wind conditions"),
        ("What's the weather in Toronto?", "Specific location"),
        ("Do I need an umbrella?", "Precipitation check"),
        ("What's the UV index?", "UV information"),
        ("When is sunset?", "Sunset time"),
        ("", "Default query (empty)")
    ]
    
    print("\n📝 Testing Query Intent Analysis:")
    print("-" * 60)
    
    for query, description in test_queries:
        print(f"\n🔍 {description}")
        print(f"   Query: '{query}'")
        
        # Test intent analysis
        intent = weather._analyze_query_intent(query)
        print(f"   Intent: {intent}")
        
        # Test without actual vision (will fail gracefully)
        result = await weather.get_weather(query)
        print(f"   Success: {result.get('success', False)}")
        if result.get('formatted_response'):
            print(f"   Response preview: {result['formatted_response'][:100]}...")


async def test_weather_bridge():
    """Test the unified weather bridge"""
    print("\n\n🌉 Testing Weather Bridge:")
    print("=" * 60)
    
    bridge = UnifiedWeatherBridge()
    
    # Test convenience methods
    print("\n📍 Testing convenience methods:")
    
    # Current weather
    print("\n1. Current Weather:")
    result = await bridge.get_current_weather()
    print(f"   Success: {result.get('success', False)}")
    
    # City weather
    print("\n2. City Weather (Toronto):")
    result = await bridge.get_weather_by_city("Toronto")
    print(f"   Success: {result.get('success', False)}")
    
    # Forecast
    print("\n3. 7-day Forecast:")
    result = await bridge.get_forecast(7)
    print(f"   Success: {result.get('success', False)}")
    
    # Precipitation
    print("\n4. Precipitation Check:")
    result = await bridge.check_precipitation()
    print(f"   Success: {result.get('success', False)}")
    
    # Query detection
    print("\n5. Query Detection:")
    test_phrases = [
        "What's the weather?",
        "Play some music",
        "Is it raining?",
        "Open Safari",
        "Temperature outside",
        "Show me photos"
    ]
    
    for phrase in test_phrases:
        is_weather = bridge.is_weather_query(phrase)
        print(f"   '{phrase}' -> Weather query: {is_weather}")


async def test_response_formatting():
    """Test response formatting with mock data"""
    print("\n\n📝 Testing Response Formatting:")
    print("=" * 60)
    
    weather = UnifiedVisionWeather()
    
    # Mock weather data
    mock_data = {
        'location': 'Toronto',
        'current': {
            'temperature': 66,
            'condition': 'Partly Cloudy'
        },
        'today': {
            'high': 73,
            'low': 56
        },
        'details': {
            'wind': '4',
            'humidity': '45',
            'uv_index': '3',
            'feels_like': '68',
            'sunrise': '6:45 am',
            'sunset': '7:32 pm'
        },
        'hourly': [
            {'time': '1pm', 'temperature': 68, 'condition': 'Partly Cloudy'},
            {'time': '2pm', 'temperature': 70, 'condition': 'Partly Cloudy'},
            {'time': '3pm', 'temperature': 73, 'condition': 'Clear'},
        ],
        'daily': [
            {'day': 'Today', 'high': 73, 'low': 56},
            {'day': 'Tomorrow', 'high': 75, 'low': 58},
            {'day': 'Wednesday', 'high': 71, 'low': 55},
        ]
    }
    
    # Test different intents
    test_intents = [
        {'type': 'current', 'timeframe': 'now', 'details_requested': []},
        {'type': 'current', 'timeframe': 'today', 'details_requested': ['wind', 'humidity']},
        {'type': 'forecast', 'timeframe': 'week', 'details_requested': []},
        {'type': 'hourly', 'timeframe': 'hourly', 'details_requested': []},
    ]
    
    for i, intent in enumerate(test_intents, 1):
        print(f"\n{i}. Intent: {intent['timeframe']}")
        response = weather._format_response_by_intent(mock_data, intent)
        print(f"   Response: {response}")


async def test_performance():
    """Test caching and performance"""
    print("\n\n⚡ Testing Performance & Caching:")
    print("=" * 60)
    
    weather = UnifiedVisionWeather()
    
    # Test cache
    query = "What's the weather?"
    
    print(f"\n1. First query (no cache):")
    start = datetime.now()
    result1 = await weather.get_weather(query)
    duration1 = (datetime.now() - start).total_seconds()
    print(f"   Duration: {duration1:.3f}s")
    print(f"   Cache key: {weather._get_cache_key(query)}")
    
    print(f"\n2. Second query (should use cache):")
    start = datetime.now()
    result2 = await weather.get_weather(query)
    duration2 = (datetime.now() - start).total_seconds()
    print(f"   Duration: {duration2:.3f}s")
    print(f"   Used cache: {duration2 < duration1}")


async def main():
    """Run all tests"""
    print("🚀 Starting Unified Weather System Tests")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n")
    
    # Run tests
    await test_unified_weather_system()
    await test_weather_bridge()
    await test_response_formatting()
    await test_performance()
    
    print("\n\n✅ All tests completed!")
    print("\nNote: These tests run without actual vision/controller handlers.")
    print("In production, the system will:")
    print("  1. Open the Weather app")
    print("  2. Use Claude Vision to read the screen")
    print("  3. Extract and format weather data")
    print("  4. Provide natural language responses")


if __name__ == "__main__":
    asyncio.run(main())