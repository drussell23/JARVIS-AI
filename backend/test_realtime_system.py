#!/usr/bin/env python3
"""Test real-time system integration"""

import asyncio
import sys
sys.path.append('.')

from system_control.macos_system_integration import MacOSSystemIntegration
from system_control.weather_bridge import WeatherBridge
from voice.jarvis_agent_voice import JARVISAgentVoice

async def test_system():
    print("🔧 Testing Real-Time System Integration\n")
    
    # Test 1: System Time
    print("1️⃣ SYSTEM TIME TEST:")
    system = MacOSSystemIntegration()
    time_data = await system.get_system_time()
    print(f"   System Time: {time_data}")
    
    # Test 2: Weather Widget
    print("\n2️⃣ WEATHER WIDGET TEST:")
    weather_data = await system.get_weather_from_widget()
    print(f"   Weather Widget: {weather_data}")
    
    # Test 3: System Location
    print("\n3️⃣ LOCATION TEST:")
    location = await system.get_system_location()
    print(f"   Location: {location}")
    
    # Test 4: Accurate Weather
    print("\n4️⃣ ACCURATE WEATHER TEST:")
    accurate_weather = await system.get_accurate_weather()
    print(f"   Accurate Weather: {accurate_weather}")
    
    # Test 5: Weather Bridge
    print("\n5️⃣ WEATHER BRIDGE TEST:")
    bridge = WeatherBridge()
    current_weather = await bridge.get_current_weather(use_cache=False)
    print(f"   Current Weather: {current_weather}")
    
    # Test 6: Process Query
    print("\n6️⃣ QUERY PROCESSING TEST:")
    query = "what's the weather for today"
    response = await bridge.process_weather_query(query)
    print(f"   Query: '{query}'")
    print(f"   Response: {response}")
    
    # Test 7: Time Query
    print("\n7️⃣ TIME QUERY TEST:")
    jarvis = JARVISAgentVoice()
    time_response = await jarvis._handle_time_command("what time is it")
    print(f"   Time Response: {time_response}")
    
    print("\n✅ All tests complete!")

if __name__ == "__main__":
    asyncio.run(test_system())