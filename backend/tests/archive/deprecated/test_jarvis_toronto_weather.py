#!/usr/bin/env python3
"""Test JARVIS Toronto Weather"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

async def test_jarvis_toronto():
    """Test JARVIS weather response for Toronto"""
    print("🌆 Testing JARVIS Toronto Weather")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Test weather bridge
    print("\n🌉 Testing Weather Bridge:")
    from system_control.weather_bridge import WeatherBridge
    bridge = WeatherBridge()
    
    response = await bridge.process_weather_query("What's the weather today?")
    print(f"Response: {response}")
    
    # Check if it says Toronto now
    if "Toronto" in response:
        print("✅ SUCCESS: Now showing Toronto!")
    elif "Willowdale" in response:
        print("⚠️  Still showing Willowdale")
    elif "North York" in response:
        print("⚠️  Still showing North York")
    else:
        print(f"❓ Location not clear in response")
    
    # Test with JARVIS handler
    print("\n🤖 Testing JARVIS Handler:")
    from voice.jarvis_agent_voice import JARVISAgentVoice
    jarvis = JARVISAgentVoice()
    
    response = await jarvis._handle_weather_command("what's the weather today?")
    print(f"JARVIS says: {response}")
    
    if "Toronto" in response:
        print("✅ SUCCESS: JARVIS now says Toronto!")
    else:
        print(f"❓ Check location in response")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    asyncio.run(test_jarvis_toronto())