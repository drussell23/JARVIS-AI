#!/usr/bin/env python3
"""
Test Direct Voice Unlock Integration
====================================

Tests the voice unlock functionality directly
"""

import asyncio
import aiohttp
import json


async def test_voice_unlock():
    """Test voice unlock functionality directly"""
    
    api_url = "http://localhost:8000/api/voice-unlock/command"
    
    print("🧪 Testing Voice Unlock Integration")
    print("=" * 50)
    
    # Test unlock command
    print("\n📝 Testing unlock command...")
    
    async with aiohttp.ClientSession() as session:
        command_data = {
            "command": "unlock_screen",
            "data": {
                "source": "test",
                "reason": "Testing unlock functionality",
                "authenticated": True
            }
        }
        
        try:
            async with session.post(api_url, json=command_data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"Response: {json.dumps(result, indent=2)}")
                else:
                    print(f"Error: HTTP {resp.status}")
                    text = await resp.text()
                    print(f"Response: {text}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")
    

async def test_screen_state():
    """Test screen state detection"""
    
    print("\n📝 Testing screen state detection...")
    
    # Try to import and test screen state directly
    try:
        from context_intelligence.core.system_state_monitor import get_system_monitor
        monitor = get_system_monitor()
        
        # Force initialize if needed
        if hasattr(monitor, 'initialize'):
            await monitor.initialize()
            
        # Get screen state
        is_locked = await monitor.get_state("screen_locked", force_refresh=True)
        print(f"Screen locked: {is_locked}")
        
    except Exception as e:
        print(f"Could not test screen state: {e}")
        
        # Try alternative method
        print("\nTrying alternative screen detection...")
        try:
            import subprocess
            result = subprocess.run(['python', '-c', '''
import Quartz
session_dict = Quartz.CGSessionCopyCurrentDictionary()
if session_dict:
    locked = session_dict.get("CGSSessionScreenIsLocked", False)
    print(f"Screen locked: {locked}")
else:
    print("Could not get session dictionary")
'''], capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Error: {result.stderr}")
        except Exception as e2:
            print(f"Alternative method failed: {e2}")


async def test_command_through_api():
    """Test a command through the normal API"""
    
    api_url = "http://localhost:8000/api/command"
    
    print("\n📝 Testing command through normal API...")
    
    async with aiohttp.ClientSession() as session:
        # Test a simple command
        command = "What time is it?"
        
        try:
            async with session.post(api_url, json={"command": command}) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"Command: {command}")
                    print(f"Response: {result.get('response', result)}")
                else:
                    print(f"Error: HTTP {resp.status}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("Testing Voice Unlock and Screen State Detection\n")
    
    # Run tests
    asyncio.run(test_voice_unlock())
    asyncio.run(test_screen_state())
    asyncio.run(test_command_through_api())