#!/usr/bin/env python3
"""
Test Lock Command - Automated
==============================

Tests lock command without user input.
"""

import asyncio
import aiohttp
import json

async def test_lock():
    """Test lock command"""
    print("🔐 Testing Lock Command Through JARVIS")
    print("=" * 50)
    
    url = "http://localhost:8000/api/command"
    
    try:
        async with aiohttp.ClientSession() as session:
            # First test simple command
            print("\n1️⃣ Testing simple command...")
            data = {"command": "what time is it"}
            
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Simple command works: {result.get('response', '')[:50]}...")
                else:
                    print(f"❌ Got status {response.status}")
                    return
            
            # Now test lock command
            print("\n2️⃣ Testing lock command...")
            data = {"command": "lock my screen"}
            
            async with session.post(url, json=data) as response:
                print(f"Status: {response.status}")
                text = await response.text()
                print(f"Raw response: {text}")
                
                try:
                    result = json.loads(text)
                    print(f"\nParsed response:")
                    print(f"  Success: {result.get('success')}")
                    print(f"  Response: {result.get('response')}")
                    print(f"  Error: {result.get('error')}")
                    
                    if result.get('success'):
                        print("\n✅ Lock command processed!")
                    else:
                        print("\n❌ Lock command failed")
                except:
                    print("Could not parse response as JSON")
                    
    except aiohttp.ClientConnectorError:
        print("\n❌ Could not connect to JARVIS")
        print("Make sure JARVIS is running on port 8000")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_lock())