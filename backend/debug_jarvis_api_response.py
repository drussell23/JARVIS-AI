#!/usr/bin/env python3
"""
Debug JARVIS API Response
=========================

Figure out why the API returns None values.
"""

import asyncio
import aiohttp
import json

async def test_jarvis_api():
    """Test JARVIS API response format"""
    print("🔍 Debugging JARVIS API Response")
    print("=" * 50)
    
    url = "http://localhost:8000/api/voice-command"
    headers = {'Content-Type': 'application/json'}
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test command
            data = {
                "text": "lock my screen",
                "audio_data": None
            }
            
            print(f"\nSending POST to {url}")
            print(f"Data: {json.dumps(data, indent=2)}")
            
            async with session.post(url, json=data, headers=headers) as response:
                print(f"\nHTTP Status: {response.status}")
                print(f"Content-Type: {response.headers.get('Content-Type')}")
                
                # Get raw text first
                text = await response.text()
                print(f"\nRaw Response:\n{text}")
                
                # Try to parse as JSON
                try:
                    result = json.loads(text)
                    print(f"\nParsed JSON:")
                    print(json.dumps(result, indent=2))
                except:
                    print("\nCould not parse as JSON")
                    
    except aiohttp.ClientConnectorError:
        print("\n❌ Could not connect to JARVIS on port 8000")
        print("Make sure JARVIS is running: python main.py")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")

# Also test a simple command
async def test_simple_command():
    """Test with a simple command"""
    print("\n\n📝 Testing Simple Command")
    print("=" * 50)
    
    url = "http://localhost:8000/api/voice-command"
    headers = {'Content-Type': 'application/json'}
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test command
            data = {
                "text": "what time is it",
                "audio_data": None
            }
            
            print(f"\nSending: '{data['text']}'")
            
            async with session.post(url, json=data, headers=headers) as response:
                text = await response.text()
                print(f"\nRaw Response:\n{text}")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")

async def main():
    await test_jarvis_api()
    await test_simple_command()

if __name__ == "__main__":
    asyncio.run(main())