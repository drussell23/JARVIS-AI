#!/usr/bin/env python3
"""Test app open/close commands through JARVIS"""

import asyncio
import websockets
import json
import time

async def test_app_commands():
    """Test opening and closing apps through JARVIS"""
    print("🎯 Testing App Commands through JARVIS")
    print("=" * 50)
    
    uri = 'ws://localhost:8000/voice/jarvis/stream'
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected to JARVIS WebSocket")
            
            # Wait for connection message
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"Connection: {data.get('message', 'Connected')}")
            
            # Test commands
            test_apps = [
                ("open safari", "Safari"),
                ("close safari", "Safari"),
                ("open music", "Music"),  
                ("close music", "Music"),
                ("open weather", "Weather"),
                ("close weather", "Weather")
            ]
            
            for command, app_name in test_apps:
                print(f"\n📍 Testing: '{command}'")
                await ws.send(json.dumps({
                    "type": "command",
                    "text": command
                }))
                
                # Collect responses
                responses = []
                start_time = time.time()
                while time.time() - start_time < 3:  # Wait up to 3 seconds
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(response)
                        responses.append(data)
                        
                        if data.get('type') == 'response':
                            print(f"  Response: {data.get('text', 'No text')}")
                            print(f"  Success: {data.get('success', 'Unknown')}")
                            
                            # Check if it includes error info
                            if 'error' in data:
                                print(f"  Error: {data['error']}")
                                
                    except asyncio.TimeoutError:
                        break
                
                if not any(r.get('type') == 'response' for r in responses):
                    print(f"  ❌ No response received for {command}")
                    
                await asyncio.sleep(1)  # Brief pause between commands
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("  1. JARVIS is running on port 8000")
        print("  2. WebSocket endpoint is available")


if __name__ == "__main__":
    print("Testing JARVIS app open/close commands...")
    print("This will try to open and close Safari, Music, and Weather apps\n")
    
    asyncio.run(test_app_commands())