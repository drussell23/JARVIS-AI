#!/usr/bin/env python3
"""
Test Simple Unlock Without Context
==================================

Tests if unlock works when bypassing context handler
"""

import asyncio
import json
import websockets
import traceback

async def test_simple_unlock():
    """Test unlock with simple command type instead of context"""
    print("\n🔧 Testing Simple Unlock Command")
    print("="*60)
    
    uri = 'ws://localhost:8000/voice/jarvis/stream'
    print(f"\n📡 Connecting to: {uri}")
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ WebSocket connected successfully")
            
            # Wait for welcome
            welcome = await ws.recv()
            welcome_data = json.loads(welcome)
            print(f"📨 Welcome: {welcome_data.get('message')}")
            
            # Send unlock with type 'simple' to bypass context
            # Looking at the code, type 'audio' goes through different path
            command = {
                "type": "audio",  # Try audio type to bypass command processing
                "data": "",  # Empty audio data
                "text": "unlock my screen"  # But provide text
            }
            
            await ws.send(json.dumps(command))
            print("✅ Command sent as audio type")
            
            # Wait for response
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(response)
                    print(f"\n📨 Response: {json.dumps(data, indent=2)}")
                    
                    if data.get('type') == 'response':
                        break
                        
            except asyncio.TimeoutError:
                print("⏱️  No response within 5 seconds")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🔓 Testing Simple Unlock")
    asyncio.run(test_simple_unlock())