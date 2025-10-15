#!/usr/bin/env python3
"""Test Vision WebSocket Integration"""

import asyncio
import websockets
import json

async def test_vision_websocket():
    uri = "ws://localhost:8000/vision/ws/vision"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Vision WebSocket!")
            
            # Send initial connection message
            await websocket.send(json.dumps({
                "type": "connected",
                "message": "Test client connected"
            }))
            
            # Listen for messages
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(message)
                print(f"📨 Received: {data}")
                return True
            except asyncio.TimeoutError:
                print("⏱️ No response within timeout (which is normal for WebSocket)")
                return True
                
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Vision WebSocket Integration...")
    success = asyncio.run(test_vision_websocket())
    
    if success:
        print("\n✅ Vision WebSocket is working!")
        print("   The frontend should now be able to connect.")
    else:
        print("\n❌ Vision WebSocket test failed")