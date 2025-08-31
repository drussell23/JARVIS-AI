#!/usr/bin/env python3
"""
Test JARVIS vision through WebSocket connection
"""

import asyncio
import websockets
import json

async def test_vision_websocket():
    """Test vision command through WebSocket like JARVIS would"""
    uri = "ws://localhost:8001/ws/vision"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Send vision command
            message = {
                "type": "vision_command",
                "command": "can you see my screen"
            }
            
            print(f"\n📤 Sending: {json.dumps(message, indent=2)}")
            await websocket.send(json.dumps(message))
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            
            print(f"\n📥 Received: {json.dumps(response_data, indent=2)}")
            
            if response_data.get("type") == "error":
                print(f"\n❌ Error: {response_data.get('message')}")
            else:
                print(f"\n✅ Success!")
                if response_data.get("result"):
                    print(f"Result: {response_data['result'][:200]}...")
                    
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        print("\nMake sure the WebSocket router is running:")
        print("  cd backend/websocket && npm start")

async def main():
    print("🔍 Testing JARVIS Vision via WebSocket")
    print("=" * 50)
    await test_vision_websocket()

if __name__ == "__main__":
    asyncio.run(main())