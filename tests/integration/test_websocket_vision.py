#!/usr/bin/env python3
"""Test WebSocket vision command processing"""

import asyncio
import websockets
import json
import time

async def test_vision_websocket():
    uri = "ws://localhost:8010/voice/jarvis/stream"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to JARVIS WebSocket")
            
            # Wait for connection message
            response = await websocket.recv()
            print(f"Connection response: {response}")
            
            # Send vision command
            command = {
                "type": "command",
                "text": "can you see my screen"
            }
            
            print(f"\nSending command: {command['text']}")
            start_time = time.time()
            
            await websocket.send(json.dumps(command))
            
            # Receive responses
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                elapsed = time.time() - start_time
                
                print(f"\n[{elapsed:.2f}s] Received: {data.get('type', 'unknown')}")
                
                if data.get('type') == 'processing':
                    print(f"Status: {data.get('message')}")
                elif data.get('type') == 'response':
                    print(f"Response: {data.get('text', '')[:200]}...")
                    break
                elif data.get('type') == 'error':
                    print(f"Error: {data.get('message')}")
                    break
                    
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    asyncio.run(test_vision_websocket())