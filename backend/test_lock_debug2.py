#!/usr/bin/env python3
"""Debug lock command response flow"""

import asyncio
import websockets
import json

async def test():
    uri = 'ws://localhost:8000/voice/jarvis/stream'
    async with websockets.connect(uri) as ws:
        # Wait for connection
        await ws.recv()
        
        # Send lock command
        await ws.send(json.dumps({
            "type": "command",
            "text": "lock my screen"
        }))
        
        # Print ALL messages received
        print("All messages received:")
        for i in range(10):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                data = json.loads(msg)
                print(f"\nMessage {i+1}:")
                print(json.dumps(data, indent=2))
            except asyncio.TimeoutError:
                break
                
asyncio.run(test())