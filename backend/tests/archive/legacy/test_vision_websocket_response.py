#!/usr/bin/env python3
"""
Test vision response through WebSocket
"""

import asyncio
import json
import websockets
import os
from dotenv import load_dotenv

load_dotenv()

async def test_websocket():
    uri = "ws://localhost:8010/voice/jarvis/stream"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        # Send a vision command
        message = {
            "type": "command",
            "text": "can you see my battery percentage?",
            "timestamp": "2025-09-19T16:40:00.000Z"
        }
        
        await websocket.send(json.dumps(message))
        print(f"Sent: {message}")
        
        # Listen for responses
        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                
                print(f"\n📊 Received message type: {data.get('type')}")
                
                if data.get('type') == 'response':
                    print(f"📊 Response text field: {data.get('text')}")
                    print(f"📊 Response text type: {type(data.get('text'))}")
                    
                    # Check if text is a string or dict
                    text = data.get('text')
                    if isinstance(text, dict):
                        print(f"⚠️  Text is a dict with keys: {text.keys()}")
                        if 'content' in text:
                            print(f"📊 Content field: {text['content']}")
                        if 'raw_result' in text:
                            print(f"📊 Raw result field present")
                    break
                    
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
                break

if __name__ == "__main__":
    asyncio.run(test_websocket())