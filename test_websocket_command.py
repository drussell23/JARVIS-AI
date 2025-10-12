#!/usr/bin/env python3
"""Test WebSocket command to JARVIS"""

import asyncio
import websockets
import json

async def test_command():
    uri = "ws://localhost:8000/ws"

    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as websocket:
        print("✅ Connected!")

        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Welcome: {welcome}")

        # Send command
        command = {
            "type": "command",
            "text": "What's happening across my desktop spaces?"
        }

        print(f"\n📤 Sending: {json.dumps(command, indent=2)}")
        await websocket.send(json.dumps(command))

        # Receive response
        response = await websocket.recv()
        print(f"\n📥 Response: {response}")

        # Parse and display
        response_data = json.loads(response)
        print(f"\n✅ Response type: {response_data.get('type')}")
        if 'response' in response_data:
            print(f"📝 Response text:\n{response_data['response']}")
        elif 'text' in response_data:
            print(f"📝 Response text:\n{response_data['text']}")
        else:
            print(f"📝 Full response:\n{json.dumps(response_data, indent=2)}")

if __name__ == "__main__":
    try:
        asyncio.run(test_command())
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
