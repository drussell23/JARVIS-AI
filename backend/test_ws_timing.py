#!/usr/bin/env python3
"""Test WebSocket timing for app commands"""

import asyncio
import websockets
import json
import time

async def test_single_command(command: str):
    """Test a single command and measure timing"""
    uri = 'ws://localhost:8000/voice/jarvis/stream'
    
    try:
        async with websockets.connect(uri) as ws:
            # Wait for connection
            conn_msg = await ws.recv()
            print(f"Connected: {json.loads(conn_msg).get('message')[:50]}...")
            
            # Send command
            print(f"\nSending: '{command}'")
            send_time = time.time()
            await ws.send(json.dumps({
                "type": "command",
                "text": command
            }))
            
            # Collect ALL messages for up to 5 seconds
            messages = []
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    recv_time = time.time()
                    data = json.loads(msg)
                    elapsed = recv_time - send_time
                    
                    messages.append({
                        'time': elapsed,
                        'type': data.get('type'),
                        'text': data.get('text', data.get('message', '')),
                        'data': data
                    })
                    
                    print(f"  [{elapsed:.3f}s] Type: {data.get('type')}, Text: {data.get('text', data.get('message', ''))[:100]}")
                    
            except asyncio.TimeoutError:
                pass
            
            # Analyze results
            response_msgs = [m for m in messages if m['type'] == 'response']
            if response_msgs:
                print(f"\nGot {len(response_msgs)} response(s):")
                for r in response_msgs:
                    print(f"  - {r['text']} (at {r['time']:.3f}s)")
            else:
                print(f"\n❌ NO RESPONSE RECEIVED")
                print(f"Got {len(messages)} other messages:")
                for m in messages:
                    print(f"  - Type: {m['type']}")
                    
    except Exception as e:
        print(f"Error: {e}")

async def main():
    """Test various commands"""
    test_commands = [
        "open safari",
        "close safari", 
        "open music"
    ]
    
    print("Testing WebSocket Command Timing")
    print("=" * 50)
    
    for cmd in test_commands:
        await test_single_command(cmd)
        await asyncio.sleep(2)  # Wait between commands

if __name__ == "__main__":
    asyncio.run(main())