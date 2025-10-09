#!/usr/bin/env python3
"""
Debug WebSocket Connection
==========================

Tests WebSocket connection with detailed debugging
"""

import asyncio
import json
import websockets
import traceback

async def test_websocket():
    """Test WebSocket connection with debugging"""
    print("\n🔧 Testing WebSocket Connection Debug")
    print("="*60)
    
    uri = 'ws://localhost:8000/voice/jarvis/stream'
    print(f"\n📡 Connecting to: {uri}")
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ WebSocket connected successfully")
            
            # Wait for welcome message
            try:
                print("\n⏳ Waiting for welcome message...")
                welcome = await asyncio.wait_for(ws.recv(), timeout=5.0)
                welcome_data = json.loads(welcome)
                print(f"📨 Welcome: {welcome_data.get('message', welcome_data)}")
            except asyncio.TimeoutError:
                print("⏱️  No welcome message received within 5 seconds")
            except Exception as e:
                print(f"❌ Error receiving welcome: {e}")
                traceback.print_exc()
            
            # Send test command
            print("\n📤 Sending test command...")
            command = {
                "type": "command",
                "text": "unlock my screen"
            }
            
            await ws.send(json.dumps(command))
            print("✅ Command sent")
            
            # Wait for response
            print("\n⏳ Waiting for response...")
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(response)
                    print(f"\n📨 Response type: {data.get('type')}")
                    print(f"   Content: {json.dumps(data, indent=2)}")
                    
                    if data.get('type') == 'response' and data.get('text'):
                        break
                        
            except asyncio.TimeoutError:
                print("⏱️  No response received within 5 seconds")
            except websockets.exceptions.ConnectionClosed as e:
                print(f"❌ Connection closed: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")
                traceback.print_exc()
                
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"\n❌ WebSocket connection rejected with status {e.status_code}")
        print(f"   Headers: {e.headers}")
    except ConnectionRefusedError:
        print("\n❌ Connection refused - is JARVIS running on port 8000?")
    except Exception as e:
        print(f"\n❌ Connection error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 WebSocket Connection Debugger")
    asyncio.run(test_websocket())