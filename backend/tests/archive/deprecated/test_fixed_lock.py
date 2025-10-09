#!/usr/bin/env python3
"""
Test Fixed Lock Command
=======================

Test if the Voice Unlock lock implementation now works.
"""

import asyncio
import websockets
import json
import time

async def test_lock():
    """Test the fixed lock implementation"""
    print("🔒 Testing Fixed Voice Unlock Lock")
    print("=" * 50)
    
    uri = "ws://localhost:8765/voice-unlock"
    
    try:
        print("Connecting to Voice Unlock daemon...")
        async with websockets.connect(uri) as ws:
            print("✅ Connected!")
            
            # Wait a moment
            await asyncio.sleep(1)
            
            # Send lock command
            lock_cmd = json.dumps({
                "type": "command",
                "command": "lock_screen",
                "parameters": {"source": "test"}
            })
            
            print("\nSending lock_screen command...")
            print("⚠️  Your screen will lock in 3 seconds!")
            print("3...")
            await asyncio.sleep(1)
            print("2...")
            await asyncio.sleep(1)
            print("1...")
            await asyncio.sleep(1)
            
            await ws.send(lock_cmd)
            
            # Get response
            response = await ws.recv()
            result = json.loads(response)
            
            print(f"\nResponse:")
            print(json.dumps(result, indent=2))
            
            if result.get('success'):
                print("\n✅ Lock command succeeded!")
                print("Your screen should be locked now.")
            else:
                print(f"\n❌ Lock command failed: {result.get('message', 'Unknown error')}")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Voice Unlock daemon is running:")
        print("cd voice_unlock/objc/server && python websocket_server.py")

if __name__ == "__main__":
    print("This test will LOCK YOUR SCREEN!")
    print("Make sure you know your password to unlock.")
    response = input("\nContinue? (y/N): ")
    
    if response.lower() == 'y':
        asyncio.run(test_lock())
    else:
        print("Test cancelled.")