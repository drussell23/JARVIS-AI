#!/usr/bin/env python3
"""
Test Fixed Lock Detection
=========================

Verify that screen lock detection is now working
"""

import asyncio
import json
import subprocess
import time

async def test_lock_detection():
    """Test if lock detection is now working"""
    import websockets
    
    print("🔍 Testing Fixed Lock Detection")
    print("="*50)
    
    # Connect to WebSocket
    uri = "ws://localhost:8765/voice-unlock"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Test 1: Check current state
            print("\n1️⃣ Checking current screen state...")
            await websocket.send(json.dumps({
                "type": "command",
                "command": "get_status"
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            is_locked = data['status']['isScreenLocked']
            print(f"   Current: {'LOCKED' if is_locked else 'UNLOCKED'}")
            
            # Test 2: Lock screen and check again
            print("\n2️⃣ Locking screen in 3 seconds...")
            for i in range(3, 0, -1):
                print(f"   {i}...")
                time.sleep(1)
                
            # Lock screen
            lock_cmd = """osascript -e 'tell app "System Events" to key code 12 using {control down, command down}'"""
            subprocess.run(lock_cmd, shell=True)
            
            # Wait for lock
            print("   Waiting for lock to complete...")
            await asyncio.sleep(3)
            
            # Check again
            print("\n3️⃣ Checking after lock...")
            await websocket.send(json.dumps({
                "type": "command", 
                "command": "get_status"
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            is_locked_after = data['status']['isScreenLocked']
            print(f"   After lock: {'LOCKED' if is_locked_after else 'UNLOCKED'}")
            
            if is_locked_after:
                print("\n✅ SUCCESS! Lock detection is now working!")
            else:
                print("\n❌ Still not detecting lock properly")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_lock_detection())