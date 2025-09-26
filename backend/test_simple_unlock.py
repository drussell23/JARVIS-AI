#!/usr/bin/env python3
"""
Simple Unlock Test
==================

Test just the unlock functionality.
"""

import asyncio
import websockets
import json
import subprocess

async def lock_screen():
    """Lock screen using Voice Unlock daemon"""
    print("🔒 Locking screen...")
    uri = "ws://localhost:8765/voice-unlock"
    
    async with websockets.connect(uri) as ws:
        lock_cmd = json.dumps({
            "type": "command",
            "command": "lock_screen"
        })
        await ws.send(lock_cmd)
        response = await ws.recv()
        result = json.loads(response)
        print(f"Lock result: {result.get('message')}")
        
async def test_unlock():
    """Test unlock directly"""
    print("\n🔓 Testing unlock...")
    
    # Import and test
    from api.direct_unlock_handler import unlock_screen_direct
    
    print("Attempting to unlock screen...")
    success = await unlock_screen_direct("Test unlock")
    
    if success:
        print("✅ Unlock succeeded!")
    else:
        print("❌ Unlock failed!")
        
    return success

async def main():
    """Run test"""
    # First lock
    await lock_screen()
    
    # Wait for lock
    print("\nWaiting 3 seconds for lock to take effect...")
    await asyncio.sleep(3)
    
    # Test unlock
    await test_unlock()

if __name__ == "__main__":
    asyncio.run(main())