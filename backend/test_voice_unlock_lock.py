#!/usr/bin/env python3
"""
Test Voice Unlock Lock Functionality
===================================

Check if Voice Unlock can actually lock the screen.
"""

import asyncio
import websockets
import json


async def test_voice_unlock_lock():
    """Test if Voice Unlock daemon can lock the screen"""
    print("\n🔐 Testing Voice Unlock Lock Command")
    print("="*50)
    
    uri = "ws://localhost:8765/voice-unlock"
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected to Voice Unlock daemon")
            
            # Send lock command
            lock_cmd = json.dumps({
                "type": "command",
                "command": "lock_screen",
                "parameters": {"source": "test"}
            })
            
            print("\nSending lock_screen command...")
            await ws.send(lock_cmd)
            
            # Get response
            response = await ws.recv()
            result = json.loads(response)
            
            print(f"\nResponse:")
            print(json.dumps(result, indent=2))
            
            if result.get('success'):
                print("\n✅ Lock command succeeded!")
            else:
                print(f"\n❌ Lock command failed: {result.get('message', 'Unknown error')}")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def test_context_intelligence_lock():
    """Test Context Intelligence lock as comparison"""
    print("\n\n🔒 Testing Context Intelligence Lock")
    print("="*50)
    
    from context_intelligence.core.unlock_manager import get_unlock_manager
    unlock_manager = get_unlock_manager()
    
    success, message = await unlock_manager.lock_screen("Test comparison")
    
    print(f"Success: {success}")
    print(f"Message: {message}")


async def main():
    # Test Voice Unlock
    await test_voice_unlock_lock()
    
    # Ask before testing Context Intelligence
    print("\n" + "-"*50)
    print("Voice Unlock test complete.")
    print("Context Intelligence would lock the screen immediately.")
    print("(Skipping to avoid actually locking your screen)")


if __name__ == "__main__":
    asyncio.run(main())