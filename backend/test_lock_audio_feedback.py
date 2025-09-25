#!/usr/bin/env python3
"""
Test Lock/Unlock Audio Feedback
===============================

Tests that JARVIS provides audio feedback for lock/unlock commands.
"""

import asyncio
import websockets
import json
import time

async def test_lock_unlock_audio():
    """Test lock/unlock commands and verify audio feedback"""
    print("🔊 Testing Lock/Unlock Audio Feedback")
    print("=" * 50)
    
    uri = 'ws://localhost:8000/voice/jarvis/stream'
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected to JARVIS WebSocket")
            
            # Wait for connection message
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"Connection: {data.get('message', 'Connected')}")
            
            # Test lock command
            print("\n📍 Testing LOCK command...")
            await ws.send(json.dumps({
                "type": "command",
                "text": "lock my screen"
            }))
            
            # Collect all responses for lock
            lock_responses = []
            start_time = time.time()
            while time.time() - start_time < 5:  # Wait up to 5 seconds
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(response)
                    lock_responses.append(data)
                    
                    if data.get('type') == 'response':
                        print(f"\nLOCK Response received:")
                        print(f"  Text: {data.get('text', 'No text')}")
                        print(f"  Speak flag: {data.get('speak', False)}")
                        print(f"  Command type: {data.get('command_type', 'unknown')}")
                        
                        if data.get('speak') == True:
                            print("  ✅ Frontend SHOULD speak this response!")
                        else:
                            print("  ❌ No speak flag - Frontend won't speak!")
                            
                except asyncio.TimeoutError:
                    break
            
            print(f"\nReceived {len(lock_responses)} messages for lock command")
            
            # Brief pause
            await asyncio.sleep(2)
            
            # Test unlock command (won't actually unlock if screen isn't locked)
            print("\n📍 Testing UNLOCK command...")
            await ws.send(json.dumps({
                "type": "command",
                "text": "unlock my screen"
            }))
            
            # Collect all responses for unlock
            unlock_responses = []
            start_time = time.time()
            while time.time() - start_time < 5:  # Wait up to 5 seconds
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(response)
                    unlock_responses.append(data)
                    
                    if data.get('type') == 'response':
                        print(f"\nUNLOCK Response received:")
                        print(f"  Text: {data.get('text', 'No text')}")
                        print(f"  Speak flag: {data.get('speak', False)}")
                        print(f"  Command type: {data.get('command_type', 'unknown')}")
                        
                        if data.get('speak') == True:
                            print("  ✅ Frontend SHOULD speak this response!")
                        else:
                            print("  ❌ No speak flag - Frontend won't speak!")
                            
                except asyncio.TimeoutError:
                    break
                    
            print(f"\nReceived {len(unlock_responses)} messages for unlock command")
            
            # Summary
            print("\n" + "=" * 50)
            print("SUMMARY:")
            
            # Check if responses had speak flags
            lock_speak = any(msg.get('type') == 'response' and msg.get('speak') == True 
                           for msg in lock_responses)
            unlock_speak = any(msg.get('type') == 'response' and msg.get('speak') == True 
                             for msg in unlock_responses)
            
            print(f"Lock command has speak flag: {'✅ YES' if lock_speak else '❌ NO'}")
            print(f"Unlock command has speak flag: {'✅ YES' if unlock_speak else '❌ NO'}")
            
            if lock_speak and unlock_speak:
                print("\n✅ JARVIS is sending audio feedback for lock/unlock commands!")
                print("If you don't hear audio, check:")
                print("  1. Frontend console for errors")
                print("  2. Browser audio permissions")
                print("  3. System volume settings")
            else:
                print("\n❌ JARVIS is NOT sending speak flags for lock/unlock commands")
                print("This needs to be fixed in the backend response handling")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("  1. JARVIS is running on port 8000")
        print("  2. WebSocket endpoint is available")


if __name__ == "__main__":
    print("Testing JARVIS lock/unlock audio feedback...")
    print("This will send lock and unlock commands to check responses\n")
    
    asyncio.run(test_lock_unlock_audio())