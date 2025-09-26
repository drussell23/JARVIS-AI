#!/usr/bin/env python3
"""
Test Lock/Unlock Flow
====================

Tests the complete flow:
1. Lock the screen with JARVIS
2. Unlock the screen with JARVIS
"""

import asyncio
import json
import websockets
import time


async def test_lock_unlock():
    """Test locking and then unlocking the screen"""
    print("🔐 Testing Lock/Unlock Flow")
    print("=" * 50)
    
    try:
        async with websockets.connect('ws://localhost:8000/voice/jarvis/stream') as ws:
            # Wait for welcome
            welcome = await ws.recv()
            print(f"Connected: {json.loads(welcome).get('message')}")
            
            # Step 1: Lock the screen
            print("\n📍 Step 1: Locking screen...")
            lock_command = {
                "type": "command",
                "text": "lock my screen"
            }
            
            await ws.send(json.dumps(lock_command))
            
            # Wait for lock response
            lock_success = False
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(response)
                    
                    if data.get('text'):
                        print(f"JARVIS: {data['text']}")
                        if 'lock' in data['text'].lower():
                            lock_success = True
                            
                    if data.get('type') == 'response' and not data.get('intermediate'):
                        break
                        
            except asyncio.TimeoutError:
                print("Timeout waiting for lock response")
                
            if lock_success:
                print("✅ Screen locked successfully")
                
                # Wait a moment for screen to lock
                print("\n⏳ Waiting 3 seconds for screen to lock...")
                await asyncio.sleep(3)
                
                # Step 2: Unlock the screen
                print("\n📍 Step 2: Unlocking screen...")
                unlock_command = {
                    "type": "command", 
                    "text": "unlock my screen"
                }
                
                await ws.send(json.dumps(unlock_command))
                
                # Wait for unlock response
                unlock_success = False
                try:
                    while True:
                        response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        data = json.loads(response)
                        
                        if data.get('text'):
                            print(f"JARVIS: {data['text']}")
                            if 'unlock' in data['text'].lower():
                                unlock_success = True
                                
                        if data.get('type') == 'response' and not data.get('intermediate'):
                            break
                            
                except asyncio.TimeoutError:
                    print("Timeout waiting for unlock response")
                    
                if unlock_success:
                    print("✅ Screen unlocked successfully!")
                else:
                    print("❌ Failed to unlock screen")
                    
            else:
                print("❌ Failed to lock screen - skipping unlock test")
                
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n" + "=" * 50)
    print("Test Complete!")


async def test_quick_lock_unlock():
    """Quick test without delays"""
    print("\n\n⚡ Quick Lock/Unlock Test")
    print("=" * 50)
    
    try:
        # Test lock
        async with websockets.connect('ws://localhost:8000/voice/jarvis/stream') as ws:
            await ws.recv()  # Welcome
            
            print("Testing: 'lock my screen'")
            await ws.send(json.dumps({"type": "command", "text": "lock my screen"}))
            
            # Just check if we get any response
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                data = json.loads(response)
                if data.get('text'):
                    print(f"Response: {data['text']}")
            except:
                print("No response for lock command")
                
        # Test unlock  
        async with websockets.connect('ws://localhost:8000/voice/jarvis/stream') as ws:
            await ws.recv()  # Welcome
            
            print("\nTesting: 'unlock my screen'")
            await ws.send(json.dumps({"type": "command", "text": "unlock my screen"}))
            
            # Just check if we get any response
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                data = json.loads(response)
                if data.get('text'):
                    print(f"Response: {data['text']}")
            except:
                print("No response for unlock command")
                
    except Exception as e:
        print(f"Quick test error: {e}")


async def main():
    """Run tests"""
    # First do quick test
    await test_quick_lock_unlock()
    
    # Then full test
    print("\n\nPress Enter to run full lock/unlock test (will actually lock your screen)")
    input()
    await test_lock_unlock()


if __name__ == "__main__":
    asyncio.run(main())