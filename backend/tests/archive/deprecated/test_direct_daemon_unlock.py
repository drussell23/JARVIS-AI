#!/usr/bin/env python3
"""
Test Direct Unlock
==================

Test unlocking directly through the voice unlock WebSocket daemon.
"""

import asyncio
import json
import websockets


async def test_daemon_unlock():
    """Test unlocking directly through daemon WebSocket"""
    print("🔓 Testing Direct Daemon Unlock")
    print("=" * 50)
    
    try:
        # Connect directly to voice unlock daemon
        async with websockets.connect('ws://localhost:8765/voice-unlock') as ws:
            print("Connected to Voice Unlock daemon")
            
            # Send unlock command
            unlock_cmd = {
                "type": "command",
                "command": "unlock_screen",
                "parameters": {
                    "source": "test_script",
                    "authenticated": True
                }
            }
            
            print(f"Sending: {json.dumps(unlock_cmd)}")
            await ws.send(json.dumps(unlock_cmd))
            
            # Wait for response
            print("Waiting for response...")
            response = await asyncio.wait_for(ws.recv(), timeout=30.0)  # Increased timeout
            result = json.loads(response)
            
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                print("✅ Unlock successful!")
            else:
                print(f"❌ Unlock failed: {result.get('message')}")
                
    except ConnectionRefusedError:
        print("❌ Voice Unlock daemon not running on port 8765")
    except asyncio.TimeoutError:
        print("❌ Timeout waiting for response")
    except Exception as e:
        print(f"❌ Error: {e}")


async def test_daemon_status():
    """Check daemon status first"""
    print("\n📊 Checking Daemon Status")
    print("=" * 50)
    
    try:
        async with websockets.connect('ws://localhost:8765/voice-unlock') as ws:
            print("Connected to Voice Unlock daemon")
            
            # Get status
            status_cmd = {"type": "command", "command": "get_status"}
            
            await ws.send(json.dumps(status_cmd))
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            result = json.loads(response)
            
            print(f"Status: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                status = result.get('status', {})
                print(f"\nScreen locked: {status.get('isScreenLocked', 'unknown')}")
                print(f"Monitoring: {status.get('isMonitoring', False)}")
                print(f"Enrolled user: {status.get('enrolledUser', 'none')}")
            
    except ConnectionRefusedError:
        print("❌ Voice Unlock daemon not running on port 8765")
    except Exception as e:
        print(f"❌ Error: {e}")


async def test_daemon_lock():
    """Test locking through daemon"""
    print("\n🔒 Testing Direct Daemon Lock")
    print("=" * 50)
    
    try:
        async with websockets.connect('ws://localhost:8765/voice-unlock') as ws:
            print("Connected to Voice Unlock daemon")
            
            lock_cmd = {
                "type": "command",
                "command": "lock_screen",
                "parameters": {"source": "test_script"}
            }
            
            await ws.send(json.dumps(lock_cmd))
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            result = json.loads(response)
            
            if result.get('success'):
                print("✅ Lock successful!")
                return True
            else:
                print(f"❌ Lock failed: {result.get('message')}")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run tests"""
    # Check status first
    await test_daemon_status()
    
    # Test lock then unlock
    print("\nPress Enter to test lock/unlock cycle")
    input()
    
    # First lock
    if await test_daemon_lock():
        print("\nWaiting 3 seconds for screen to lock...")
        await asyncio.sleep(3)
        
        # Then unlock
        await test_daemon_unlock()
    else:
        print("Skipping unlock test since lock failed")


if __name__ == "__main__":
    asyncio.run(main())