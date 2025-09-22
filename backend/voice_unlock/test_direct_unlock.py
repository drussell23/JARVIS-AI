#!/usr/bin/env python3
"""
Test Direct Screen Unlock via JARVIS Command
============================================

This script tests the new direct unlock functionality.
"""

import asyncio
import json
import websockets
import subprocess
import time


async def test_unlock_command():
    """Test sending unlock command through WebSocket"""
    print("🔐 Testing Direct Unlock Command")
    print("================================")
    print()
    
    # Check if WebSocket server is running
    try:
        uri = "ws://localhost:8765/voice-unlock"
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Voice Unlock WebSocket server")
            
            # Send handshake
            handshake = {
                "type": "command",
                "command": "handshake",
                "parameters": {
                    "client": "test-script",
                    "version": "1.0"
                }
            }
            await websocket.send(json.dumps(handshake))
            response = await websocket.recv()
            print(f"Handshake response: {response}")
            
            # Test direct unlock command
            print("\n📱 Sending unlock_screen command...")
            unlock_cmd = {
                "type": "command",
                "command": "unlock_screen",
                "parameters": {
                    "source": "jarvis_command",
                    "authenticated": True
                }
            }
            
            await websocket.send(json.dumps(unlock_cmd))
            response = await websocket.recv()
            result = json.loads(response)
            
            print(f"\nUnlock response: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                print("\n✅ Unlock command succeeded!")
            else:
                print(f"\n❌ Unlock failed: {result.get('message', 'Unknown error')}")
                
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        print("\nMake sure the Voice Unlock system is running:")
        print("./start_voice_unlock_system.sh")


def test_password_retrieval():
    """Test if password can be retrieved from Keychain"""
    print("\n🔑 Testing Password Retrieval")
    print("==============================")
    
    try:
        result = subprocess.run([
            'security', 'find-generic-password',
            '-s', 'com.jarvis.voiceunlock',
            '-a', 'unlock_token',
            '-g'
        ], capture_output=True, text=True)
        
        if 'password:' in result.stdout or 'password:' in result.stderr:
            print("✅ Password is stored in Keychain")
            # Don't print the actual password
            return True
        else:
            print("❌ No password found in Keychain")
            print("Run: ./enable_screen_unlock.sh")
            return False
    except Exception as e:
        print(f"❌ Error checking Keychain: {e}")
        return False


def test_screen_lock_detection():
    """Test if we can detect screen lock state"""
    print("\n🖥️  Testing Screen Lock Detection")
    print("==================================")
    
    try:
        # Check using Python
        result = subprocess.run([
            'python3', '-c',
            'import Quartz; d = Quartz.CGSessionCopyCurrentDictionary(); print("locked" if d and d.get("CGSSessionScreenIsLocked") else "unlocked")'
        ], capture_output=True, text=True)
        
        is_locked = 'locked' in result.stdout
        print(f"Screen is: {'🔒 LOCKED' if is_locked else '🔓 UNLOCKED'}")
        return is_locked
        
    except Exception as e:
        print(f"❌ Error checking screen state: {e}")
        return False


async def main():
    print("🚀 Direct Screen Unlock Test")
    print("============================")
    print()
    print("This test will:")
    print("1. Check if password is stored")
    print("2. Test the direct unlock command")
    print("3. Verify screen state")
    print()
    
    # Test password
    if not test_password_retrieval():
        return
    
    # Test screen state
    was_locked = test_screen_lock_detection()
    
    # Test unlock
    await test_unlock_command()
    
    # Wait a bit
    print("\n⏳ Waiting 2 seconds...")
    await asyncio.sleep(2)
    
    # Check screen state again
    is_locked = test_screen_lock_detection()
    
    if was_locked and not is_locked:
        print("\n🎉 SUCCESS! Screen was unlocked!")
    elif not was_locked:
        print("\n⚠️  Screen was already unlocked. Lock it and try again.")
    else:
        print("\n❌ Screen is still locked. Check logs for errors.")


if __name__ == "__main__":
    asyncio.run(main())