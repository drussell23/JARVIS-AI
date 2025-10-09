#!/usr/bin/env python3
"""
Debug Lock Detection Issue
==========================

Test to understand why JARVIS isn't detecting locked screens
"""

import asyncio
import subprocess
import time
import json
from datetime import datetime

# Import the detection function
from api.direct_unlock_handler_fixed import check_screen_locked_direct

async def test_actual_lock_detection():
    """Test if we can detect actual screen lock"""
    print("\n" + "="*60)
    print("🔍 Testing Real Screen Lock Detection")
    print("="*60)
    
    # First check current state
    print("\n1️⃣ Checking current screen state...")
    is_locked = await check_screen_locked_direct()
    print(f"   Current state: {'LOCKED' if is_locked else 'UNLOCKED'}")
    
    # Now lock the screen
    print("\n2️⃣ Locking screen in 3 seconds...")
    print("   (Press Ctrl+C to cancel)")
    
    try:
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n❌ Cancelled")
        return
    
    # Lock screen
    print("\n   Locking screen now...")
    lock_cmd = """osascript -e 'tell app "System Events" to key code 12 using {control down, command down}'"""
    subprocess.run(lock_cmd, shell=True)
    
    # Wait for lock to take effect
    print("   Waiting 3 seconds for lock to complete...")
    time.sleep(3)
    
    # Check again
    print("\n3️⃣ Checking screen state after lock...")
    is_locked_after = await check_screen_locked_direct()
    print(f"   State after lock: {'LOCKED' if is_locked_after else 'UNLOCKED'}")
    
    if not is_locked_after:
        print("\n❌ PROBLEM: Screen should be locked but detection says UNLOCKED!")
        print("   This is why JARVIS doesn't detect the lock!")
    else:
        print("\n✅ Lock detection is working correctly")

async def test_websocket_connection():
    """Test direct WebSocket connection to voice unlock daemon"""
    print("\n" + "="*60)
    print("🔌 Testing WebSocket Connection to Voice Unlock Daemon")
    print("="*60)
    
    try:
        import websockets
        
        # Connect directly
        uri = "ws://localhost:8765/voice-unlock"
        print(f"\nConnecting to: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            
            # Send status command
            command = {"type": "command", "command": "get_status"}
            print(f"\nSending: {json.dumps(command)}")
            await websocket.send(json.dumps(command))
            
            # Get response
            response = await websocket.recv()
            data = json.loads(response)
            print(f"\nReceived: {json.dumps(data, indent=2)}")
            
            # Check lock status
            if 'status' in data:
                is_locked = data['status'].get('isScreenLocked', False)
                print(f"\n📊 Screen locked according to daemon: {is_locked}")
    
    except Exception as e:
        print(f"\n❌ WebSocket connection error: {e}")
        print("   Is the voice unlock daemon running?")

async def test_alternative_detection():
    """Test alternative screen lock detection methods"""
    print("\n" + "="*60)
    print("🔬 Testing Alternative Detection Methods")
    print("="*60)
    
    # Method 1: Using system_profiler
    print("\n1️⃣ Testing CGSessionCopyCurrentDictionary...")
    cmd1 = """python3 -c "
import Quartz
session_dict = Quartz.CGSessionCopyCurrentDictionary()
if session_dict:
    print('Session info:', dict(session_dict))
    locked = session_dict.get('CGSSessionScreenIsLocked', False)
    print('Screen locked:', locked)
else:
    print('No session info available')
"
"""
    result1 = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
    print(result1.stdout)
    if result1.stderr:
        print(f"Error: {result1.stderr}")
    
    # Method 2: Check screensaver
    print("\n2️⃣ Testing screensaver status...")
    cmd2 = """osascript -e 'tell application "System Events" to get running of screen saver'"""
    result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    print(f"Screensaver running: {result2.stdout.strip()}")
    
    # Method 3: Check if loginwindow is frontmost
    print("\n3️⃣ Testing loginwindow status...")
    cmd3 = """osascript -e 'tell application "System Events" to get name of first process whose frontmost is true'"""
    result3 = subprocess.run(cmd3, shell=True, capture_output=True, text=True)
    front_app = result3.stdout.strip()
    print(f"Frontmost app: {front_app}")
    if "loginwindow" in front_app.lower():
        print("   ⚠️  Login window is frontmost - screen may be locked")

if __name__ == "__main__":
    print("🚀 Starting Lock Detection Debug")
    
    # Run all tests
    asyncio.run(test_websocket_connection())
    asyncio.run(test_alternative_detection())
    asyncio.run(test_actual_lock_detection())