#!/usr/bin/env python3
"""
Test Screen Lock Context Awareness
==================================

Tests the context-aware handling of screen lock scenarios
"""

import asyncio
import aiohttp
import json
import subprocess
import time


def lock_screen():
    """Lock the screen using AppleScript"""
    try:
        # Use the CGSession method to lock screen
        subprocess.run([
            'osascript', '-e', 
            'tell application "System Events" to keystroke "q" using {control down, command down}'
        ])
        time.sleep(2)  # Give time for lock to take effect
        print("✅ Screen locked")
    except Exception as e:
        print(f"❌ Failed to lock screen: {e}")


def check_screen_locked():
    """Check if screen is locked"""
    try:
        result = subprocess.run(['python', '-c', '''
import Quartz
session_dict = Quartz.CGSessionCopyCurrentDictionary()
if session_dict:
    locked = session_dict.get("CGSSessionScreenIsLocked", False)
    print("true" if locked else "false")
else:
    print("false")
'''], capture_output=True, text=True)
        
        return result.stdout.strip().lower() == "true"
    except Exception as e:
        print(f"Error checking screen lock: {e}")
        return False


async def test_context_awareness():
    """Test context-aware command processing"""
    
    api_url = "http://localhost:8000/api/command"
    
    print("🧪 Testing JARVIS Screen Lock Context Awareness")
    print("=" * 60)
    
    # Wait for backend to be ready
    print("\n⏳ Waiting for backend to be ready...")
    await asyncio.sleep(20)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Command that doesn't require screen
        print("\n📝 Test 1: Command that doesn't require screen access")
        command1 = "What time is it?"
        
        try:
            async with session.post(api_url, json={"command": command1}) as resp:
                result1 = await resp.json()
                print(f"Command: {command1}")
                print(f"Response: {result1.get('response', result1)}")
                print(f"Context handled: {result1.get('context_handled', False)}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test 2: Lock the screen
        print("\n📝 Test 2: Lock the screen")
        lock_command = "lock my screen"
        
        try:
            async with session.post(api_url, json={"command": lock_command}) as resp:
                result = await resp.json()
                print(f"Command: {lock_command}")
                print(f"Response: {result.get('response', result)}")
                
                # Wait for lock to complete
                await asyncio.sleep(3)
                
                # Check if screen is locked
                is_locked = check_screen_locked()
                print(f"Screen locked: {is_locked}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test 3: Command that requires screen while locked
        print("\n📝 Test 3: Command that requires screen access (while locked)")
        command3 = "Open Safari and search for puppies"
        
        # First make sure screen is locked
        if not check_screen_locked():
            print("⚠️  Screen is not locked. Locking it now...")
            lock_screen()
            await asyncio.sleep(3)
        
        print(f"Screen is locked: {check_screen_locked()}")
        print(f"Command: {command3}")
        
        try:
            async with session.post(api_url, json={"command": command3}) as resp:
                result3 = await resp.json()
                print(f"\nResponse: {result3.get('response', result3)}")
                print(f"Success: {result3.get('success', False)}")
                print(f"Context handled: {result3.get('context_handled', False)}")
                print(f"Screen unlocked: {result3.get('screen_unlocked', False)}")
                
                # Wait and check if screen was unlocked
                await asyncio.sleep(3)
                is_locked_after = check_screen_locked()
                print(f"\nScreen locked after command: {is_locked_after}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Context awareness test completed!")
    print("\n🎯 Expected behavior:")
    print("1. JARVIS should detect the locked screen")
    print("2. JARVIS should say: 'Your screen is locked. I'll unlock it now by typing in the password.'")
    print("3. JARVIS should unlock the screen")
    print("4. JARVIS should then execute the command (open Safari)")
    print("5. The response should include confirmation of all steps taken")


if __name__ == "__main__":
    print("Starting Screen Lock Context Awareness Test\n")
    asyncio.run(test_context_awareness())