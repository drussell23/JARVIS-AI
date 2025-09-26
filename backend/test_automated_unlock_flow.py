#!/usr/bin/env python3
"""
Test Full Automated Unlock Flow
================================

Tests the complete automated scenario:
1. Lock screen
2. Issue command requiring screen
3. JARVIS automatically unlocks and executes
"""

import asyncio
import aiohttp
import json
import time

async def test_automated_flow():
    """Test the complete automated flow"""
    print("🤖 TESTING FULL AUTOMATED UNLOCK FLOW")
    print("=" * 60)
    print("\nThis test will:")
    print("1. Lock your screen")
    print("2. Send 'open safari and search for dogs'")
    print("3. JARVIS should automatically:")
    print("   - Detect screen is locked")
    print("   - Queue the command")
    print("   - Unlock the screen (type password)")
    print("   - Execute the command")
    print("   - Report success")
    
    print("\n⚠️  YOUR SCREEN WILL LOCK AND THEN AUTOMATICALLY UNLOCK!")
    await asyncio.sleep(3)
    
    url = "http://localhost:8000/api/command"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Lock the screen
            print("\n1️⃣ LOCKING SCREEN...")
            data = {"command": "lock my screen"}
            
            async with session.post(url, json=data) as response:
                result = await response.json()
                print(f"   Response: {result.get('response', 'No response')}")
                
            # Wait for lock to take effect
            print("\n   Waiting 5 seconds for screen to lock...")
            await asyncio.sleep(5)
            
            # Step 2: Send command that requires screen
            print("\n2️⃣ SENDING COMMAND (JARVIS SHOULD AUTO-UNLOCK)...")
            start_time = time.time()
            data = {"command": "open safari and search for dogs"}
            
            async with session.post(url, json=data) as response:
                result = await response.json()
                elapsed = time.time() - start_time
                
                print(f"\n   Response: {result.get('response', 'No response')}")
                print(f"   Success: {result.get('success')}")
                print(f"   Time taken: {elapsed:.1f}s")
                
                # Check if context was handled
                if result.get('context_handled'):
                    print(f"   Context handled: ✅")
                    if result.get('screen_unlocked'):
                        print(f"   Screen unlocked: ✅")
                
                # Analyze the response
                response_text = result.get('response', '').lower()
                
                if "unlocked your screen" in response_text and "safari" in response_text:
                    print("\n✅ SUCCESS! Full automated flow worked!")
                    print("   - Screen was locked")
                    print("   - JARVIS detected it")
                    print("   - JARVIS unlocked it automatically")
                    print("   - JARVIS executed the command")
                    print("   - All automated, no manual intervention!")
                elif "couldn't unlock" in response_text:
                    print("\n❌ FAIL! JARVIS detected lock but couldn't unlock")
                    print("   Check if password is stored in Voice Unlock")
                elif result.get('success') and 'safari' in response_text:
                    print("\n⚠️  WARNING! Command executed but no unlock mentioned")
                    print("   Either screen wasn't locked or Context Intelligence was bypassed")
                else:
                    print("\n❌ FAIL! Unexpected result")
                
                # Show full result for debugging
                print(f"\n   Full result:")
                print(json.dumps(result, indent=4))
                    
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

async def check_voice_unlock_status():
    """Check if Voice Unlock has password stored"""
    print("\n\n🔍 CHECKING VOICE UNLOCK STATUS")
    print("=" * 60)
    
    import websockets
    
    try:
        uri = "ws://localhost:8765/voice-unlock"
        async with websockets.connect(uri) as ws:
            # Get status
            status_cmd = json.dumps({
                "type": "command",
                "command": "get_status"
            })
            await ws.send(status_cmd)
            
            response = await ws.recv()
            result = json.loads(response)
            
            print("Voice Unlock daemon status:")
            status = result.get('status', {})
            print(f"  Enrolled user: {status.get('enrolledUser', 'none')}")
            print(f"  Monitoring: {status.get('isMonitoring', False)}")
            
            # Check if password is available
            print("\nChecking if password is stored...")
            from api.direct_unlock_handler import check_screen_locked_system
            import subprocess
            
            # Check keychain
            try:
                result = subprocess.run([
                    'security', 'find-generic-password',
                    '-s', 'com.jarvis.voiceunlock',
                    '-a', 'unlock_token'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Password is stored in keychain")
                else:
                    print("❌ No password found in keychain")
                    print("   Run: voice_unlock/enable_screen_unlock.sh")
            except:
                print("Could not check keychain")
                
    except Exception as e:
        print(f"Error checking Voice Unlock: {e}")

async def main():
    """Run all tests"""
    await test_automated_flow()
    await check_voice_unlock_status()

if __name__ == "__main__":
    asyncio.run(main())