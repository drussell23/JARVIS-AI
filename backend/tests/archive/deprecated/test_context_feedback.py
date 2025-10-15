#!/usr/bin/env python3
"""
Test Context Intelligence Feedback
==================================

Tests that JARVIS properly provides feedback when screen is locked.
"""

import asyncio
import aiohttp
import json
import time

async def test_lock_and_command():
    """Test lock screen then command flow"""
    print("\n" + "="*70)
    print("🧪 TESTING CONTEXT INTELLIGENCE FEEDBACK")
    print("="*70)
    
    url = "http://localhost:8000/api/command"
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Lock the screen
        print(f"\n[{time.strftime('%H:%M:%S')}] 🔒 Step 1: Locking screen...")
        data = {"command": "lock my screen"}
        
        async with session.post(url, json=data) as response:
            result = await response.json()
            print(f"[{time.strftime('%H:%M:%S')}] Response: {result.get('response', 'No response')}")
            
        # Wait for lock
        print(f"\n[{time.strftime('%H:%M:%S')}] ⏳ Waiting 5 seconds for screen to lock...")
        await asyncio.sleep(5)
        
        # Step 2: Test command with locked screen
        print(f"\n[{time.strftime('%H:%M:%S')}] 🗣️  Step 2: Testing command with locked screen...")
        print(f"[{time.strftime('%H:%M:%S')}] Command: 'open safari and search for dogs'")
        
        data = {"command": "open safari and search for dogs"}
        
        async with session.post(url, json=data) as response:
            result = await response.json()
            response_text = result.get('response', 'No response')
            
            print(f"\n[{time.strftime('%H:%M:%S')}] JARVIS Response: '{response_text}'")
            
            # Check if we got the proper feedback
            if "locked" in response_text.lower() and "unlocking" in response_text.lower():
                print(f"\n✅ SUCCESS: JARVIS provided proper feedback!")
                print("   - Detected screen was locked")
                print("   - Informed user about unlocking")
                print("   - Response matches PRD requirements")
            else:
                print(f"\n⚠️  WARNING: Expected feedback about screen lock")
                print(f"   Got: '{response_text}'")
                print(f"   Expected something like: 'Your screen is locked, unlocking now.'")
                
        # Step 3: Wait and check if command executes
        print(f"\n[{time.strftime('%H:%M:%S')}] ⏳ Waiting to see if command executes after unlock...")
        await asyncio.sleep(10)
        
        print("\n" + "="*70)
        print("Test complete. Check if:")
        print("1. JARVIS detected screen was locked")
        print("2. JARVIS provided feedback: 'Your screen is locked, unlocking now.'")  
        print("3. Screen was unlocked automatically")
        print("4. Safari opened and searched for dogs")
        print("="*70)

async def main():
    """Run the test"""
    await test_lock_and_command()

if __name__ == "__main__":
    asyncio.run(main())