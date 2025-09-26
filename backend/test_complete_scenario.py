#!/usr/bin/env python3
"""
Test Complete Context Intelligence Scenario
===========================================

Tests the full flow: lock screen -> command -> queue -> unlock -> execute
"""

import asyncio
import aiohttp
import json

async def test_scenario():
    """Test the complete scenario"""
    print("🔒 TESTING COMPLETE CONTEXT INTELLIGENCE SCENARIO")
    print("=" * 60)
    print("\nThis test will:")
    print("1. Lock your screen")
    print("2. Send a command that requires screen")
    print("3. Verify Context Intelligence queues it")
    print("\n⚠️  YOUR SCREEN WILL LOCK!")
    
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
            
            # Step 2: Test command with locked screen
            print("\n2️⃣ SENDING COMMAND WITH LOCKED SCREEN...")
            data = {"command": "open safari and search for dogs"}
            
            async with session.post(url, json=data) as response:
                result = await response.json()
                
                print(f"\n   Response: {result.get('response', 'No response')}")
                print(f"   Success: {result.get('success')}")
                
                # Check the full result
                print(f"\n   Full result:")
                print(json.dumps(result, indent=4))
                
                # Analyze the response
                response_text = str(result).lower()
                if any(word in response_text for word in ['locked', 'unlock', 'queue']):
                    print("\n✅ SUCCESS! Context Intelligence detected locked screen!")
                    print("   The command was properly handled for locked state.")
                else:
                    print("\n❌ FAIL! Context Intelligence did NOT handle locked screen properly.")
                    print("   The command was executed immediately.")
                    
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

async def test_direct_context():
    """Test Context Intelligence directly"""
    print("\n\n🔍 DIRECT CONTEXT INTELLIGENCE TEST")
    print("=" * 60)
    
    from context_intelligence.core.screen_state import ScreenStateDetector
    from context_intelligence.core.context_manager import ContextManager
    
    # Check screen state
    detector = ScreenStateDetector()
    state = await detector.get_screen_state()
    
    print(f"\nCurrent screen state: {state.state.value}")
    print(f"Confidence: {state.confidence:.2f}")
    
    # Test context manager
    manager = ContextManager()
    system_state = await manager.system_monitor.get_states()
    
    print(f"\nSystem state:")
    print(f"  Screen locked: {system_state.get('screen_locked')}")
    print(f"  Active apps: {system_state.get('active_apps', [])[:3]}...")

async def main():
    """Run all tests"""
    await test_scenario()
    await test_direct_context()

if __name__ == "__main__":
    asyncio.run(main())