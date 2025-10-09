#!/usr/bin/env python3
"""
Test Context Awareness for JARVIS
==================================

Tests the screen lock detection and context-aware command handling
"""

import asyncio
import aiohttp
import json
import subprocess
import time

async def test_context_awareness():
    """Test context-aware command processing"""
    
    # API endpoint
    api_url = "http://localhost:8000/api/command"
    
    print("🧪 Testing JARVIS Context Awareness")
    print("=" * 50)
    
    # Test 1: Simple command that doesn't require screen
    print("\n📝 Test 1: Command that doesn't require screen access")
    command1 = "What time is it?"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json={"command": command1}) as resp:
            result1 = await resp.json()
            print(f"Command: {command1}")
            print(f"Response: {result1.get('response', result1)}")
            if result1.get('context_aware'):
                print("✅ Context awareness is active")
                if result1.get('steps_taken'):
                    print("Steps taken:")
                    for step in result1['steps_taken']:
                        print(f"  - {step['description']}")
            print()
    
    # Test 2: Command that requires screen access
    print("\n📝 Test 2: Command that requires screen access")
    command2 = "Open Safari and search for puppies"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json={"command": command2}) as resp:
            result2 = await resp.json()
            print(f"Command: {command2}")
            print(f"Response: {result2.get('response', result2)}")
            if result2.get('context_aware'):
                print("✅ Context awareness is active")
                if result2.get('steps_taken'):
                    print("\nSteps taken:")
                    for step in result2['steps_taken']:
                        print(f"  {step['step']}. {step['description']}")
                        if 'details' in step and isinstance(step['details'], dict):
                            for key, value in step['details'].items():
                                print(f"     - {key}: {value}")
            print()
    
    # Test 3: Check screen lock detection
    print("\n📝 Test 3: Direct screen lock detection")
    
    # Check if context intelligence components are available
    try:
        from context_intelligence.core.system_state_monitor import get_system_monitor
        monitor = get_system_monitor()
        
        # Get screen lock state
        is_locked = await monitor.get_state("screen_locked")
        print(f"Screen locked: {is_locked}")
        
        # Get full system context
        context = await monitor.get_system_context()
        print("\nSystem Context:")
        print(f"  - Screen accessible: {context['state_summary']['screen_accessible']}")
        print(f"  - Apps running: {context['state_summary']['apps_running']}")
        print(f"  - Network available: {context['state_summary']['network_available']}")
        
    except Exception as e:
        print(f"⚠️  Could not test screen lock detection directly: {e}")
        
    print("\n" + "=" * 50)
    print("✅ Context awareness test completed!")
    print("\nTo test with a locked screen:")
    print("1. Lock your screen (Cmd+Ctrl+Q or Apple menu > Lock Screen)")
    print("2. Say: 'Hey JARVIS, open Safari and search for dogs'")
    print("3. JARVIS should:")
    print("   - Detect the locked screen")
    print("   - Tell you it will unlock by typing the password")
    print("   - Unlock the screen")
    print("   - Open Safari and search")
    print("   - Confirm what was done")

if __name__ == "__main__":
    asyncio.run(test_context_awareness())