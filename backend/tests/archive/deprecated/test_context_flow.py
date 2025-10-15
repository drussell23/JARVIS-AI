#!/usr/bin/env python3
"""
Test Context Intelligence Flow
==============================

Test if Context Intelligence properly detects locked screen.
"""

import asyncio
import aiohttp
import json

async def test_flow():
    """Test the complete flow"""
    print("🔍 Testing Context Intelligence Flow")
    print("=" * 50)
    
    url = "http://localhost:8000/api/command"
    
    try:
        async with aiohttp.ClientSession() as session:
            # First lock the screen
            print("\n1️⃣ Locking screen...")
            data = {"command": "lock my screen"}
            
            async with session.post(url, json=data) as response:
                result = await response.json()
                print(f"Lock result: {result.get('response', 'No response')}")
            
            # Wait for lock to take effect
            await asyncio.sleep(3)
            
            # Now try a command that requires screen
            print("\n2️⃣ Testing command with locked screen...")
            data = {"command": "open safari and search for dogs"}
            
            async with session.post(url, json=data) as response:
                result = await response.json()
                print(f"\nResponse: {result.get('response', 'No response')}")
                
                # Check if it detected locked screen
                if "locked" in str(result).lower() or "unlock" in str(result).lower():
                    print("\n✅ Context Intelligence detected locked screen!")
                else:
                    print("\n❌ Context Intelligence did NOT detect locked screen")
                    print(f"Full response: {json.dumps(result, indent=2)}")
                    
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    print("⚠️  This test will lock your screen!")
    print("Make sure you can unlock it.")
    asyncio.run(test_flow())