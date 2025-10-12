#!/usr/bin/env python3
"""
Test script to verify desktop spaces query is working correctly
Run this after JARVIS is fully started
"""

import asyncio
import aiohttp
import json

async def test_desktop_spaces():
    """Test the desktop spaces query through JARVIS API"""

    test_queries = [
        "What's happening across my desktop spaces?",
        "What is happening across my desktop spaces?",
        "Show me what's happening across my desktop"
    ]

    base_url = "http://localhost:8010"

    print("=" * 60)
    print("🧪 Testing Desktop Spaces Query")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # First check if backend is ready
        try:
            async with session.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    print("✓ JARVIS backend is ready")
                else:
                    print("❌ JARVIS backend not ready")
                    return
        except Exception as e:
            print(f"❌ Cannot connect to JARVIS: {e}")
            print("Make sure JARVIS is running (check jarvis_test.log)")
            return

        # Test each query
        for query in test_queries:
            print(f"\n📝 Testing: '{query}'")
            print("-" * 40)

            try:
                # Send command through voice API endpoint
                async with session.post(
                    f"{base_url}/api/voice/command",
                    json={"text": query},
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        response_text = result.get("response", "No response")

                        print(f"✅ Response received:")
                        print(f"   {response_text[:200]}...")

                        # Check if response is generic
                        if "I processed your command" in response_text:
                            print("   ⚠️ WARNING: Still getting generic response!")
                        elif "Desktop 1" in response_text or "Desktop 2" in response_text:
                            print("   ⚠️ WARNING: Using generic desktop names!")
                        elif "Cursor" in response_text or "Terminal" in response_text:
                            print("   ✅ SUCCESS: Using actual workspace names!")

                        # Check metadata
                        if "metadata" in result:
                            meta = result["metadata"]
                            if "handled_by" in meta:
                                print(f"   Handler: {meta['handled_by']}")
                            if "vision_handled" in meta:
                                print(f"   Vision handled: {meta['vision_handled']}")
                    else:
                        print(f"❌ Error: HTTP {resp.status}")

            except Exception as e:
                print(f"❌ Error testing query: {e}")

    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("=" * 60)
    print("\n💡 Expected behavior:")
    print("  • Response should use Claude's API for intelligent analysis")
    print("  • Should mention actual workspace names (Cursor, Terminal, etc)")
    print("  • Should NOT say 'I processed your command...'")
    print("  • Should NOT use 'Desktop 1', 'Desktop 2' etc.")

if __name__ == "__main__":
    asyncio.run(test_desktop_spaces())