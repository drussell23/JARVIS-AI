#!/usr/bin/env python3
"""
Test display connection with detailed logging
"""
import asyncio
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from backend.display.advanced_display_monitor import get_display_monitor

async def test_connection():
    print("\n" + "="*70)
    print("🔍 Testing Display Connection with Debug Logging")
    print("="*70)

    # Get monitor singleton
    monitor = get_display_monitor()

    # Try to connect
    print("\n🚀 Attempting to connect to Living Room TV...")
    result = await monitor._execute_display_connection("living_room_tv")

    print("\n" + "="*70)
    print("📊 Result:")
    print("="*70)
    print(f"Success: {result.get('success')}")
    print(f"Message: {result.get('message')}")
    print(f"Duration: {result.get('duration', 0):.2f}s")
    if 'tier' in result:
        print(f"Tier: {result['tier']}")
    if 'strategies_attempted' in result:
        print(f"Strategies: {result['strategies_attempted']}")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_connection())
