#!/usr/bin/env python3
"""
Test Lock Command Fix
====================

Tests that lock commands now work through Context Intelligence.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_lock_command():
    """Test that lock command works without Voice Unlock daemon"""
    from api.voice_unlock_integration import handle_voice_unlock_in_jarvis
    
    print("\n🔐 Testing Lock Command Fix")
    print("="*50)
    
    # Test lock command
    command = "lock my screen"
    print(f"\nCommand: '{command}'")
    
    result = await handle_voice_unlock_in_jarvis(command)
    
    print(f"\nResult:")
    print(f"- Success: {result.get('success')}")
    print(f"- Response: {result.get('response')}")
    print(f"- Method: {result.get('method', 'voice_unlock')}")
    
    if result.get('success'):
        print("\n✅ Lock command working!")
    else:
        print("\n❌ Lock command failed")
        if 'error' in result:
            print(f"Error: {result['error']}")


async def main():
    """Run the test"""
    await test_lock_command()
    
    print("\n" + "="*50)
    print("The lock command should now work even without Voice Unlock daemon.")
    print("It will use the Context Intelligence System as a fallback.")


if __name__ == "__main__":
    asyncio.run(main())