#!/usr/bin/env python3
"""Test direct unlock functionality"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from api.simple_unlock_handler import handle_unlock_command


async def main():
    print("Testing direct unlock with keychain password...")
    print("=" * 60)

    # First lock the screen
    print("\n1. Locking screen...")
    lock_result = await handle_unlock_command("lock my screen")
    print(f"   Lock result: {lock_result.get('success')}")
    print(f"   Response: {lock_result.get('response')}")

    # Wait for lock to take effect
    await asyncio.sleep(3)

    # Now try to unlock
    print("\n2. Attempting unlock with keychain password...")
    unlock_result = await handle_unlock_command("unlock my screen")
    print(f"   Unlock result: {unlock_result.get('success')}")
    print(f"   Response: {unlock_result.get('response')}")
    print(f"   Method: {unlock_result.get('method', 'N/A')}")

    # Wait for unlock
    await asyncio.sleep(3)

    print("\n" + "=" * 60)
    if unlock_result.get('success'):
        print("✅ SUCCESS: Unlock worked!")
        return 0
    else:
        print("❌ FAILED: Unlock did not work")
        if 'setup_instructions' in unlock_result:
            print(f"\nSetup required: {unlock_result['setup_instructions']['command']}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
