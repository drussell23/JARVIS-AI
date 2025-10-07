#!/usr/bin/env python3
"""Test script for screen lock functionality"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.system_control.macos_controller import MacOSController

async def test_lock_screen():
    """Test the screen lock functionality"""
    controller = MacOSController()

    print("Testing screen lock functionality...")
    print("WARNING: This will lock your screen!")
    print("Press Enter to continue or Ctrl+C to cancel...")
    input()

    print("\nAttempting to lock screen...")
    success, message = await controller.lock_screen()

    if success:
        print(f"✅ SUCCESS: {message}")
    else:
        print(f"❌ FAILED: {message}")

    return success

if __name__ == "__main__":
    result = asyncio.run(test_lock_screen())
    sys.exit(0 if result else 1)