#!/usr/bin/env python3
"""Test lock screen functionality with async pipeline"""

import asyncio
import sys
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

from system_control.macos_controller import MacOSController

async def test_lock_screen():
    """Test the lock screen command"""
    print("Testing lock screen with async pipeline...")

    controller = MacOSController()
    success, message = await controller.lock_screen()

    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"Message: {message}")

    return success

if __name__ == "__main__":
    result = asyncio.run(test_lock_screen())
    sys.exit(0 if result else 1)