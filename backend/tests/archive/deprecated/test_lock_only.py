#!/usr/bin/env python3
"""Simple test for lock screen functionality with async pipeline"""

import asyncio
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from system_control.macos_controller import MacOSController

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_lock():
    """Test only the lock screen functionality"""
    logger.info("Testing lock screen with async pipeline...")

    controller = MacOSController()

    # Test lock
    logger.info("Attempting to lock screen...")
    success, message = await controller.lock_screen()

    if success:
        logger.info(f"✅ Lock SUCCESSFUL: {message}")
        return True
    else:
        logger.error(f"❌ Lock FAILED: {message}")
        return False


async def main():
    try:
        success = await test_lock()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())