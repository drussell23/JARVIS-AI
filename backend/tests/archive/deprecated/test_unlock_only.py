#!/usr/bin/env python3
"""Simple test for unlock screen functionality without loops"""

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


async def test_unlock():
    """Test only the unlock screen functionality with timeout protection"""
    logger.info("Testing unlock screen without infinite loops...")

    controller = MacOSController()

    # Test unlock with timeout
    logger.info("Attempting to unlock screen (max 5 seconds)...")
    try:
        success, message = await asyncio.wait_for(
            controller.unlock_screen(),
            timeout=5.0
        )

        if success:
            logger.info(f"✅ Unlock SUCCESSFUL: {message}")
            return True
        else:
            logger.warning(f"⚠️ Unlock FAILED (expected if daemon not running): {message}")
            # This is expected without the daemon running
            return True  # Still return True since it handled gracefully
    except asyncio.TimeoutError:
        logger.error("❌ Unlock TIMEOUT - infinite loop detected!")
        return False
    except Exception as e:
        logger.error(f"❌ Unlock ERROR: {e}")
        return False


async def main():
    try:
        success = await test_unlock()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())