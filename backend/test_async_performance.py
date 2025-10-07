#!/usr/bin/env python3
"""
Test to demonstrate performance improvement of async vs sync methods
"""

import asyncio
import sys
import logging
import time
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


async def test_async_volume():
    """Test async volume control"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Async Volume Control (Non-blocking)")
    logger.info("="*70)

    controller = MacOSController()

    # Test multiple volume commands in parallel
    start_time = time.time()

    tasks = [
        controller.set_volume_async(50),
        controller.set_volume_async(60),
        controller.set_volume_async(70),
    ]

    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    logger.info(f"✅ Completed 3 volume changes in {elapsed:.3f}s (parallel execution)")
    return elapsed


def test_sync_volume():
    """Test sync volume control"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Sync Volume Control (Blocking)")
    logger.info("="*70)

    controller = MacOSController()

    # Test multiple volume commands sequentially
    start_time = time.time()

    controller.set_volume(50)
    controller.set_volume(60)
    controller.set_volume(70)

    elapsed = time.time() - start_time

    logger.info(f"⏱️  Completed 3 volume changes in {elapsed:.3f}s (sequential execution)")
    return elapsed


async def main():
    """Compare async vs sync performance"""
    try:
        logger.info("\n" + "="*70)
        logger.info("PERFORMANCE COMPARISON: Async vs Sync")
        logger.info("="*70)

        # Test async version
        async_time = await test_async_volume()

        # Wait a bit between tests
        await asyncio.sleep(1)

        # Test sync version
        sync_time = test_sync_volume()

        # Results
        logger.info("\n" + "="*70)
        logger.info("RESULTS")
        logger.info("="*70)
        logger.info(f"Async version: {async_time:.3f}s")
        logger.info(f"Sync version:  {sync_time:.3f}s")

        if async_time < sync_time:
            speedup = (sync_time / async_time - 1) * 100
            logger.info(f"\n🚀 Async is {speedup:.1f}% FASTER!")
            logger.info("This is because async methods can run operations in parallel,")
            logger.info("while sync methods must wait for each operation to complete.")
        else:
            logger.info("\n⚠️ Results may vary due to system load")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    logger.info("Async vs Sync Performance Test")
    logger.info("Testing volume control operations\n")

    asyncio.run(main())