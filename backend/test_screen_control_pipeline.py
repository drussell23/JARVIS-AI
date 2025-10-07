#!/usr/bin/env python3
"""
Test Script for Screen Lock/Unlock with Async Pipeline
========================================================

This script tests the lock and unlock functionality to ensure:
1. Commands don't get stuck in "Processing..." state
2. Async pipeline is properly integrated
3. Both lock and unlock work correctly
"""

import asyncio
import sys
import logging
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the controller
from system_control.macos_controller import MacOSController

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_lock_screen():
    """Test the lock screen command with async pipeline"""
    logger.info("\n" + "="*50)
    logger.info("TEST: Lock Screen")
    logger.info("="*50)

    controller = MacOSController()

    logger.info("Attempting to lock screen...")
    start_time = time.time()

    try:
        success, message = await controller.lock_screen()
        elapsed = time.time() - start_time

        if success:
            logger.info(f"✅ Lock SUCCESSFUL in {elapsed:.2f}s: {message}")
            return True
        else:
            logger.error(f"❌ Lock FAILED in {elapsed:.2f}s: {message}")
            return False
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ Lock ERROR in {elapsed:.2f}s: {e}")
        return False


async def test_unlock_screen():
    """Test the unlock screen command with async pipeline"""
    logger.info("\n" + "="*50)
    logger.info("TEST: Unlock Screen")
    logger.info("="*50)

    controller = MacOSController()

    logger.info("Attempting to unlock screen...")
    start_time = time.time()

    try:
        # Set a timeout to prevent infinite loops
        success, message = await asyncio.wait_for(
            controller.unlock_screen(),
            timeout=10.0  # Maximum 10 seconds for unlock
        )
        elapsed = time.time() - start_time

        if success:
            logger.info(f"✅ Unlock SUCCESSFUL in {elapsed:.2f}s: {message}")
            return True
        else:
            logger.warning(f"⚠️ Unlock FAILED in {elapsed:.2f}s: {message}")
            logger.info("This is expected if Voice Unlock daemon is not running")
            return False
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"❌ Unlock TIMEOUT after {elapsed:.2f}s - possible infinite loop detected!")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ Unlock ERROR in {elapsed:.2f}s: {e}")
        return False


async def test_screen_status():
    """Test checking screen lock status"""
    logger.info("\n" + "="*50)
    logger.info("TEST: Check Screen Lock Status")
    logger.info("="*50)

    controller = MacOSController()

    try:
        is_locked = controller._check_screen_lock_status()
        logger.info(f"Screen is currently: {'🔒 LOCKED' if is_locked else '🔓 UNLOCKED'}")
        return True
    except Exception as e:
        logger.warning(f"Could not check screen status: {e}")
        return False


async def run_all_tests():
    """Run all screen control tests"""
    logger.info("\n" + "="*70)
    logger.info("STARTING SCREEN CONTROL PIPELINE TESTS")
    logger.info("="*70)

    results = []

    # Test 1: Check status
    results.append(await test_screen_status())

    # Test 2: Lock screen
    logger.info("\n⏳ Testing lock screen (should complete in < 5 seconds)...")
    lock_result = await test_lock_screen()
    results.append(lock_result)

    if lock_result:
        logger.info("\n⏰ Waiting 3 seconds before unlock test...")
        await asyncio.sleep(3)

        # Test 3: Unlock screen
        logger.info("\n⏳ Testing unlock screen (may take up to 20 seconds)...")
        results.append(await test_unlock_screen())

    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)

    total = len(results)
    passed = sum(1 for r in results if r)

    logger.info(f"Tests Passed: {passed}/{total}")

    if passed == total:
        logger.info("✅ All tests passed!")
    elif passed > 0:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
    else:
        logger.error("❌ All tests failed")

    return passed == total


async def test_pipeline_integration():
    """Test that the async pipeline is actually being used"""
    logger.info("\n" + "="*70)
    logger.info("TESTING ASYNC PIPELINE INTEGRATION")
    logger.info("="*70)

    from core.async_pipeline import get_async_pipeline

    # Get the pipeline instance
    pipeline = get_async_pipeline()

    # Check it's the AdvancedAsyncPipeline
    logger.info(f"Pipeline Class: {pipeline.__class__.__name__}")

    if pipeline.__class__.__name__ == "AdvancedAsyncPipeline":
        logger.info("✅ Using AdvancedAsyncPipeline (correct!)")
    else:
        logger.error("❌ Not using AdvancedAsyncPipeline!")
        return False

    # Check registered stages
    stages = list(pipeline.stages.keys())
    logger.info(f"\nRegistered Pipeline Stages: {len(stages)}")

    # Check for screen control stages
    screen_stages = [s for s in stages if 'screen' in s.lower() or 'lock' in s.lower() or 'unlock' in s.lower()]
    if screen_stages:
        logger.info(f"✅ Found screen control stages: {screen_stages}")
    else:
        logger.warning("⚠️ No screen control specific stages found")

    return True


async def main():
    """Main test runner"""
    try:
        # Test pipeline integration first
        await test_pipeline_integration()

        # Run all screen control tests
        success = await run_all_tests()

        # Exit code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    logger.info("Screen Control Pipeline Test Script")
    logger.info("This will test lock/unlock functionality with async pipeline")
    logger.info("Press Ctrl+C to cancel\n")

    asyncio.run(main())