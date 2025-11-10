#!/usr/bin/env python3
"""
Test Vision Capture Timeout Fix
Tests that screen capture operations complete or timeout gracefully
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_vision_capture_timeout():
    """Test that vision capture times out properly instead of hanging"""
    try:
        logger.info("=" * 80)
        logger.info("TESTING VISION CAPTURE WITH TIMEOUT PROTECTION")
        logger.info("=" * 80)

        # Import vision command handler
        from api.vision_command_handler import VisionCommandHandler

        # Create handler
        handler = VisionCommandHandler()
        logger.info("✅ VisionCommandHandler created")

        # Test 1: Screen capture with timeout
        logger.info("\n[TEST 1] Testing _capture_screen with timeout...")
        start_time = asyncio.get_event_loop().time()

        try:
            screenshot = await asyncio.wait_for(
                handler._capture_screen(multi_space=False),
                timeout=20.0  # Overall test timeout
            )
            elapsed = asyncio.get_event_loop().time() - start_time

            if screenshot:
                logger.info(f"✅ [TEST 1] Screen capture succeeded in {elapsed:.2f}s")
            else:
                logger.info(f"⚠️  [TEST 1] Screen capture returned None in {elapsed:.2f}s (this is OK - means timeout was handled)")

        except asyncio.TimeoutError:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ [TEST 1] FAILED - Overall test timeout after {elapsed:.2f}s")
            return False

        # Test 2: Full command handling
        logger.info("\n[TEST 2] Testing full handle_command flow...")
        start_time = asyncio.get_event_loop().time()

        try:
            result = await asyncio.wait_for(
                handler.handle_command("can you see my screen"),
                timeout=30.0  # Overall command timeout
            )
            elapsed = asyncio.get_event_loop().time() - start_time

            logger.info(f"✅ [TEST 2] Command completed in {elapsed:.2f}s")
            logger.info(f"   Result: handled={result.get('handled')}, has_response={bool(result.get('response'))}")

            # Check if we got a proper error message for timeout
            if result.get('response'):
                response = result['response']
                if 'timeout' in response.lower() or 'longer than expected' in response.lower():
                    logger.info(f"✅ [TEST 2] Got appropriate timeout error message")
                else:
                    logger.info(f"✅ [TEST 2] Got response: {response[:100]}...")

        except asyncio.TimeoutError:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ [TEST 2] FAILED - Command hung for {elapsed:.2f}s (should have timed out internally)")
            return False

        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS PASSED - Timeout protection is working correctly")
        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}", exc_info=True)
        return False


async def main():
    """Run the test"""
    success = await test_vision_capture_timeout()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
