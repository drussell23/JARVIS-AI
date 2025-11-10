#!/usr/bin/env python3
"""
Test Vision Performance Improvements
Measures actual response time for "can you see my screen?"
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_vision_performance():
    """Test the actual performance of vision command handling"""
    try:
        logger.info("=" * 80)
        logger.info("VISION PERFORMANCE TEST")
        logger.info("=" * 80)

        from api.vision_command_handler import VisionCommandHandler

        # Create handler
        handler = VisionCommandHandler()

        # Initialize intelligence if needed
        try:
            await handler.initialize_intelligence()
            logger.info("✅ Intelligence initialized")
        except Exception as e:
            logger.warning(f"Could not initialize intelligence: {e}")

        # Test the actual command
        test_query = "can you see my screen?"
        logger.info(f"\n📊 Testing query: '{test_query}'")
        logger.info("-" * 80)

        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                handler.handle_command(test_query),
                timeout=30.0  # Overall timeout
            )

            elapsed = time.time() - start_time

            logger.info("-" * 80)
            logger.info(f"⏱️  TOTAL TIME: {elapsed:.2f} seconds")
            logger.info("-" * 80)

            if result.get('handled'):
                response = result.get('response', '')
                logger.info(f"✅ Response received: {response[:200]}...")

                # Check performance targets
                if elapsed < 10:
                    logger.info(f"🎉 EXCELLENT! Response in {elapsed:.2f}s (target: <10s)")
                elif elapsed < 15:
                    logger.info(f"✅ GOOD! Response in {elapsed:.2f}s (target: <10s, acceptable: <15s)")
                else:
                    logger.warning(f"⚠️  SLOW! Response in {elapsed:.2f}s (target: <10s)")

                return elapsed < 15  # Pass if under 15 seconds
            else:
                logger.error(f"❌ Command not handled: {result}")
                return False

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"❌ TIMEOUT after {elapsed:.2f}s - Command should complete within 30s")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


async def main():
    """Run performance test"""
    logger.info("\n🚀 Starting Vision Performance Test\n")

    success = await test_vision_performance()

    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ PERFORMANCE TEST PASSED")
    else:
        logger.info("❌ PERFORMANCE TEST FAILED")
    logger.info("=" * 80 + "\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
